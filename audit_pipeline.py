import os
import json
import pandas as pd
import re
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
import nest_asyncio

nest_asyncio.apply()

openai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
sem = asyncio.Semaphore(5)

def clean_json_string(s):
    s = re.sub(r"```json\s*|\s*```", "", s, flags=re.I)
    start, end = s.find('{'), s.rfind('}')
    return s[start:end+1] if start != -1 and end != -1 else s.strip()

async def perform_audit(case_text, candidates):
    """Stage 2: For/against reasoning on candidates."""
    candidate_str = "\n".join([f"- {c}" for c in candidates])
    
    # Adjust prompt based on number of candidates
    if len(candidates) == 1:
        task_desc = "Critically evaluate this diagnosis. Consider what could be wrong with it."
    else:
        task_desc = "Perform a 'For and Against' audit for each diagnosis, then select the most likely."
    
    prompt = f"""CASE: {case_text}

CANDIDATE DIAGNOSES:
{candidate_str}

TASK: {task_desc}

Respond ONLY with valid JSON:
{{
    "audit": [{{"diagnosis": "...", "evidence_for": ["..."], "evidence_against": ["..."]}}],
    "final_decision": "most likely diagnosis",
    "confidence": 0.0-1.0,
    "rationale": "brief explanation"
}}"""
    
    async with sem:
        try:
            res = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(clean_json_string(res.choices[0].message.content))
        except Exception as e:
            return {"final_decision": "Error", "confidence": 0, "rationale": str(e), "audit": []}

async def process_case(case_id, row):
    case_text = row.get('clean text') or row.get('100%') or ""
    gold = row.get('final diagnosis', 'Unknown')
    
    # Get candidates from Stage 1 output or columns
    if 'candidates' in row and pd.notna(row['candidates']):
        candidates = json.loads(row['candidates'])
    elif 'gpt4o_diagnosis' in row and 'deepseek_diagnosis' in row:
        # Reconstruct from separate columns
        candidates = []
        if pd.notna(row.get('gpt4o_diagnosis')):
            candidates.append(row['gpt4o_diagnosis'])
        if pd.notna(row.get('deepseek_diagnosis')) and row.get('deepseek_diagnosis') != row.get('gpt4o_diagnosis'):
            candidates.append(row['deepseek_diagnosis'])
        if not candidates:
            candidates = ["Unknown"]
    else:
        # No Stage 1 data - this shouldn't happen in normal pipeline
        print(f"Warning: No candidates for case {case_id}, skipping audit")
        return {
            'case_id': case_id,
            'gold': gold,
            'candidates': "[]",
            'final_diagnosis': "Error: No candidates",
            'audit_confidence': 0,
            'audit_rationale': "No candidates provided from Stage 1",
            'audit_details': "[]"
        }
    
    audit = await perform_audit(case_text, candidates)
    
    return {
        'case_id': case_id,
        'gold': gold,
        'candidates': json.dumps(candidates),
        'num_candidates': len(candidates),
        'final_diagnosis': audit.get('final_decision', 'Unknown'),
        'audit_confidence': audit.get('confidence', 0),
        'audit_rationale': audit.get('rationale', ''),
        'audit_details': json.dumps(audit.get('audit', []))
    }

async def main(args):
    # Load Stage 1 results if provided, else raw data
    if args.ensemble_results:
        print(f"Loading Stage 1 results from: {args.ensemble_results}")
        df = pd.read_csv(args.ensemble_results)
    else:
        print(f"Warning: No --ensemble-results provided. Stage 2 needs Stage 1 output.")
        print(f"Loading raw data from: {args.data_path}")
        df = pd.read_csv(args.data_path)
    
    if args.samples and args.samples < len(df):
        df = df.sample(n=args.samples, random_state=args.seed)
    
    print(f"Stage 2: Processing {len(df)} cases (Audit)...")
    
    tasks = [process_case(row.get('case_id', idx), row) for idx, row in df.iterrows()]
    results = []
    async for f in async_tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await f)
    
    final_df = pd.DataFrame(results).sort_values('case_id').reset_index(drop=True)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/audit_results_{len(df)}.csv"
    final_df.to_csv(output_file, index=False)
    
    multi = (final_df['num_candidates'] > 1).sum()
    print(f"\nSTAGE 2 RESULTS:")
    print(f"  Total: {len(final_df)}")
    print(f"  Cases with multiple candidates (real audit): {multi}")
    print(f"  Cases with single candidate: {len(final_df) - multi}")
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None, help='Raw data path (fallback)')
    parser.add_argument('--ensemble-results', default=None, help='Path to Stage 1 CSV output')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if not args.ensemble_results and not args.data_path:
        raise ValueError("Must provide either --ensemble-results or --data-path")
    
    asyncio.run(main(args))
