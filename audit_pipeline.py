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
    
    if len(candidates) == 1:
        task_desc = "Critically evaluate this diagnosis. Consider what could be wrong with it. Then provide your final diagnosis."
    else:
        task_desc = "Perform a 'For and Against' audit for each diagnosis, then select the most likely one."
    
    prompt = f"""CASE: {case_text}

CANDIDATE DIAGNOSES:
{candidate_str}

TASK: {task_desc}

Respond ONLY with valid JSON:
{{
    "audit": [{{"diagnosis": "<diagnosis name>", "evidence_for": ["point1", "point2"], "evidence_against": ["point1", "point2"]}}],
    "final_decision": "<YOUR CHOSEN DIAGNOSIS - must be an actual medical diagnosis, NOT a placeholder>",
    "confidence": <0.0 to 1.0>,
    "rationale": "<brief explanation>"
}}

IMPORTANT: "final_decision" must contain the actual diagnosis name (e.g., "Pneumonia", "Type 2 Diabetes"), NOT placeholder text like "most likely diagnosis"."""

    async with sem:
        try:
            res = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            parsed = json.loads(clean_json_string(res.choices[0].message.content))
            
            # Fallback if model still returns placeholder
            final = parsed.get('final_decision', 'Unknown')
            if final.lower() in ['most likely diagnosis', 'unknown', '']:
                # Use first candidate as fallback
                final = candidates[0] if candidates else 'Unknown'
                parsed['final_decision'] = final
            
            return parsed
        except Exception as e:
            return {"final_decision": candidates[0] if candidates else "Error", "confidence": 0, "rationale": str(e), "audit": []}

async def process_case(case_id, row):
    case_text = row.get('clean text') or row.get('100%') or ""
    gold = row.get('final diagnosis') or row.get('gold') or 'Unknown'
    
    if 'candidates' in row and pd.notna(row['candidates']):
        candidates = json.loads(row['candidates'])
    elif 'gpt4o_diagnosis' in row and 'deepseek_diagnosis' in row:
        candidates = []
        if pd.notna(row.get('gpt4o_diagnosis')):
            candidates.append(row['gpt4o_diagnosis'])
        if pd.notna(row.get('deepseek_diagnosis')) and row.get('deepseek_diagnosis') != row.get('gpt4o_diagnosis'):
            candidates.append(row['deepseek_diagnosis'])
        if not candidates:
            candidates = ["Unknown"]
    else:
        candidates = ["Unknown"]
    
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
    if args.ensemble_results:
        print(f"Loading Stage 1 results from: {args.ensemble_results}")
        df = pd.read_csv(args.ensemble_results)
    else:
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
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--ensemble-results', default=None)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if not args.ensemble_results and not args.data_path:
        raise ValueError("Must provide either --ensemble-results or --data-path")
    
    asyncio.run(main(args))
