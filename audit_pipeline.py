import os
import json
import pandas as pd
import re
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import nest_asyncio

nest_asyncio.apply()

openai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
sem = asyncio.Semaphore(5)

def clean_json_string(s):
    s = re.sub(r"```json\s*|\s*```", "", s, flags=re.I)
    start, end = s.find('{'), s.rfind('}')
    return s[start:end+1] if start != -1 and end != -1 else s.strip()

async def perform_audit(case_text, candidates):
    candidate_str = "\n".join([f"- {c}" for c in candidates])
    prompt = f"""CASE: {case_text}

DIFFERENTIAL DIAGNOSES:
{candidate_str}

TASK: Perform a 'For and Against' audit for each diagnosis. Then select the most likely.

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
            return {"final_decision": "Error", "confidence": 0, "rationale": str(e)}

async def process_case(case_id, row):
    case_text = row.get('clean text') or row.get('100%') or ""
    gold = row.get('final diagnosis', 'Unknown')
    
    # For Stage 2 standalone: just use GPT-4o as single candidate
    candidates = [row.get('gpt4o_diagnosis')] if 'gpt4o_diagnosis' in row else []
    if not candidates or not candidates[0]:
        # Generate candidate if not provided
        prompt = f"CASE: {case_text}\n\nTASK: Provide the single most likely diagnosis.\nDIAGNOSIS:"
        async with sem:
            res = await openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0)
            candidates = [res.choices[0].message.content.strip()]
    
    audit = await perform_audit(case_text, candidates)
    
    return {
        'case_id': case_id,
        'gold': gold,
        'candidates': json.dumps(candidates),
        'final_diagnosis': audit.get('final_decision', 'Unknown'),
        'audit_confidence': audit.get('confidence', 0),
        'audit_rationale': audit.get('rationale', ''),
        'audit_details': json.dumps(audit.get('audit', []))
    }

async def main(args):
    df = pd.read_csv(args.data_path)
    if args.samples:
        df = df.sample(n=min(args.samples, len(df)), random_state=args.seed)
    
    print(f"Stage 2: Processing {len(df)} cases (Audit only)...")
    tasks = [process_case(idx, row) for idx, row in df.iterrows()]
    results = [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]
    
    final_df = pd.DataFrame(results).sort_values('case_id').reset_index(drop=True)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/audit_results_{len(df)}.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\nSTAGE 2 RESULTS: {len(final_df)} cases")
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--samples', type=int, default=350)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args))
