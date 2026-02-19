import os
import json
import pandas as pd
import re
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import nest_asyncio

# Apply nest_asyncio
nest_asyncio.apply()

# Initialize Async Clients
openai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
deepseek_client = AsyncOpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# Limit concurrency
sem = asyncio.Semaphore(5)

def clean_json_string(s):
    s = re.sub(r"```json\s*|\s*```", "", s, flags=re.I)
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1:
        return s[start:end+1]
    return s.strip()

def check_semantic_agreement(diag1, diag2):
    if not diag1 or not diag2:
        return False
    d1 = diag1.lower().strip()
    d2 = diag2.lower().strip()
    if d1 == d2:
        return True
    if d1 in d2 or d2 in d1:
        return True
    words1 = set(d1.split())
    words2 = set(d2.split())
    stopwords = {'with', 'and', 'the', 'of', 'in', 'a', 'an', 'due', 'to', 'secondary', 'primary'}
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    if len(words1) > 0 and len(words2) > 0:
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        if overlap >= 0.5:
            return True
    return False

async def get_candidates(case_text):
    """Stage 1: Get candidates from multiple models with consensus tracking."""
    prompt = f"CASE: {case_text}\n\nTASK: Provide the single most likely diagnosis. Be concise.\nDIAGNOSIS:"
    async with sem:
        try:
            tasks = [
                openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0),
                deepseek_client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            return {
                "candidates": [],
                "candidates_with_votes": [],
                "gpt4o_response": "Error",
                "deepseek_response": "Error",
                "agreement": False,
                "consensus_diagnosis": None,
                "error": str(e)
            }
    
    gpt4o_response = ""
    deepseek_response = ""
    
    if not isinstance(results[0], Exception):
        gpt4o_response = results[0].choices[0].message.content.strip()
    if not isinstance(results[1], Exception):
        deepseek_response = results[1].choices[0].message.content.strip()
    
    agreement = check_semantic_agreement(gpt4o_response, deepseek_response)
    
    if agreement:
        consensus = gpt4o_response if len(gpt4o_response) >= len(deepseek_response) else deepseek_response
        candidates_with_votes = [{"diagnosis": consensus, "votes": 2, "sources": ["gpt4o", "deepseek"]}]
        candidates = [consensus]
    else:
        candidates_with_votes = []
        candidates = []
        if gpt4o_response:
            candidates_with_votes.append({"diagnosis": gpt4o_response, "votes": 1, "sources": ["gpt4o"]})
            candidates.append(gpt4o_response)
        if deepseek_response:
            candidates_with_votes.append({"diagnosis": deepseek_response, "votes": 1, "sources": ["deepseek"]})
            candidates.append(deepseek_response)
    
    return {
        "candidates": candidates,
        "candidates_with_votes": candidates_with_votes,
        "gpt4o_response": gpt4o_response,
        "deepseek_response": deepseek_response,
        "agreement": agreement,
        "consensus_diagnosis": candidates[0] if agreement else None,
        "error": None
    }

async def perform_audit(case_text, candidates_with_votes):
    """Stage 2: Audit with consensus info passed to auditor."""
    candidate_lines = []
    for c in candidates_with_votes:
        vote_info = f"[{c['votes']} model(s): {', '.join(c['sources'])}]"
        candidate_lines.append(f"- {c['diagnosis']} {vote_info}")
    candidate_str = "\n".join(candidate_lines)
    
    prompt = f"""CASE: {case_text}

DIFFERENTIAL DIAGNOSES (with model consensus info):
{candidate_str}

TASK: Perform a 'For and Against' audit for each candidate diagnosis.
- Consider the clinical evidence that supports and contradicts each option.
- Note: Diagnoses with 2 votes have consensus from both models, which may indicate higher confidence.
- Select the most likely final diagnosis.

Respond ONLY with valid JSON matching this schema:
{{
    "audit": [
        {{
            "diagnosis": "...",
            "model_votes": 1 or 2,
            "evidence_for": ["point 1", "point 2"],
            "evidence_against": ["point 1", "point 2"]
        }}
    ],
    "final_decision": "the most likely diagnosis",
    "confidence": 0.0 to 1.0,
    "rationale": "brief explanation",
    "consensus_influenced": true/false
}}"""
    
    async with sem:
        try:
            res = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            raw_content = res.choices[0].message.content
            parsed = json.loads(clean_json_string(raw_content))
            parsed["error"] = None
            return parsed
        except Exception as e:
            return {
                "audit": [],
                "final_decision": "Error",
                "confidence": 0,
                "rationale": "API Failure",
                "consensus_influenced": False,
                "error": str(e)
            }

async def process_case(case_id, row):
    """Process single case through full pipeline."""
    case_text = row.get('clean text') or row.get('100%') or ""
    gold = row.get('final diagnosis', 'Unknown')
    
    stage1 = await get_candidates(case_text)
    candidates_with_votes = stage1["candidates_with_votes"]
    
    stage2 = await perform_audit(case_text, candidates_with_votes)
    pred = stage2.get('final_decision', "Unknown")
    
    return {
        'case_id': case_id,
        'gold': gold,
        'gpt4o_diagnosis': stage1["gpt4o_response"],
        'deepseek_diagnosis': stage1["deepseek_response"],
        'model_agreement': stage1["agreement"],
        'consensus_diagnosis': stage1["consensus_diagnosis"],
        'candidates_with_votes': json.dumps(candidates_with_votes),
        'final_diagnosis': pred,
        'audit_confidence': stage2.get('confidence', 0),
        'audit_rationale': stage2.get('rationale', ''),
        'consensus_influenced': stage2.get('consensus_influenced', False),
        'audit_details': json.dumps(stage2.get('audit', [])),
        'stage1_error': stage1.get('error'),
        'stage2_error': stage2.get('error')
    }

async def main(args):
    df = pd.read_csv(args.data_path)
    if args.samples:
        df = df.sample(n=min(args.samples, len(df)), random_state=args.seed)
    
    print(f"Processing {len(df)} cases...")
    
    tasks = [process_case(idx, row) for idx, row in df.iterrows()]
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await f)
    
    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values('case_id').reset_index(drop=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/hybrid_v3_results_{len(df)}.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {len(final_df)}")
    
    agreed = final_df[final_df['model_agreement'] == True]
    disagreed = final_df[final_df['model_agreement'] == False]
    print(f"\nModel Agreement:")
    print(f"  Agreed: {len(agreed)} ({len(agreed)/len(final_df)*100:.1f}%)")
    print(f"  Disagreed: {len(disagreed)} ({len(disagreed)/len(final_df)*100:.1f}%)")
    
    consensus_used = final_df[final_df['consensus_influenced'] == True]
    print(f"\nConsensus influenced final decision: {len(consensus_used)} cases")
    
    print(f"\nSaved to: {output_file}")
    
    return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--samples', type=int, default=350)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args))
