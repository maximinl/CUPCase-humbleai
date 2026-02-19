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
deepseek_client = AsyncOpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

sem = asyncio.Semaphore(5)

def check_semantic_agreement(diag1, diag2):
    if not diag1 or not diag2:
        return False
    d1, d2 = diag1.lower().strip(), diag2.lower().strip()
    if d1 == d2 or d1 in d2 or d2 in d1:
        return True
    stopwords = {'with', 'and', 'the', 'of', 'in', 'a', 'an', 'due', 'to', 'secondary', 'primary'}
    words1 = set(d1.split()) - stopwords
    words2 = set(d2.split()) - stopwords
    if words1 and words2:
        return len(words1 & words2) / min(len(words1), len(words2)) >= 0.5
    return False

async def get_candidates(case_text):
    prompt = f"CASE: {case_text}\n\nTASK: Provide the single most likely diagnosis. Be concise.\nDIAGNOSIS:"
    async with sem:
        try:
            tasks = [
                openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0),
                deepseek_client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            return {"gpt4o": "Error", "deepseek": "Error", "agreement": False, "candidates": [], "error": str(e)}
    
    gpt4o = results[0].choices[0].message.content.strip() if not isinstance(results[0], Exception) else "Error"
    deepseek = results[1].choices[0].message.content.strip() if not isinstance(results[1], Exception) else "Error"
    agreement = check_semantic_agreement(gpt4o, deepseek)
    
    # Build candidates list for Stage 2 handoff
    if agreement:
        # Use longer/more detailed response as consensus
        consensus = gpt4o if len(gpt4o) >= len(deepseek) else deepseek
        candidates = [consensus]
    else:
        candidates = [gpt4o, deepseek]
    
    return {"gpt4o": gpt4o, "deepseek": deepseek, "agreement": agreement, "candidates": candidates, "error": None}

async def process_case(case_id, row):
    case_text = row.get('clean text') or row.get('100%') or ""
    gold = row.get('final diagnosis', 'Unknown')
    result = await get_candidates(case_text)
    
    # Stage 1 final: Use consensus if agreed, else GPT-4o as tiebreaker
    # Note: Real disambiguation happens in Stage 2 audit
    if result["agreement"]:
        final = result["candidates"][0]  # Consensus diagnosis
    else:
        final = result["gpt4o"]  # Fallback to GPT-4o; Stage 2 will audit both
    
    return {
        'case_id': case_id,
        'gold': gold,
        'gpt4o_diagnosis': result["gpt4o"],
        'deepseek_diagnosis': result["deepseek"],
        'model_agreement': result["agreement"],
        'candidates': json.dumps(result["candidates"]),  # Unified handoff for Stage 2
        'final_diagnosis': final,
        'error': result.get('error')
    }

async def main(args):
    df = pd.read_csv(args.data_path)
    if args.samples:
        df = df.sample(n=min(args.samples, len(df)), random_state=args.seed)
    
    print(f"Stage 1: Processing {len(df)} cases (Ensemble only, NO audit)...")
    
    tasks = [process_case(idx, row) for idx, row in df.iterrows()]
    results = []
    async for f in async_tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await f)
    
    final_df = pd.DataFrame(results).sort_values('case_id').reset_index(drop=True)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/ensemble_v2_results_{len(df)}.csv"
    final_df.to_csv(output_file, index=False)
    
    agreed = final_df['model_agreement'].sum()
    print(f"\nSTAGE 1 RESULTS:")
    print(f"  Total: {len(final_df)}")
    print(f"  Agreement: {agreed} ({agreed/len(final_df)*100:.1f}%)")
    print(f"  Disagreement: {len(final_df)-agreed} ({(len(final_df)-agreed)/len(final_df)*100:.1f}%)")
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--samples', type=int, default=350)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args))
