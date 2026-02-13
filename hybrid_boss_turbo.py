import os, json, pandas as pd, re, asyncio, argparse
from openai import AsyncOpenAI
import requests
from tqdm.asyncio import tqdm
import nest_asyncio

# Apply nest_asyncio to allow nested event loops in Colab
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
    """Extracts the first JSON-like block from a string."""
    s = re.sub(r"```json\s*|\s*```", "", s, flags=re.I)
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1:
        return s[start:end+1]
    return s.strip()

async def get_candidates(case_text):
    prompt = f"CASE: {case_text}\n\nTASK: Provide the single most likely diagnosis. Be concise.\nDIAGNOSIS:"
    async with sem:
        try:
            tasks = [
                openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0),
                deepseek_client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            return ["Error"]
    
    candidates = []
    for r in results:
        if not isinstance(r, Exception):
            candidates.append(r.choices[0].message.content.strip())
    return list(set(candidates))

async def perform_audit(case_text, candidates):
    candidate_str = "\n".join([f"- {c}" for c in candidates])
    prompt = f"CASE: {case_text}\n\nDIFFERENTIAL:\n{candidate_str}\n\nTASK: Perform a 'For and Against' audit. Respond ONLY with valid JSON: {{\"final_decision\": \"...\", \"rationale\": \"...\"}}"
    
    async with sem:
        try:
            res = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            raw_content = res.choices[0].message.content
            return json.loads(clean_json_string(raw_content))
        except Exception:
            return {"final_decision": "Error", "rationale": "API Failure"}

async def process_case(row):
    case_text = row.get('clean text') or row.get('100%') or ""
    gold = row.get('final diagnosis', 'Unknown')
    
    candidates = await get_candidates(case_text)
    audit = await perform_audit(case_text, candidates)
    pred = audit.get('final_decision', "Unknown")
    
    return {'gold': gold, 'pred': pred, 'audit_rationale': audit.get('rationale', '')}

async def main(args):
    # Load Data
    df = pd.read_csv(args.data_path)
    if args.samples:
        df = df.sample(n=args.samples, random_state=args.seed)
        
    tasks = [process_case(row) for _, row in df.iterrows()]
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await f)
    
    # Save Results
    final_df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/turbo_results_{args.samples}.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--samples', type=int, default=350)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args))
