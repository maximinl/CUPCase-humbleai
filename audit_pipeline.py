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


def _build_legacy_audit_prompt(case_text, candidates):
    candidate_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    return f"""CASE: {case_text}

CANDIDATE DIAGNOSES:
{candidate_str}

TASK: For each candidate diagnosis, list the clinical evidence FOR and AGAINST it based on the case details. Then select the best-supported candidate as your final decision.

CRITICAL RULES:
- You MUST select your final_decision from the CANDIDATE DIAGNOSES listed above.
- Do NOT invent a new diagnosis. Only choose from the numbered candidates.
- If there is only one candidate, confirm or reject it — but if rejecting, still return that candidate as final_decision.

Respond ONLY with valid JSON:
{{
    "audit": [{{"diagnosis": "<candidate diagnosis>", "evidence_for": ["point1", "point2"], "evidence_against": ["point1", "point2"]}}],
    "final_decision": "<one of the candidate diagnoses above, copied exactly>",
    "confidence": <0.0 to 1.0>,
    "rationale": "<brief explanation>"
}}"""


def _build_differential_audit_prompt(case_text, candidates):
    candidate_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    return f"""CASE: {case_text}

INITIAL CANDIDATE DIAGNOSES:
{candidate_str}

TASK: Perform a differential diagnosis audit.

STEP 1 — COUNTER-HYPOTHESES: For each candidate above, propose 2 alternative diagnoses that could also explain this clinical presentation.

STEP 2 — COMPARATIVE EVALUATION: For ALL diagnoses (initial candidates + your alternatives), evaluate:
  a) Which clinical findings SUPPORT this diagnosis?
  b) Which clinical findings ARGUE AGAINST this diagnosis?
  c) Are there expected findings for this diagnosis that are ABSENT from the case?

STEP 3 — FINAL DECISION: Select the single best-supported diagnosis from ALL considered options (initial candidates OR your alternatives). You are NOT limited to the initial candidates.

Respond ONLY with valid JSON:
{{
    "counter_hypotheses": [
        {{"original": "<candidate>", "alternatives": ["<alt1>", "<alt2>"]}}
    ],
    "evaluation": [
        {{"diagnosis": "<name>", "evidence_for": ["..."], "evidence_against": ["..."], "missing_findings": ["..."]}}
    ],
    "final_decision": "<best-supported diagnosis>",
    "confidence": <0.0 to 1.0>,
    "rationale": "<why this diagnosis is best supported compared to alternatives>"
}}"""


async def perform_audit(case_text, candidates, mode="legacy"):
    """Stage 2 audit in legacy or differential mode."""
    prompt = (
        _build_differential_audit_prompt(case_text, candidates)
        if mode == "differential"
        else _build_legacy_audit_prompt(case_text, candidates)
    )
    system_prompt = (
        "You are a medical differential diagnosis expert. Systematically evaluate all diagnostic possibilities. Respond with ONLY valid JSON."
        if mode == "differential"
        else "You are a medical audit assistant. You MUST respond with ONLY valid JSON. No markdown, no explanation, no preamble. Just the JSON object."
    )

    async with sem:
        try:
            res = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            parsed = json.loads(clean_json_string(res.choices[0].message.content))

            final = parsed.get('final_decision', 'Unknown')
            if final.lower() in ['most likely diagnosis', 'unknown', '']:
                final = candidates[0] if candidates else 'Unknown'
                parsed['final_decision'] = final

            return parsed
        except Exception as e:
            empty_key = "evaluation" if mode == "differential" else "audit"
            return {"final_decision": candidates[0] if candidates else "Error", "confidence": 0, "rationale": str(e), empty_key: []}


async def process_case(case_id, row, audit_mode="legacy"):
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

    audit = await perform_audit(case_text, candidates, mode=audit_mode)

    return {
        'case_id': case_id,
        'gold': gold,
        'candidates': json.dumps(candidates),
        'num_candidates': len(candidates),
        'final_diagnosis': audit.get('final_decision', 'Unknown'),
        'audit_confidence': audit.get('confidence', 0),
        'audit_rationale': audit.get('rationale', ''),
        'audit_details': json.dumps(audit.get('audit', audit.get('evaluation', []))),
        'counter_hypotheses': json.dumps(audit.get('counter_hypotheses', [])),
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
    
    tasks = [process_case(row.get('case_id', idx), row, audit_mode=args.audit_mode) for idx, row in df.iterrows()]
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
    parser.add_argument('--audit-mode', type=str, default='legacy',
                        choices=['legacy', 'differential'])
    args = parser.parse_args()
    
    if not args.ensemble_results and not args.data_path:
        raise ValueError("Must provide either --ensemble-results or --data-path")
    
    asyncio.run(main(args))
