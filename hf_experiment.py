"""
Full diagnostic pipeline with local HuggingFace models (Qwen3.5).
Runs: Baseline, Ensemble, Audit, Hybrid — all with Qwen3.5 models.
Judge: Qwen3.5-9B for clinical equivalence evaluation.

Usage:
    python hf_experiment.py --model-main Qwen/Qwen3.5-27B --model-small Qwen/Qwen3.5-9B --samples 10
"""

import os
import json
import re
import asyncio
import argparse
import time
import pandas as pd
from tqdm import tqdm
import nest_asyncio

nest_asyncio.apply()

from hf_client import HFClient

sem = asyncio.Semaphore(1)


def clean_json_string(s):
    s = re.sub(r"```json\s*|\s*```", "", s, flags=re.I)
    start, end = s.find('{'), s.rfind('}')
    return s[start:end + 1] if start != -1 and end != -1 else s.strip()


# ---------------------------------------------------------------------------
# Semantic agreement check (from ensemble_v2.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Diagnosis generation
# ---------------------------------------------------------------------------

def extract_diagnosis(text):
    """Extract a concise diagnosis from potentially verbose model output."""
    if not text or text.startswith("Error"):
        return text

    # If the response is already short (< 80 chars), assume it's a diagnosis
    stripped = text.strip().strip('"').strip("'").strip(".")
    if len(stripped) < 80 and '\n' not in stripped:
        return stripped

    # Try to find explicit diagnosis markers in the text
    patterns = [
        r'(?:final|most likely|primary)\s*(?:diagnosis|dx)\s*(?:is|:)\s*\**\s*([^\n\*\.]+)',
        r'\*\*(?:Final |Most Likely )?Diagnosis\s*(?::|is)\s*\**\s*([^\n\*]+)',
        r'(?:^|\n)\s*(?:Diagnosis|DIAGNOSIS)\s*:\s*\**\s*([^\n\*]+)',
        r'(?:the (?:most likely|final|primary) diagnosis is)\s*\**\s*([^\n\*\.]+)',
        r'(?:I (?:would|will) (?:diagnose|conclude))\s*.*?\s*\**\s*(?:as |with |is )?\**\s*([^\n\*\.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            diag = m.group(1).strip().strip('"').strip("'").strip("*").strip()
            if 5 < len(diag) < 120:
                return diag

    # Fallback: look for bolded text near the end (common Qwen pattern)
    bold_matches = re.findall(r'\*\*([^*]{5,80})\*\*', text)
    if bold_matches:
        # Return the last bolded phrase — usually the conclusion
        return bold_matches[-1].strip()

    # Last resort: return first non-empty line that looks like a diagnosis
    for line in text.split('\n'):
        line = line.strip().strip('*').strip('-').strip()
        if 5 < len(line) < 100 and not line.lower().startswith(('the user', 'case', 'patient', 'this', 'based on', 'i ')):
            return line

    # Ultimate fallback: return truncated original
    return stripped[:100]


def get_diagnosis(client, case_text):
    prompt = f"""CASE: {case_text}

TASK: Provide the single most likely diagnosis. Respond with ONLY the diagnosis name — no explanation, no reasoning, no bullet points. Just the condition name.

DIAGNOSIS:"""
    try:
        res = client.completion([
            {"role": "system", "content": "You are a medical diagnosis assistant. When asked for a diagnosis, respond with ONLY the diagnosis name. No explanations, no reasoning, no analysis. Just the condition name."},
            {"role": "user", "content": prompt},
        ])
        return extract_diagnosis(res.strip())
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Audit (For/Against reasoning)
# ---------------------------------------------------------------------------

def perform_audit(client, case_text, candidates):
    candidate_str = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])

    prompt = f"""CASE: {case_text}

CANDIDATE DIAGNOSES:
{candidate_str}

TASK: For each candidate diagnosis, list the clinical evidence FOR and AGAINST it based on the case details. Then select the best-supported candidate as your final decision.

CRITICAL RULES:
- You MUST select your final_decision from the CANDIDATE DIAGNOSES listed above.
- Do NOT invent a new diagnosis. Only choose from the numbered candidates.
- If there is only one candidate, confirm or reject it — but if rejecting, still return that candidate as final_decision.

Respond ONLY with valid JSON (no markdown, no explanation before or after):
{{
    "audit": [{{"diagnosis": "<candidate diagnosis>", "evidence_for": ["point1", "point2"], "evidence_against": ["point1", "point2"]}}],
    "final_decision": "<one of the candidate diagnoses above, copied exactly>",
    "confidence": <0.0 to 1.0>,
    "rationale": "<brief explanation>"
}}"""

    raw = None
    try:
        raw = client.completion([
            {"role": "system", "content": "You are a medical audit assistant. You MUST respond with ONLY valid JSON. No markdown, no explanation, no preamble. Just the JSON object."},
            {"role": "user", "content": prompt},
        ])
        parsed = json.loads(clean_json_string(raw))
        final = parsed.get('final_decision', 'Unknown')
        if final.lower() in ['most likely diagnosis', 'unknown', '']:
            final = candidates[0] if candidates else 'Unknown'
            parsed['final_decision'] = final
        return parsed
    except Exception:
        # JSON parsing failed — try to extract the best candidate from raw text
        raw_lower = raw.lower() if raw else ""
        for c in candidates:
            if c.lower() in raw_lower:
                return {
                    "final_decision": c,
                    "confidence": 0.5,
                    "rationale": "Extracted from non-JSON response",
                    "audit": [],
                }
        return {
            "final_decision": candidates[0] if candidates else "Error",
            "confidence": 0,
            "rationale": "Failed to parse JSON response",
            "audit": [],
        }


# ---------------------------------------------------------------------------
# HF Judge (replaces DeepSeek judge)
# ---------------------------------------------------------------------------

def hf_judge(judge_client, pred, gold):
    """Judge clinical equivalence using local HF model."""
    if not pred or not gold or pred == "Error":
        return False

    # Quick string-based check first (avoids LLM call for obvious matches)
    pred_lower = pred.lower().strip()
    gold_lower = gold.lower().strip()
    if pred_lower == gold_lower or pred_lower in gold_lower or gold_lower in pred_lower:
        return True

    prompt = f"""GOLD DIAGNOSIS: {gold}
PREDICTED DIAGNOSIS: {pred}

Is the predicted diagnosis the SAME CONDITION as the gold diagnosis?

Rules:
- CORRECT: Same disease/condition, even if wording differs (e.g., "heart attack" = "myocardial infarction")
- CORRECT: Same core diagnosis with additional details (e.g., "pneumonia" vs "bacterial pneumonia")
- INCORRECT: Different conditions, even if related (e.g., "diabetes type 1" vs "diabetes type 2")
- INCORRECT: Partial match or only mentions a symptom/complication instead of the diagnosis
- INCORRECT: Overly broad or vague when gold is specific

Be STRICT. When in doubt, mark as INCORRECT.
Respond with ONLY: {{"correct": true}} or {{"correct": false}}"""

    try:
        raw = judge_client.completion([
            {"role": "system", "content": "You are a strict medical diagnosis grader. Respond with ONLY a JSON object: {\"correct\": true} or {\"correct\": false}. Nothing else."},
            {"role": "user", "content": prompt},
        ])
        # Try JSON first
        try:
            parsed = json.loads(clean_json_string(raw))
            return bool(parsed.get("correct", False))
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: look for true/false keywords in response
        raw_lower = raw.lower()
        if '"correct": true' in raw_lower or '"correct":true' in raw_lower:
            return True
        if 'correct' in raw_lower and 'true' in raw_lower and 'false' not in raw_lower:
            return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Process a single case through ALL pipeline stages
# ---------------------------------------------------------------------------

def process_case(main_client, small_client, case_text, gold):
    """Run baseline, ensemble, audit, hybrid for one case."""

    # 1. Baseline: main model alone
    baseline = get_diagnosis(main_client, case_text)

    # 2. Ensemble: main + small model, check agreement
    small_diag = get_diagnosis(small_client, case_text)
    agreement = check_semantic_agreement(baseline, small_diag)

    if agreement:
        consensus = baseline if len(baseline) >= len(small_diag) else small_diag
        ensemble_candidates = [consensus]
        ensemble_final = consensus
    else:
        ensemble_candidates = [baseline, small_diag]
        ensemble_final = baseline  # main model as tiebreaker

    # 3. Audit: main model does For/Against on ensemble candidates
    audit_result = perform_audit(main_client, case_text, ensemble_candidates)
    audit_final = audit_result.get('final_decision', 'Unknown')

    # 4. Hybrid = ensemble candidates -> audit (already done above)
    hybrid_final = audit_final

    return {
        'gold': gold,
        'baseline': baseline,
        'ensemble_main': baseline,
        'ensemble_small': small_diag,
        'model_agreement': agreement,
        'ensemble_final': ensemble_final,
        'audit_final': audit_final,
        'audit_confidence': audit_result.get('confidence', 0),
        'audit_rationale': audit_result.get('rationale', ''),
        'hybrid_final': hybrid_final,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(args):
    enable_thinking = getattr(args, 'enable_thinking', False)

    print(f"\n{'=' * 60}")
    print(f"HuggingFace Qwen3.5 Experiment")
    print(f"Main model:  {args.model_main} (quant={args.quantize_main})")
    print(f"Small model: {args.model_small} (quant={args.quantize_small})")
    print(f"Thinking:    {enable_thinking}")
    print(f"Samples:     {args.samples}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"{'=' * 60}\n")

    # Load main model (27B)
    t0 = time.time()
    main_client = HFClient(
        model_name=args.model_main,
        quantize=args.quantize_main,
        max_tokens=args.max_tokens,
        truncate_input_tokens=args.max_input,
        temperature=0.0,
        enable_thinking=enable_thinking,
    )
    print(f"Main model loaded in {time.time() - t0:.1f}s\n")

    # Load small model (9B) — for ensemble + judge
    t0 = time.time()
    small_client = HFClient(
        model_name=args.model_small,
        quantize=args.quantize_small,
        max_tokens=args.max_tokens,
        truncate_input_tokens=args.max_input,
        temperature=0.0,
        enable_thinking=enable_thinking,
    )
    print(f"Small model loaded in {time.time() - t0:.1f}s\n")

    # Process each dataset
    all_results = {}

    for dataset_label, data_path in [
        ('Easy', args.easy_path),
        ('Hard', args.hard_path),
    ]:
        if not os.path.exists(data_path):
            print(f"Skipping {dataset_label}: {data_path} not found")
            continue

        df = pd.read_csv(data_path)
        if args.samples and args.samples < len(df):
            df = df.sample(n=args.samples, random_state=args.seed)

        print(f"\n{'=' * 40}")
        print(f"Processing {dataset_label}: {len(df)} cases")
        print(f"{'=' * 40}")

        results = []
        t_start = time.time()
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=dataset_label):
            case_text = row.get('clean text') or row.get('100%') or ""
            gold = row.get('final diagnosis') or row.get('gold') or 'Unknown'
            result = process_case(main_client, small_client, case_text, gold)
            result['case_id'] = idx
            results.append(result)
        elapsed = time.time() - t_start

        result_df = pd.DataFrame(results)

        # Save raw results
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = f"{args.output_dir}/qwen35_{dataset_label.lower()}_{len(df)}.csv"
        result_df.to_csv(out_file, index=False)
        print(f"Saved: {out_file} ({elapsed:.1f}s, {elapsed / len(df):.1f}s/case)")

        # Judge with small model (Qwen3.5-9B)
        print(f"\nJudging {dataset_label} with {args.model_small}...")
        methods = {
            'Baseline': 'baseline',
            'Ensemble': 'ensemble_final',
            'Audit': 'audit_final',
            'Hybrid': 'hybrid_final',
        }
        dataset_results = {}
        for method_name, col in methods.items():
            correct = 0
            for _, row in result_df.iterrows():
                if hf_judge(small_client, row[col], row['gold']):
                    correct += 1
            acc = correct / len(result_df) * 100
            dataset_results[method_name] = acc
            print(f"  {method_name}: {acc:.1f}% ({correct}/{len(result_df)})")

        all_results[dataset_label] = dataset_results

    # Print summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS (Qwen3.5 Judge)")
    print(f"Main: {args.model_main}")
    print(f"Small/Judge: {args.model_small}")
    print(f"{'=' * 60}")
    print(f"{'Method':<20} {'Easy':>10} {'Hard':>10}")
    print("-" * 42)
    for method in ['Baseline', 'Ensemble', 'Audit', 'Hybrid']:
        easy = all_results.get('Easy', {}).get(method, 0)
        hard = all_results.get('Hard', {}).get(method, 0)
        print(f"{method:<20} {easy:>9.1f}% {hard:>9.1f}%")

    # Print usage
    main_client.print_usage()
    small_client.print_usage()

    print(f"\nDone. Results in {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 diagnostic pipeline experiment")
    parser.add_argument('--model-main', type=str, default='Qwen/Qwen3.5-27B',
                        help="Main diagnosis model (replaces GPT-4o)")
    parser.add_argument('--model-small', type=str, default='Qwen/Qwen3.5-9B',
                        help="Ensemble partner + judge (replaces DeepSeek)")
    parser.add_argument('--quantize-main', type=str, default=None, choices=[None, '4bit', '8bit'])
    parser.add_argument('--quantize-small', type=str, default=None, choices=[None, '4bit', '8bit'])
    parser.add_argument('--easy-path', default='datasets/easy_medqa.csv')
    parser.add_argument('--hard-path', default='datasets/Case_report_w_images_dis_VF.csv')
    parser.add_argument('--output-dir', default='output-hf')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-input', type=int, default=4096)
    parser.add_argument('--enable-thinking', action='store_true', default=False,
                        help="Enable Qwen3.5 thinking mode (generates <think> blocks)")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
