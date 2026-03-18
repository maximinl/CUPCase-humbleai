"""
Re-judge existing CSV outputs with a different judge model.
Reads raw predictions from hf_experiment.py outputs and scores them
with a specified judge configuration.

Usage:
    python rejudge.py --judge-model Qwen/Qwen3.5-27B --csv-dir output-hf/v01-9b-bf16
    python rejudge.py --judge-model Qwen/Qwen3.5-27B --csv-dir output-hf/v01-9b-bf16 output-hf/v03-9b-4bit output-hf/v05-27b4bit-9b
"""

import os
import json
import re
import argparse
import time
import pandas as pd
from hf_client import HFClient


def clean_json_string(s):
    s = re.sub(r"```json\s*|\s*```", "", s, flags=re.I)
    start, end = s.find('{'), s.rfind('}')
    return s[start:end + 1] if start != -1 and end != -1 else s.strip()


def hf_judge(judge_client, pred, gold):
    if not pred or not gold or str(pred).startswith("Error"):
        return False

    pred_lower = str(pred).lower().strip()
    gold_lower = str(gold).lower().strip()
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
            {"role": "system", "content": 'You are a strict medical diagnosis grader. Respond with ONLY a JSON object: {"correct": true} or {"correct": false}. Nothing else.'},
            {"role": "user", "content": prompt},
        ])
        try:
            parsed = json.loads(clean_json_string(raw))
            return bool(parsed.get("correct", False))
        except (json.JSONDecodeError, ValueError):
            pass
        raw_lower = raw.lower()
        if '"correct": true' in raw_lower or '"correct":true' in raw_lower:
            return True
        if 'correct' in raw_lower and 'true' in raw_lower and 'false' not in raw_lower:
            return True
        return False
    except Exception:
        return False


def rejudge_csv(judge_client, csv_path, judge_label):
    df = pd.read_csv(csv_path)
    methods = {
        'Baseline': 'baseline',
        'Ensemble': 'ensemble_final',
        'Audit': 'audit_final',
        'Hybrid': 'hybrid_final',
    }

    results = {}
    for method_name, col in methods.items():
        if col not in df.columns:
            continue
        correct = 0
        total = len(df)
        for _, row in df.iterrows():
            if hf_judge(judge_client, row[col], row['gold']):
                correct += 1
        acc = correct / total * 100
        results[method_name] = (correct, total, acc)

    return results


def main():
    parser = argparse.ArgumentParser(description="Re-judge existing experiment CSVs")
    parser.add_argument('--judge-model', type=str, required=True)
    parser.add_argument('--quantize-judge', type=str, default=None, choices=[None, '4bit', '8bit'])
    parser.add_argument('--judge-thinking', action='store_true', default=False)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--csv-dir', nargs='+', required=True,
                        help="One or more output directories to re-judge")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Re-judging with: {args.judge_model} (quant={args.quantize_judge}, thinking={args.judge_thinking})")
    print(f"Directories: {args.csv_dir}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    judge_client = HFClient(
        model_name=args.judge_model,
        quantize=args.quantize_judge,
        max_tokens=args.max_tokens,
        truncate_input_tokens=4096,
        temperature=0.0,
        enable_thinking=args.judge_thinking,
    )
    print(f"Judge loaded in {time.time() - t0:.1f}s\n")

    # Collect all CSVs
    all_results = []

    for csv_dir in args.csv_dir:
        variant = os.path.basename(csv_dir)
        for csv_file in sorted(os.listdir(csv_dir)):
            if not csv_file.endswith('.csv'):
                continue
            dataset = 'Easy' if 'easy' in csv_file.lower() else 'Hard'
            csv_path = os.path.join(csv_dir, csv_file)

            print(f"Judging {variant}/{csv_file} ...")
            t1 = time.time()
            results = rejudge_csv(judge_client, csv_path, variant)
            elapsed = time.time() - t1

            for method, (correct, total, acc) in results.items():
                all_results.append({
                    'variant': variant,
                    'dataset': dataset,
                    'method': method,
                    'correct': correct,
                    'total': total,
                    'accuracy': acc,
                })
                print(f"  {method}: {acc:.1f}% ({correct}/{total})")
            print(f"  ({elapsed:.1f}s)\n")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"REJUDGE SUMMARY — Judge: {args.judge_model} (quant={args.quantize_judge}, thinking={args.judge_thinking})")
    print(f"{'=' * 70}")
    print(f"{'Variant':<25} {'Dataset':<6} {'Baseline':>9} {'Ensemble':>9} {'Audit':>7} {'Hybrid':>7}")
    print("-" * 70)

    # Group by variant + dataset
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in all_results:
        grouped[(r['variant'], r['dataset'])][r['method']] = r['accuracy']

    for (variant, dataset), methods in sorted(grouped.items()):
        b = methods.get('Baseline', 0)
        e = methods.get('Ensemble', 0)
        a = methods.get('Audit', 0)
        h = methods.get('Hybrid', 0)
        print(f"{variant:<25} {dataset:<6} {b:>8.1f}% {e:>8.1f}% {a:>6.1f}% {h:>6.1f}%")

    judge_client.print_usage()
    print("\nDone.")


if __name__ == "__main__":
    main()
