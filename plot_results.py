"""
Grouped bar chart comparing Baseline + 3 diagnostic methods on Easy vs Hard datasets.
Uses LLM judge for clinical equivalence (not string matching).
Usage: python plot_results.py
"""
import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def llm_judge(pred, gold):
    """Use DeepSeek to judge clinical equivalence."""
    if pd.isna(pred) or pd.isna(gold):
        return False
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not set, falling back to fuzzy match")
        return fuzzy_match(pred, gold)
    
    system = (
        "You are a strict medical answer grader. "
        "Output ONLY valid JSON: {"correct": true/false, "rationale": "..."}"
    )
    
    prompt = f"""GOLD ANSWER: {gold}
MODEL ANSWER: {pred}

Is the model answer clinically correct with respect to the gold answer?
Consider: same condition with different names, abbreviations, or added detail = CORRECT.
Return JSON only."""

    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0
            },
            timeout=60
        )
        content = r.json()["choices"][0]["message"]["content"]
        # Parse JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            return result.get("correct", False)
    except Exception as e:
        print(f"Judge error: {e}, falling back to fuzzy match")
        return fuzzy_match(pred, gold)
    
    return False

def fuzzy_match(pred, gold):
    """Fallback: case-insensitive substring match."""
    if pd.isna(pred) or pd.isna(gold):
        return False
    p = re.sub(r'[^\w\s]', '', str(pred).lower().strip())
    g = re.sub(r'[^\w\s]', '', str(gold).lower().strip())
    if not p or not g:
        return False
    return p in g or g in p or p == g

def compute_accuracy(df, pred_col, gold_col, use_llm=True):
    """Compute accuracy using LLM judge or fuzzy match."""
    if use_llm:
        matches = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Judging {pred_col}"):
            if llm_judge(row[pred_col], row[gold_col]):
                matches += 1
        return matches / len(df) * 100
    else:
        matches = sum(fuzzy_match(row[pred_col], row[gold_col]) for _, row in df.iterrows())
        return matches / len(df) * 100

def main(args):
    # --- Load CSVs ---
    results = {}
    methods = ['Baseline (GPT-4o)', 'Ensemble', 'Audit', 'Hybrid']

    for dataset_label, folder in [('Easy (MedQA)', args.easy_dir), ('Hard (CUPCase)', args.hard_dir)]:
        print(f"\n=== Processing {dataset_label} ===")
        
        ensemble_path = os.path.join(folder, 'ensemble_v2_results_100.csv')
        audit_path = os.path.join(folder, 'audit_results_100.csv')
        hybrid_path = os.path.join(folder, 'turbo_results_100.csv')
        
        if not all(os.path.exists(p) for p in [ensemble_path, audit_path, hybrid_path]):
            print(f"Missing files in {folder}, skipping...")
            continue
            
        ensemble_df = pd.read_csv(ensemble_path)
        audit_df = pd.read_csv(audit_path)
        hybrid_df = pd.read_csv(hybrid_path)

        results[dataset_label] = {
            'Baseline (GPT-4o)': compute_accuracy(ensemble_df, 'gpt4o_diagnosis', 'gold', use_llm=args.use_llm),
            'Ensemble': compute_accuracy(ensemble_df, 'final_diagnosis', 'gold', use_llm=args.use_llm),
            'Audit': compute_accuracy(audit_df, 'final_diagnosis', 'gold', use_llm=args.use_llm),
            'Hybrid': compute_accuracy(hybrid_df, 'pred', 'gold', use_llm=args.use_llm),
        }

    if not results:
        print("No results to plot. Run the pipeline first.")
        return

    # --- Print results ---
    print("\n" + "="*50)
    print("ACCURACY RESULTS")
    print("="*50)
    judge_type = "LLM Judge" if args.use_llm else "Fuzzy Match"
    print(f"Judge: {judge_type}")
    print(f"{'Method':<20} {'Easy (MedQA)':>14} {'Hard (CUPCase)':>16}")
    print("-" * 52)
    for m in methods:
        easy_val = results.get('Easy (MedQA)', {}).get(m, 0)
        hard_val = results.get('Hard (CUPCase)', {}).get(m, 0)
        print(f"{m:<20} {easy_val:>13.1f}% {hard_val:>15.1f}%")

    # --- Plot ---
    datasets = [k for k in ['Easy (MedQA)', 'Hard (CUPCase)'] if k in results]
    x = np.arange(len(datasets))
    n_methods = len(methods)
    width = 0.18
    colors = ['#999999', '#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, method in enumerate(methods):
        vals = [results[ds][method] for ds in datasets]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=colors[i], edgecolor='white', linewidth=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.8,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Diagnostic Accuracy: Baseline + 3 Methods ({judge_type}, N=100)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(0, min(max(max(v.values()) for v in results.values()) + 15, 105))
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'accuracy_comparison.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--easy-dir', default='output-100-test-easy')
    parser.add_argument('--hard-dir', default='output-100-test-hard')
    parser.add_argument('--output-dir', default='output-100-test-results')
    parser.add_argument('--use-llm', action='store_true', default=True, help='Use LLM judge (default)')
    parser.add_argument('--no-llm', dest='use_llm', action='store_false', help='Use fuzzy match instead')
    args = parser.parse_args()
    main(args)
