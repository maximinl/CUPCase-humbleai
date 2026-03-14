"""
Grouped bar chart comparing Baseline + 3 diagnostic methods on Easy vs Hard datasets.
Uses LLM judge for clinical equivalence.
Usage: python plot_results.py
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from judge import UnifiedJudge

load_dotenv()


def compute_accuracy(df, pred_col, gold_col):
    """Compute accuracy using LLM judge (no fuzzy fallback)."""
    judge = UnifiedJudge(enable_cache=True)
    matches = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Judging {pred_col}"):
        result = judge.judge(pred=row[pred_col], gold=row[gold_col])
        if result.correct:
            matches += 1

    return matches / len(df) * 100


def main(args):
    results = {}
    methods = ['Baseline (GPT-4o)', 'Ensemble', 'Audit', 'Hybrid']

    for dataset_label, folder in [('Easy (MedQA)', args.easy_dir), ('Hard (CUPCase)', args.hard_dir)]:
        print(f"\n=== Processing {dataset_label} ===")

        ensemble_path = os.path.join(folder, 'ensemble_v2_results_300.csv')
        audit_path = os.path.join(folder, 'audit_results_300.csv')
        hybrid_path = os.path.join(folder, 'turbo_results_300.csv')

        # Fallback to 100 sample files if 300 not found
        if not os.path.exists(ensemble_path):
            ensemble_path = os.path.join(folder, 'ensemble_v2_results_100.csv')
            audit_path = os.path.join(folder, 'audit_results_100.csv')
            hybrid_path = os.path.join(folder, 'turbo_results_100.csv')

        if not all(os.path.exists(p) for p in [ensemble_path, audit_path, hybrid_path]):
            print(f"Missing files in {folder}, skipping...")
            continue

        ensemble_df = pd.read_csv(ensemble_path)
        audit_df = pd.read_csv(audit_path)
        hybrid_df = pd.read_csv(hybrid_path)

        print(f"Loaded {len(ensemble_df)} cases")

        results[dataset_label] = {
            'Baseline (GPT-4o)': compute_accuracy(ensemble_df, 'gpt4o_diagnosis', 'gold'),
            'Ensemble': compute_accuracy(ensemble_df, 'final_diagnosis', 'gold'),
            'Audit': compute_accuracy(audit_df, 'final_diagnosis', 'gold'),
            'Hybrid': compute_accuracy(hybrid_df, 'pred', 'gold'),
        }

    if not results:
        print("No results to plot. Run the pipeline first.")
        return

    print("\n" + "="*50)
    print("ACCURACY RESULTS")
    print("="*50)
    print("Judge: LLM Judge (strict)")
    print(f"{'Method':<20} {'Easy (MedQA)':>14} {'Hard (CUPCase)':>16}")
    print("-" * 52)
    for m in methods:
        easy_val = results.get('Easy (MedQA)', {}).get(m, 0)
        hard_val = results.get('Hard (CUPCase)', {}).get(m, 0)
        print(f"{m:<20} {easy_val:>13.1f}% {hard_val:>15.1f}%")

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
    n_samples = len(ensemble_df) if 'ensemble_df' in dir() else 100
    ax.set_title(f'Diagnostic Accuracy: Baseline + 3 Methods (LLM Judge, N={n_samples})', fontsize=13, fontweight='bold')
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
    parser.add_argument('--easy-dir', default='output-300-easy')
    parser.add_argument('--hard-dir', default='output-300-hard')
    parser.add_argument('--output-dir', default='output-300-results')
    args = parser.parse_args()
    main(args)
