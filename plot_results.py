"""
Grouped bar chart comparing Baseline + 3 diagnostic methods on Easy vs Hard datasets.
Usage: python plot_results.py
"""
import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def fuzzy_match(pred, gold):
    """Case-insensitive substring match after stripping punctuation."""
    if pd.isna(pred) or pd.isna(gold):
        return False
    p = re.sub(r'[^\w\s]', '', str(pred).lower().strip())
    g = re.sub(r'[^\w\s]', '', str(gold).lower().strip())
    if not p or not g:
        return False
    return p in g or g in p or p == g


def compute_accuracy(df, pred_col, gold_col):
    matches = sum(fuzzy_match(row[pred_col], row[gold_col]) for _, row in df.iterrows())
    return matches / len(df) * 100


def main(args):
    # --- Load CSVs ---
    results = {}
    methods = ['Baseline (GPT-4o)', 'Ensemble', 'Audit', 'Hybrid']

    for dataset_label, folder in [('Easy (MedQA)', args.easy_dir), ('Hard (CUPCase)', args.hard_dir)]:
        ensemble_df = pd.read_csv(os.path.join(folder, 'ensemble_v2_results_100.csv'))
        audit_df = pd.read_csv(os.path.join(folder, 'audit_results_100.csv'))
        hybrid_df = pd.read_csv(os.path.join(folder, 'turbo_results_100.csv'))

        results[dataset_label] = {
            'Baseline (GPT-4o)': compute_accuracy(ensemble_df, 'gpt4o_diagnosis', 'gold'),
            'Ensemble': compute_accuracy(ensemble_df, 'final_diagnosis', 'gold'),
            'Audit': compute_accuracy(audit_df, 'final_diagnosis', 'gold'),
            'Hybrid': compute_accuracy(hybrid_df, 'pred', 'gold'),
        }

    # --- Print results ---
    print("\nAccuracy Results (%):")
    print(f"{'Method':<12} {'Easy (MedQA)':>14} {'Hard (CUPCase)':>16}")
    print("-" * 44)
    for m in methods:
        print(f"{m:<12} {results['Easy (MedQA)'][m]:>13.1f}% {results['Hard (CUPCase)'][m]:>15.1f}%")

    # --- Plot ---
    x = np.arange(len(['Easy (MedQA)', 'Hard (CUPCase)']))
    n_methods = len(methods)
    width = 0.18
    colors = ['#999999', '#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, method in enumerate(methods):
        vals = [results[ds][method] for ds in ['Easy (MedQA)', 'Hard (CUPCase)']]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=colors[i], edgecolor='white', linewidth=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.8,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Diagnostic Accuracy: Baseline + 3 Methods on Easy vs Hard (N=100)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy (MedQA)', 'Hard (CUPCase)'], fontsize=11)
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
    args = parser.parse_args()
    main(args)
