
import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

def analyze_selective_triggering(base_df, pm_df, dataset_name):
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {dataset_name}")
    print(f"{'='*60}")

    # Merge on case_id
    merged = pd.merge(
        base_df[['case_id', 'judge_correct']],
        pm_df[['case_id', 'judge_correct', 'complexity_score', 'stakes_score']],
        on='case_id',
        suffixes=('_base', '_pm')
    )
    
    n = len(merged)
    baseline_acc = merged['judge_correct_base'].mean()
    
    print(f"Total Cases (n): {n}")
    print(f"Baseline Accuracy:  {baseline_acc:.2%}")

    print(f"\n--- Selective Triggering Simulation ---")
    print(f"{'Threshold':<10} | {'Triggered':<10} | {'Helped':<8} | {'Hurt':<8} | {'Net':<5} | {'New Acc':<10} | {'vs Base':<10}")
    print("-" * 85)

    best_acc = 0
    best_thresh = 0

    for thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        mask = (merged['complexity_score'] > thresh) | (merged['stakes_score'] > thresh)
        
        selective_correct = np.where(
            mask, 
            merged['judge_correct_pm'], 
            merged['judge_correct_base']
        )
        
        acc = selective_correct.mean()
        triggered_subset = merged[mask]
        helped = ((triggered_subset['judge_correct_base'] == False) & (triggered_subset['judge_correct_pm'] == True)).sum()
        hurt = ((triggered_subset['judge_correct_base'] == True) & (triggered_subset['judge_correct_pm'] == False)).sum()
        net = helped - hurt
        
        print(f"> {thresh:<8} | {mask.sum():<10} | {helped:<8} | {hurt:<8} | {net:<+5} | {acc:.2%}   | {acc - baseline_acc:+.2%}")

if __name__ == "__main__":
    # Update paths as needed when running locally
    print("Running Analysis...")
    # Add logic to load your CSVs here
