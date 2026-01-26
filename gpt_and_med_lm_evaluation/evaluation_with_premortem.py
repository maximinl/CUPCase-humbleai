#!/usr/bin/env python3
"""
CUPCASE Evaluation with Pre-Mortem Selective Analysis.

This script implements the complete evaluation pipeline for the CUPCase dataset
using the Pre-Mortem Selective system with Risk Quadrant Triggering.

Usage:
    # Baseline evaluation (without Pre-Mortem)
    python evaluation_with_premortem.py --task mcq --no-premortem

    # Evaluation with Pre-Mortem
    python evaluation_with_premortem.py --task mcq --premortem

    # Free-text evaluation with Pre-Mortem
    python evaluation_with_premortem.py --task free_text --premortem --samples 100

    # Custom thresholds
    python evaluation_with_premortem.py --task mcq --premortem \\
        --complexity-threshold 0.4 --stakes-threshold 0.6
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from premortem.config import PreMortemConfig, RiskQuadrant
from premortem.belief_revision import BeliefRevisionEngine, DiagnosisResult
from premortem.quadrant_classifier import QuadrantClassifier

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CUPCASE Evaluation with Pre-Mortem Selective Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run MCQ baseline
    python evaluation_with_premortem.py --task mcq --no-premortem

    # Run MCQ with Pre-Mortem
    python evaluation_with_premortem.py --task mcq --premortem

    # Run free-text with verbose output
    python evaluation_with_premortem.py --task free_text --premortem --verbose
        """
    )

    # Task configuration
    parser.add_argument(
        '--task',
        type=str,
        choices=['mcq', 'free_text'],
        default='mcq',
        help='Evaluation task type (default: mcq)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=250,
        help='Number of samples per batch (default: 250)'
    )
    parser.add_argument(
        '--batches',
        type=int,
        default=4,
        help='Number of batches to run (default: 4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Pre-Mortem configuration
    premortem_group = parser.add_mutually_exclusive_group()
    premortem_group.add_argument(
        '--premortem',
        action='store_true',
        help='Enable Pre-Mortem analysis'
    )
    premortem_group.add_argument(
        '--no-premortem',
        action='store_true',
        help='Disable Pre-Mortem (baseline mode)'
    )
    parser.add_argument(
        '--complexity-threshold',
        type=float,
        default=0.5,
        help='Threshold for complexity classification (default: 0.5)'
    )
    parser.add_argument(
        '--stakes-threshold',
        type=float,
        default=0.5,
        help='Threshold for stakes classification (default: 0.5)'
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use for inference (default: gpt-4o)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for generation (default: 0.0)'
    )

    # Data paths
    parser.add_argument(
        '--data-path',
        type=str,
        default='datasets/Case_report_w_images_dis_VF.csv',
        help='Path to dataset CSV'
    )
    parser.add_argument(
        '--ablation-path',
        type=str,
        default='ablation_study_tokens.csv',
        help='Path to ablation study CSV with token percentages'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )

    # Misc
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load the dataset with token-truncated columns.

    Attempts to load from ablation study CSV first, falls back to
    main dataset and generates truncated columns if needed.
    """
    # Try ablation study CSV first
    if os.path.exists(args.ablation_path):
        print(f"Loading data from {args.ablation_path}")
        df = pd.read_csv(args.ablation_path)

        # Ensure we have the required columns
        if '20%' in df.columns and '100%' not in df.columns:
            df['100%'] = df['clean text']

        return df

    # Fall back to main dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.data_path}. "
            f"Please provide a valid path with --data-path"
        )

    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Generate truncated columns
    print("Generating token-truncated columns...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

        def truncate_text_by_tokens(text: str, percentage: float) -> str:
            """Truncate text to a percentage of its tokens."""
            tokens = tokenizer.tokenize(str(text))
            num_tokens = int(len(tokens) * percentage)
            truncated_tokens = tokens[:num_tokens]
            return tokenizer.convert_tokens_to_string(truncated_tokens)

        df['20%'] = df['clean text'].apply(
            lambda x: truncate_text_by_tokens(x, 0.2)
        )
        df['100%'] = df['clean text']

    except ImportError:
        print("Warning: transformers not available, using character-based truncation")
        df['20%'] = df['clean text'].apply(lambda x: str(x)[:int(len(str(x)) * 0.2)])
        df['100%'] = df['clean text']

    return df


def evaluate_batch_with_premortem(
    batch: pd.DataFrame,
    engine: BeliefRevisionEngine,
    task: str,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of cases using the Pre-Mortem pipeline.

    Args:
        batch: DataFrame batch to evaluate
        engine: BeliefRevisionEngine instance
        task: Task type ("mcq" or "free_text")
        verbose: Enable verbose output

    Returns:
        List of result dictionaries
    """
    results = []

    for idx, row in tqdm(batch.iterrows(), total=len(batch), desc="Processing"):
        case_20pct = row.get('20%', row.get('clean text', ''))[:int(len(str(row.get('clean text', ''))) * 0.2)]
        case_full = row.get('100%', row.get('clean text', ''))
        true_diagnosis = row['final diagnosis']

        # Prepare MCQ options if needed
        options = None
        correct_idx = None

        if task == 'mcq':
            options = [
                true_diagnosis,
                row['distractor2'],
                row['distractor3'],
                row['distractor4']
            ]
            random.shuffle(options)
            correct_idx = options.index(true_diagnosis)

        # Run evaluation
        try:
            result = engine.evaluate_case(
                case_text_20pct=case_20pct,
                case_text_full=case_full,
                task_type=task,
                options=options,
                true_diagnosis=true_diagnosis
            )

            # Build result record
            record = {
                'case_presentation': case_full[:500] + '...' if len(case_full) > 500 else case_full,
                'true_diagnosis': true_diagnosis,
                'initial_hypothesis': result.initial_hypothesis,
                'final_diagnosis': result.final_diagnosis,
                'quadrant': result.quadrant.name,
                'complexity_score': result.quadrant_result.complexity_score,
                'stakes_score': result.quadrant_result.stakes_score,
                'red_flags': ', '.join(result.quadrant_result.red_flags_detected),
                'premortem_applied': result.premortem_applied,
                'belief_revision': result.belief_revision_occurred,
                'revision_magnitude': result.revision_magnitude,
                'latency_total_ms': sum(result.latency_ms.values()),
            }

            # Add task-specific fields
            if task == 'mcq':
                try:
                    pred_idx = int(result.final_diagnosis.strip()[0]) - 1
                except (ValueError, IndexError):
                    pred_idx = -1

                record['correct_idx'] = correct_idx
                record['predicted_idx'] = pred_idx
                record['correct'] = (correct_idx == pred_idx)
            else:
                record['generated_diagnosis'] = result.final_diagnosis

            # Add Pre-Mortem info if applied
            if result.premortem_result:
                record['premortem_alternative'] = result.premortem_result.alternative_diagnosis
                record['premortem_evidence_strength'] = result.premortem_result.evidence_strength
                record['premortem_recommendation'] = result.premortem_result.recommendation
                record['premortem_red_flags'] = ', '.join(result.premortem_result.missed_red_flags)

            results.append(record)

            if verbose:
                print(f"\n--- Case {idx} ---")
                print(f"Quadrant: {result.quadrant.name}")
                print(f"Pre-Mortem applied: {result.premortem_applied}")
                print(f"Belief revision: {result.belief_revision_occurred}")
                if task == 'mcq':
                    print(f"Correct: {record['correct']}")

        except Exception as e:
            print(f"Error processing case {idx}: {e}")
            results.append({
                'case_presentation': case_full[:500] + '...' if len(case_full) > 500 else case_full,
                'true_diagnosis': true_diagnosis,
                'error': str(e)
            })

        # Rate limiting
        time.sleep(1)

    return results


def evaluate_batch_baseline(
    batch: pd.DataFrame,
    client: OpenAI,
    task: str,
    model: str = 'gpt-4o',
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch without Pre-Mortem (baseline mode).

    Args:
        batch: DataFrame batch to evaluate
        client: OpenAI client instance
        task: Task type ("mcq" or "free_text")
        model: Model name to use
        verbose: Enable verbose output

    Returns:
        List of result dictionaries
    """
    results = []

    for idx, row in tqdm(batch.iterrows(), total=len(batch), desc="Baseline"):
        case_full = row.get('100%', row.get('clean text', ''))
        true_diagnosis = row['final diagnosis']

        # Build prompt
        if task == 'mcq':
            options = [
                true_diagnosis,
                row['distractor2'],
                row['distractor3'],
                row['distractor4']
            ]
            random.shuffle(options)
            correct_idx = options.index(true_diagnosis)

            options_text = "\n".join(
                [f"{i+1}. {opt}" for i, opt in enumerate(options)]
            )
            prompt = (
                f"Predict the diagnosis of this case presentation of a patient. "
                f"Return only the correct index from the following list, for example: 3\n"
                f"{options_text}\n"
                f"Case presentation: {case_full}"
            )
        else:
            prompt = (
                f"Predict the diagnosis of this case presentation of a patient. "
                f"Return the final diagnosis in one concise sentence without any further elaboration.\n"
                f"For example: <diagnosis name here>\n"
                f"Case presentation: {case_full}\n"
                f"Diagnosis:"
            )

        # Call API
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            generated = response.choices[0].message.content.strip()

            record = {
                'case_presentation': case_full[:500] + '...' if len(case_full) > 500 else case_full,
                'true_diagnosis': true_diagnosis,
                'final_diagnosis': generated,
                'premortem_applied': False,
            }

            if task == 'mcq':
                try:
                    pred_idx = int(generated[0]) - 1
                except (ValueError, IndexError):
                    pred_idx = -1

                record['correct_idx'] = correct_idx
                record['predicted_idx'] = pred_idx
                record['correct'] = (correct_idx == pred_idx)
            else:
                record['generated_diagnosis'] = generated

            results.append(record)

            if verbose:
                print(f"\n--- Case {idx} ---")
                print(f"Generated: {generated[:100]}...")
                if task == 'mcq':
                    print(f"Correct: {record['correct']}")

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)
            results.append({
                'case_presentation': case_full[:500] + '...' if len(case_full) > 500 else case_full,
                'true_diagnosis': true_diagnosis,
                'error': str(e)
            })

        # Rate limiting
        time.sleep(1)

    return results


def compute_metrics(results: List[Dict], task: str) -> Dict[str, float]:
    """
    Compute evaluation metrics from results.

    Args:
        results: List of result dictionaries
        task: Task type ("mcq" or "free_text")

    Returns:
        Dictionary of computed metrics
    """
    df = pd.DataFrame(results)

    # Filter out errors
    error_mask = df.get('error', pd.Series([None] * len(df))).isna()
    df_valid = df[error_mask]

    metrics = {
        'total_samples': len(df),
        'valid_samples': len(df_valid),
        'error_rate': (len(df) - len(df_valid)) / len(df) if len(df) > 0 else 0
    }

    if len(df_valid) == 0:
        return metrics

    # Task-specific metrics
    if task == 'mcq':
        if 'correct' in df_valid.columns:
            metrics['accuracy'] = df_valid['correct'].mean()
    else:
        # Compute BERTScore for free-text
        try:
            import bert_score
            predictions = df_valid['generated_diagnosis'].tolist()
            references = df_valid['true_diagnosis'].tolist()

            P, R, F1 = bert_score.score(
                predictions, references,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli"
            )

            metrics['bertscore_precision_mean'] = P.mean().item()
            metrics['bertscore_recall_mean'] = R.mean().item()
            metrics['bertscore_f1_mean'] = F1.mean().item()
            metrics['bertscore_f1_std'] = F1.std().item()
        except ImportError:
            print("Warning: bert_score not available, skipping BERTScore computation")
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")

    # Pre-Mortem specific metrics
    if 'premortem_applied' in df_valid.columns:
        metrics['premortem_rate'] = df_valid['premortem_applied'].mean()

        # Accuracy by Pre-Mortem status
        if task == 'mcq' and 'correct' in df_valid.columns:
            pm_applied = df_valid[df_valid['premortem_applied'] == True]
            pm_not_applied = df_valid[df_valid['premortem_applied'] == False]

            if len(pm_applied) > 0:
                metrics['accuracy_with_premortem'] = pm_applied['correct'].mean()
                metrics['samples_with_premortem'] = len(pm_applied)
            if len(pm_not_applied) > 0:
                metrics['accuracy_without_premortem'] = pm_not_applied['correct'].mean()
                metrics['samples_without_premortem'] = len(pm_not_applied)

    # Belief revision metrics
    if 'belief_revision' in df_valid.columns:
        metrics['belief_revision_rate'] = df_valid['belief_revision'].mean()

        # Accuracy for revised vs non-revised
        if task == 'mcq' and 'correct' in df_valid.columns:
            revised = df_valid[df_valid['belief_revision'] == True]
            not_revised = df_valid[df_valid['belief_revision'] == False]

            if len(revised) > 0:
                metrics['accuracy_revised'] = revised['correct'].mean()
                metrics['samples_revised'] = len(revised)
            if len(not_revised) > 0:
                metrics['accuracy_not_revised'] = not_revised['correct'].mean()
                metrics['samples_not_revised'] = len(not_revised)

    # Quadrant-specific metrics
    if 'quadrant' in df_valid.columns and task == 'mcq' and 'correct' in df_valid.columns:
        for q in df_valid['quadrant'].unique():
            q_df = df_valid[df_valid['quadrant'] == q]
            metrics[f'accuracy_{q}'] = q_df['correct'].mean()
            metrics[f'samples_{q}'] = len(q_df)

    return metrics


def main():
    """Main entry point."""
    args = parse_args()

    # Setup
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    try:
        df = load_data(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} cases")

    # Determine mode
    enable_premortem = args.premortem and not args.no_premortem

    # Configure Pre-Mortem
    config = PreMortemConfig(
        enable_premortem=enable_premortem,
        complexity_threshold=args.complexity_threshold,
        stakes_threshold=args.stakes_threshold,
        model_name=args.model,
        temperature=args.temperature,
        verbose=args.verbose
    )

    engine = BeliefRevisionEngine(client, config) if enable_premortem else None

    # Run evaluation
    all_results = []

    for batch_num in range(args.batches):
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_num + 1}/{args.batches}")
        print(f"Mode: {'Pre-Mortem' if enable_premortem else 'Baseline'}")
        print(f"Task: {args.task}")
        print(f"{'='*60}")

        # Sample batch
        batch = df.sample(n=min(args.samples, len(df)), random_state=args.seed + batch_num)

        # Evaluate batch
        if enable_premortem:
            batch_results = evaluate_batch_with_premortem(
                batch, engine, args.task, args.verbose
            )
        else:
            batch_results = evaluate_batch_baseline(
                batch, client, args.task, args.model, args.verbose
            )

        all_results.extend(batch_results)

        print(f"Completed batch {batch_num + 1}")

        # Pause between batches
        if batch_num < args.batches - 1:
            time.sleep(10)

    # Compute metrics
    metrics = compute_metrics(all_results, args.task)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "premortem" if enable_premortem else "baseline"

    results_path = f"{args.output_dir}/{args.task}_{mode}_{timestamp}.csv"
    metrics_path = f"{args.output_dir}/{args.task}_{mode}_{timestamp}_metrics.json"

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Task: {args.task}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid samples: {metrics['valid_samples']}")

    if args.task == 'mcq':
        print(f"\nAccuracy: {metrics.get('accuracy', 0):.4f}")
    else:
        print(f"\nBERTScore F1: {metrics.get('bertscore_f1_mean', 0):.4f} "
              f"+/- {metrics.get('bertscore_f1_std', 0):.4f}")

    if enable_premortem:
        print(f"\n--- Pre-Mortem Statistics ---")
        print(f"Pre-Mortem trigger rate: {metrics.get('premortem_rate', 0):.2%}")
        print(f"Belief revision rate: {metrics.get('belief_revision_rate', 0):.2%}")

        if args.task == 'mcq':
            if 'accuracy_with_premortem' in metrics:
                print(f"Accuracy WITH Pre-Mortem: {metrics['accuracy_with_premortem']:.4f} "
                      f"(n={metrics.get('samples_with_premortem', 0)})")
            if 'accuracy_without_premortem' in metrics:
                print(f"Accuracy WITHOUT Pre-Mortem: {metrics['accuracy_without_premortem']:.4f} "
                      f"(n={metrics.get('samples_without_premortem', 0)})")

            # Print quadrant breakdown
            print(f"\n--- Accuracy by Quadrant ---")
            for q in ['ROUTINE', 'WATCHFUL', 'CURIOSITY', 'ESCALATE']:
                if f'accuracy_{q}' in metrics:
                    print(f"{q}: {metrics[f'accuracy_{q}']:.4f} "
                          f"(n={metrics.get(f'samples_{q}', 0)})")

    print(f"\nResults saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
