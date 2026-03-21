#!/usr/bin/env python3
"""
Explain variance ratio calculation and check how many questions are filtered.

This script demonstrates:
1. How variance ratio is calculated step-by-step
2. How many questions have <10 instances (and would be filtered)
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# Add src to path
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_src_dir))

from synthetic_sampling.evaluation import load_results, ParsedInstance


def explain_variance_ratio_calculation():
    """
    Explain how variance ratio is calculated with a concrete example.
    """
    print("=" * 80)
    print("VARIANCE RATIO CALCULATION EXPLAINED")
    print("=" * 80)
    
    print("\nStep 1: Group instances by question")
    print("-" * 80)
    print("Questions are identified as: {survey}_{target_code}")
    print("Example: 'wvs_Q1', 'afrobarometer_Q38A', etc.")
    print("\nFor each question, we collect all instances (responses) for that question.")
    
    print("\n\nStep 2: Build empirical distribution (ground truth)")
    print("-" * 80)
    print("For a question with 3 options: ['Yes', 'No', 'Maybe']")
    print("If we have 100 responses:")
    print("  - 60 people answered 'Yes'")
    print("  - 30 people answered 'No'")
    print("  - 10 people answered 'Maybe'")
    print("\nEmpirical distribution (probabilities):")
    print("  P('Yes') = 60/100 = 0.60")
    print("  P('No') = 30/100 = 0.30")
    print("  P('Maybe') = 10/100 = 0.10")
    print("\nEmpirical distribution vector: [0.60, 0.30, 0.10]")
    
    print("\n\nStep 3: Compute empirical variance")
    print("-" * 80)
    print("Variance measures how spread out the probabilities are.")
    print("Formula: var(p) = mean((p_i - mean(p))²)")
    print("\nFor [0.60, 0.30, 0.10]:")
    print("  mean = (0.60 + 0.30 + 0.10) / 3 = 0.333")
    print("  var = [(0.60-0.333)² + (0.30-0.333)² + (0.10-0.333)²] / 3")
    print("      = [0.071 + 0.001 + 0.054] / 3")
    print("      = 0.042")
    
    emp_dist = np.array([0.60, 0.30, 0.10])
    emp_var = np.var(emp_dist)
    print(f"\n  → Empirical variance = {emp_var:.4f}")
    
    print("\n\nStep 4: Build predicted distribution (model's probabilities)")
    print("-" * 80)
    print("For each of the 100 responses, the model outputs probabilities:")
    print("  Response 1: P('Yes')=0.7, P('No')=0.2, P('Maybe')=0.1")
    print("  Response 2: P('Yes')=0.6, P('No')=0.3, P('Maybe')=0.1")
    print("  Response 3: P('Yes')=0.8, P('No')=0.15, P('Maybe')=0.05")
    print("  ... (100 responses total)")
    print("\nWe average these probabilities across all responses:")
    print("  Average P('Yes') = mean([0.7, 0.6, 0.8, ...])")
    print("  Average P('No') = mean([0.2, 0.3, 0.15, ...])")
    print("  Average P('Maybe') = mean([0.1, 0.1, 0.05, ...])")
    print("\nExample result: [0.65, 0.28, 0.07]")
    
    pred_dist = np.array([0.65, 0.28, 0.07])
    pred_var = np.var(pred_dist)
    print(f"\n  → Predicted variance = {pred_var:.4f}")
    
    print("\n\nStep 5: Compute variance ratio")
    print("-" * 80)
    print("Variance Ratio = Predicted Variance / Empirical Variance")
    vr = pred_var / emp_var
    print(f"\n  Variance Ratio = {pred_var:.4f} / {emp_var:.4f} = {vr:.4f}")
    
    print("\n\nInterpretation:")
    print("-" * 80)
    print("  • VR = 1.0  → Model predictions match empirical diversity")
    print("  • VR < 1.0  → Model predictions are LESS diverse than reality (flattening)")
    print("  • VR > 1.0  → Model predictions are MORE diverse than reality (unlikely)")
    print(f"\n  In this example: VR = {vr:.4f} < 1.0, so the model shows flattening")
    print("  (predictions are less diverse than the actual responses)")
    
    print("\n\nStep 6: Aggregate across questions")
    print("-" * 80)
    print("We compute variance ratio for each question separately.")
    print("Then we take the MEDIAN of all variance ratios across questions.")
    print("\nExample:")
    print("  Question 1: VR = 0.85")
    print("  Question 2: VR = 0.92")
    print("  Question 3: VR = 0.78")
    print("  Question 4: VR = 0.95")
    print("  Question 5: VR = 0.88")
    print("\n  Median VR = 0.88 (middle value when sorted)")
    print("\nWhy median? It's robust to outliers (questions with extreme VR values)")


def check_question_distribution(results_dir: Path, profile_filter: str = 's6m4'):
    """
    Check how many instances per question we have, and how many would be filtered.
    """
    print("\n" + "=" * 80)
    print("CHECKING QUESTION DISTRIBUTION")
    print("=" * 80)
    
    # Find model folders
    model_folders = [f for f in results_dir.iterdir() if f.is_dir()]
    if not model_folders:
        print(f"No model folders found in {results_dir}")
        return
    
    # Use first model as example
    model_folder = model_folders[0]
    print(f"\nAnalyzing: {model_folder.name}")
    print(f"Profile filter: {profile_filter}")
    
    # Load instances
    instances = []
    for jsonl_file in sorted(model_folder.glob("*.jsonl")):
        try:
            batch = load_results(str(jsonl_file))
            for inst in batch:
                if inst.profile_type == profile_filter:
                    instances.append(inst)
        except Exception as e:
            print(f"  Warning: Could not load {jsonl_file.name}: {e}")
            continue
    
    print(f"\nTotal instances loaded: {len(instances):,}")
    
    # Group by question
    by_question = defaultdict(list)
    for inst in instances:
        key = f"{inst.survey}_{inst.target_code}"
        by_question[key].append(inst)
    
    print(f"Total unique questions: {len(by_question):,}")
    
    # Count instances per question
    question_sizes = [len(insts) for insts in by_question.values()]
    question_sizes.sort()
    
    print(f"\nQuestion size distribution:")
    print(f"  Minimum: {min(question_sizes)} instances")
    print(f"  Maximum: {max(question_sizes):,} instances")
    print(f"  Median: {np.median(question_sizes):.1f} instances")
    print(f"  Mean: {np.mean(question_sizes):.1f} instances")
    
    # Count how many would be filtered with different min_n values
    print(f"\nFiltering with different min_n thresholds:")
    for min_n in [5, 10, 20, 30, 50]:
        included = sum(1 for size in question_sizes if size >= min_n)
        excluded = len(question_sizes) - included
        pct_included = 100 * included / len(question_sizes) if question_sizes else 0
        print(f"  min_n={min_n:2d}: {included:4d} questions included ({pct_included:5.1f}%), "
              f"{excluded:4d} excluded")
    
    # Show distribution histogram
    print(f"\nQuestion size distribution (histogram):")
    bins = [0, 5, 10, 20, 30, 50, 100, 200, 500, 1000, float('inf')]
    bin_counts = [0] * (len(bins) - 1)
    bin_labels = []
    for i in range(len(bins) - 1):
        if bins[i+1] == float('inf'):
            bin_labels.append(f"{bins[i]}+")
        else:
            bin_labels.append(f"{bins[i]}-{bins[i+1]-1}")
    
    for size in question_sizes:
        for i in range(len(bins) - 1):
            if bins[i] <= size < bins[i+1] or (i == len(bins) - 2 and size >= bins[i]):
                bin_counts[i] += 1
                break
    
    for label, count in zip(bin_labels, bin_counts):
        bar = '█' * int(50 * count / max(bin_counts)) if max(bin_counts) > 0 else ''
        print(f"  {label:>8}: {count:4d} {bar}")
    
    # Show some examples of small questions
    print(f"\nExamples of questions with <10 instances:")
    small_questions = [(key, len(insts)) for key, insts in by_question.items() if len(insts) < 10]
    small_questions.sort(key=lambda x: x[1])
    
    if small_questions:
        for key, size in small_questions[:10]:
            print(f"  {key}: {size} instances")
        if len(small_questions) > 10:
            print(f"  ... and {len(small_questions) - 10} more")
    else:
        print("  None! All questions have ≥10 instances")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Explain variance ratio calculation and check question distribution'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        help='Results directory to check (optional, for distribution analysis)'
    )
    parser.add_argument(
        '--profile',
        type=str,
        default='s6m4',
        help='Profile filter (default: s6m4)'
    )
    
    args = parser.parse_args()
    
    # Always explain the calculation
    explain_variance_ratio_calculation()
    
    # Check distribution if results dir provided
    if args.results_dir:
        if not args.results_dir.exists():
            print(f"\nError: Results directory not found: {args.results_dir}")
            return
        check_question_distribution(args.results_dir, args.profile)
    else:
        print("\n" + "=" * 80)
        print("To check question distribution in your data, run:")
        print(f"  python {Path(__file__).name} --results-dir <path_to_results>")
        print("=" * 80)


if __name__ == '__main__':
    main()
