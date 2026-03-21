#!/usr/bin/env python3
"""
Analyze profile richness effect disaggregated by section.

Tests whether the benefit from profile richness varies across sections.
This helps answer: Do harder sections benefit more or less from richer profiles?

Usage:
    python analyze_profile_richness_by_section.py \\
        --results-dir results/ \\
        --inputs outputs/main_data \\
        --output analysis/profile_richness_by_section
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add src to path
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_src_dir))

from synthetic_sampling.evaluation import ResultsAnalyzer
from shared_data_cache import get_all_models_enriched


def analyze_profile_richness_by_section(
    instances_by_model: Dict[str, List],
    min_n_per_section: int = 50,
) -> Dict[str, Dict[str, Dict]]:
    """
    Analyze profile richness effect by section.
    
    For each section, computes metrics for each profile level (sparse/medium/rich).
    
    Returns
    -------
    Dict mapping model_name -> Dict mapping section -> Dict mapping profile_type -> metrics
    """
    results = {}
    
    for model_name, all_instances in instances_by_model.items():
        print(f"\nAnalyzing {model_name}...")
        
        # Group by section and profile type
        by_section_profile = defaultdict(lambda: defaultdict(list))
        
        for inst in all_instances:
            # Use target_section
            section = inst.target_section or "unknown"
            if inst.profile_type in ['s3m2', 's4m3', 's6m4']:
                by_section_profile[section][inst.profile_type].append(inst)
        
        # Compute metrics for each section × profile combination
        model_results = {}
        
        for section, by_profile in by_section_profile.items():
            section_results = {}
            
            for profile_type in ['s3m2', 's4m3', 's6m4']:
                instances = by_profile.get(profile_type, [])
                
                if len(instances) < min_n_per_section:
                    continue
                
                analyzer = ResultsAnalyzer(instances)
                overall = analyzer.overall_metrics()
                
                # Heterogeneity
                hetero = analyzer.heterogeneity_analysis(min_n=10)
                vr_soft = hetero.get('variance_ratio_soft', {}).get('median')
                js_soft = hetero.get('js_divergence_soft', {}).get('median')
                
                section_results[profile_type] = {
                    'n_instances': len(instances),
                    'accuracy': overall.accuracy,
                    'macro_f1': overall.macro_f1,
                    'variance_ratio_soft': vr_soft,
                    'js_divergence_soft': js_soft,
                    'brier_score': overall.brier_score,
                    'mean_log_loss': overall.mean_log_loss,
                }
            
            if section_results:
                model_results[section] = section_results
                # Print summary
                if 's3m2' in section_results and 's6m4' in section_results:
                    sparse_acc = section_results['s3m2']['accuracy']
                    rich_acc = section_results['s6m4']['accuracy']
                    improvement = rich_acc - sparse_acc
                    print(f"  {section:40s} n={section_results['s6m4']['n_instances']:>5,} "
                          f"acc: {sparse_acc:.1%}→{rich_acc:.1%} ({improvement:+.1%})")
        
        if model_results:
            results[model_name] = model_results
    
    return results


def compute_section_difficulty(instances_by_model: Dict[str, List]) -> Dict[str, float]:
    """
    Compute section difficulty based on average accuracy across all models and profiles.
    
    Lower accuracy = harder section.
    """
    # Group by section
    by_section = defaultdict(list)
    
    for model_name, instances in instances_by_model.items():
        for inst in instances:
            section = inst.target_section or "unknown"
            if inst.correct is not None:
                by_section[section].append(int(inst.correct))
    
    # Compute average accuracy per section
    section_difficulty = {}
    for section, correct_list in by_section.items():
        if len(correct_list) >= 50:  # Need enough samples
            avg_accuracy = np.mean(correct_list)
            section_difficulty[section] = avg_accuracy
    
    return section_difficulty


def analyze_richness_benefit_by_difficulty(
    results: Dict[str, Dict[str, Dict]],
    section_difficulty: Dict[str, float],
) -> None:
    """
    Analyze whether harder sections benefit more or less from profile richness.
    """
    print("\n" + "=" * 80)
    print("PROFILE RICHNESS BENEFIT BY SECTION DIFFICULTY")
    print("=" * 80)
    
    # Collect improvements by section difficulty
    improvements_by_difficulty = []
    
    for model_name, section_results in results.items():
        for section, profile_metrics in section_results.items():
            if section not in section_difficulty:
                continue
            
            if 's3m2' in profile_metrics and 's6m4' in profile_metrics:
                sparse_acc = profile_metrics['s3m2']['accuracy']
                rich_acc = profile_metrics['s6m4']['accuracy']
                improvement = rich_acc - sparse_acc
                difficulty = section_difficulty[section]
                
                improvements_by_difficulty.append({
                    'model': model_name,
                    'section': section,
                    'difficulty': difficulty,
                    'improvement': improvement,
                })
    
    if not improvements_by_difficulty:
        print("\nInsufficient data for difficulty analysis.")
        return
    
    # Split into easy vs hard sections
    difficulties = [d['difficulty'] for d in improvements_by_difficulty]
    median_difficulty = np.median(difficulties)
    
    easy_sections = [d for d in improvements_by_difficulty if d['difficulty'] >= median_difficulty]
    hard_sections = [d for d in improvements_by_difficulty if d['difficulty'] < median_difficulty]
    
    avg_easy = np.mean([d['improvement'] for d in easy_sections]) if easy_sections else 0
    avg_hard = np.mean([d['improvement'] for d in hard_sections]) if hard_sections else 0
    
    print(f"\nSection difficulty threshold: {median_difficulty:.1%} accuracy")
    print(f"\nEasy sections (≥{median_difficulty:.1%} accuracy):")
    print(f"  Average improvement: {avg_easy:+.2%}")
    print(f"  N sections: {len(easy_sections)}")
    
    print(f"\nHard sections (<{median_difficulty:.1%} accuracy):")
    print(f"  Average improvement: {avg_hard:+.2%}")
    print(f"  N sections: {len(hard_sections)}")
    
    if avg_hard > avg_easy:
        print(f"\n→ Harder sections benefit MORE from profile richness "
              f"({avg_hard - avg_easy:+.2%} difference)")
    elif avg_easy > avg_hard:
        print(f"\n→ Easier sections benefit MORE from profile richness "
              f"({avg_easy - avg_hard:+.2%} difference)")
    else:
        print(f"\n→ No clear difference between easy and hard sections")
    
    # Show top/bottom sections
    print(f"\nTop 10 sections by improvement (sparse → rich):")
    sorted_by_improvement = sorted(improvements_by_difficulty, key=lambda x: -x['improvement'])
    for i, item in enumerate(sorted_by_improvement[:10], 1):
        print(f"  {i:2d}. {item['section']:40s} {item['improvement']:+.2%} "
              f"(difficulty: {item['difficulty']:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze profile richness effect by section',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help='Results root directory (containing model folders)'
    )
    parser.add_argument(
        '--inputs',
        type=Path,
        required=True,
        help='Input data directory (main_data) for enrichment'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/profile_richness_by_section'),
        help='Output directory'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='*',
        help='Restrict to these model names (default: all)'
    )
    parser.add_argument(
        '--min-n',
        type=int,
        default=50,
        help='Minimum instances per section × profile combination (default: 50)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Cache directory (default: analysis/.cache/enriched - shared with other scripts)'
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload (ignore cache)'
    )
    
    args = parser.parse_args()
    
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use shared cache directory so all analysis scripts can reuse the same cache
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        # Default to shared cache location (analysis/.cache/enriched)
        # This allows profile_richness, profile_richness_by_topic, and disaggregated to share cache
        cache_dir = Path('analysis/.cache/enriched')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load enriched instances for all models (using cache)
    print("Loading enriched instances (using shared cache)...")
    print("=" * 80)
    
    input_paths = [args.inputs]
    instances_by_model = get_all_models_enriched(
        results_dir=args.results_dir,
        input_paths=input_paths,
        cache_dir=cache_dir,
        profile_filter=None,  # Load all profiles
        model_whitelist=args.models,
        force_reload=args.force_reload,
        verbose=True,
    )
    
    if not instances_by_model:
        print("\nError: No instances loaded.")
        sys.exit(1)
    
    # Compute section difficulty
    print("\nComputing section difficulty...")
    section_difficulty = compute_section_difficulty(instances_by_model)
    print(f"Found {len(section_difficulty)} sections with sufficient data")
    
    # Analyze profile richness by section
    print("\nAnalyzing profile richness by section...")
    print("=" * 80)
    results = analyze_profile_richness_by_section(
        instances_by_model,
        min_n_per_section=args.min_n,
    )
    
    # Analyze benefit by difficulty
    analyze_richness_benefit_by_difficulty(results, section_difficulty)
    
    # Save results
    serializable_results = {}
    for model_name, section_results in results.items():
        serializable_results[model_name] = {}
        for section, profile_metrics in section_results.items():
            serializable_results[model_name][section] = profile_metrics
    
    json_path = output_dir / 'profile_richness_by_section.json'
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Saved results to {json_path}")
    
    # Also save section difficulty
    difficulty_path = output_dir / 'section_difficulty.json'
    with open(difficulty_path, 'w') as f:
        json.dump(section_difficulty, f, indent=2)
    print(f"✓ Saved section difficulty to {difficulty_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
