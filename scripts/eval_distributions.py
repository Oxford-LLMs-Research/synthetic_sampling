#!/usr/bin/env python
"""
Distribution Comparison Evaluation Script.

This script performs country-level and exploratory intersectional analysis
of LLM survey prediction results, comparing predicted response distributions
against empirical (ground truth) distributions.

Analyses:
1. Country-level comparison (primary)
   - Both conditioned and unconditioned
   - Aggregated metrics by country, target, profile type
   
2. Intersectional (country × gender) comparison (exploratory)
   - Only for subgroups with n ≥ 30
   - Flagged as exploratory in output

Outputs:
- JSON summary with all metrics
- Per-country breakdown
- Per-target breakdown  
- Per-profile-type breakdown
- Flagged outliers (JS > threshold)

Usage:
    python scripts/eval_distributions.py \
        --predictions predictions.jsonl \
        --output results/ \
        --min_n 30 \
        --survey_type ess
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

# Import from synthetic_sampling package
from synthetic_sampling.evaluation import (
    DistributionComparator,
    AggregatedResults,
    ComparisonResult,
    SubgroupDefinition,
    FeatureMapper,
    load_instances_jsonl,
    save_results_json,
    print_summary,
    compute_all_metrics,
    aggregate_to_distribution,
)


# =============================================================================
# Analysis Functions
# =============================================================================

def run_country_analysis(
    comparator: DistributionComparator,
    conditioned: bool,
    verbose: bool = True
) -> AggregatedResults:
    """
    Run country-level distribution comparison.
    
    Parameters
    ----------
    comparator : DistributionComparator
        Configured comparator with loaded instances
    conditioned : bool
        Whether to use conditioned (profile contains country) or 
        unconditioned (all instances from country) analysis
    verbose : bool
        Print progress
        
    Returns
    -------
    AggregatedResults
        Results for all country-target-profile combinations
    """
    analysis_type = "conditioned" if conditioned else "unconditioned"
    if verbose:
        print(f"\nRunning country-level {analysis_type} analysis...")
    
    results = comparator.compare_by_country(conditioned=conditioned)
    
    if verbose:
        print(f"  Completed {len(results.results)} comparisons")
        print_summary(results, f"Country-Level ({analysis_type})")
    
    return results


def run_intersectional_analysis(
    comparator: DistributionComparator,
    gender_values: List[str] = None,
    conditioned: bool = True,
    verbose: bool = True
) -> AggregatedResults:
    """
    Run intersectional (country × gender) distribution comparison.
    
    This is exploratory analysis - smaller sample sizes expected.
    
    Parameters
    ----------
    comparator : DistributionComparator
        Configured comparator
    gender_values : list[str]
        Gender categories to analyze
    conditioned : bool
        Analysis type
    verbose : bool
        Print progress
        
    Returns
    -------
    AggregatedResults
        Results flagged as exploratory
    """
    if gender_values is None:
        gender_values = ['Male', 'Female']
    
    analysis_type = "conditioned" if conditioned else "unconditioned"
    if verbose:
        print(f"\nRunning intersectional {analysis_type} analysis (exploratory)...")
    
    results = comparator.compare_intersectional(
        gender_values=gender_values,
        conditioned=conditioned
    )
    
    if verbose:
        print(f"  Completed {len(results.results)} comparisons")
        if results.results:
            print_summary(results, f"Intersectional ({analysis_type}) - EXPLORATORY")
    
    return results


def summarize_by_dimensions(
    results: AggregatedResults,
    comparator: DistributionComparator
) -> Dict[str, Dict]:
    """
    Create summaries grouped by different dimensions.
    
    Returns
    -------
    dict
        Summaries by country, target, profile_type
    """
    return {
        'by_country': comparator.summary_by_dimension(results, 'subgroup'),
        'by_target': comparator.summary_by_dimension(results, 'target_code'),
        'by_profile_type': comparator.summary_by_dimension(results, 'profile_type'),
    }


def identify_outliers(
    results: AggregatedResults,
    js_threshold: float = 0.5
) -> List[Dict]:
    """
    Identify comparisons with unusually high divergence.
    
    Parameters
    ----------
    results : AggregatedResults
        Results to scan
    js_threshold : float
        Jensen-Shannon divergence threshold for flagging
        
    Returns
    -------
    list[dict]
        List of outlier comparisons with details
    """
    outliers = []
    
    for r in results.results:
        js = r.metrics.get('jensen_shannon', 0)
        if js > js_threshold:
            outliers.append({
                'subgroup': r.subgroup,
                'target_code': r.target_code,
                'profile_type': r.profile_type,
                'jensen_shannon': js,
                'n_instances': r.n_instances,
                'predicted_dist': r.predicted_dist.tolist(),
                'empirical_dist': r.empirical_dist.tolist(),
            })
    
    return sorted(outliers, key=lambda x: -x['jensen_shannon'])


def compute_overall_metrics(
    country_cond: AggregatedResults,
    country_uncond: AggregatedResults,
    intersect_cond: Optional[AggregatedResults] = None,
    intersect_uncond: Optional[AggregatedResults] = None
) -> Dict[str, Any]:
    """
    Compute overall summary metrics across all analyses.
    """
    summary = {
        'country_conditioned': {
            'n_comparisons': len(country_cond.results),
            'mean_js': country_cond.mean_js,
            'median_js': country_cond.median_js,
            'std_js': country_cond.std_js,
        },
        'country_unconditioned': {
            'n_comparisons': len(country_uncond.results),
            'mean_js': country_uncond.mean_js,
            'median_js': country_uncond.median_js,
            'std_js': country_uncond.std_js,
        },
    }
    
    if intersect_cond and intersect_cond.results:
        summary['intersectional_conditioned'] = {
            'n_comparisons': len(intersect_cond.results),
            'mean_js': intersect_cond.mean_js,
            'median_js': intersect_cond.median_js,
            'std_js': intersect_cond.std_js,
            'exploratory': True,
        }
    
    if intersect_uncond and intersect_uncond.results:
        summary['intersectional_unconditioned'] = {
            'n_comparisons': len(intersect_uncond.results),
            'mean_js': intersect_uncond.mean_js,
            'median_js': intersect_uncond.median_js,
            'std_js': intersect_uncond.std_js,
            'exploratory': True,
        }
    
    return summary


# =============================================================================
# Profile Type Comparison
# =============================================================================

def compare_across_profile_types(
    results: AggregatedResults
) -> Dict[str, Any]:
    """
    Compare how performance varies across profile richness levels.
    
    Tests hypothesis: richer profiles → better predictions
    """
    by_type = defaultdict(list)
    
    for r in results.results:
        by_type[r.profile_type].append(r.metrics.get('jensen_shannon', 0))
    
    comparison = {}
    for ptype, js_values in sorted(by_type.items()):
        comparison[ptype] = {
            'n': len(js_values),
            'mean_js': np.mean(js_values),
            'median_js': np.median(js_values),
            'std_js': np.std(js_values),
        }
    
    # Statistical test: does richness matter?
    if len(by_type) >= 2:
        from scipy import stats
        types_sorted = sorted(by_type.keys())
        values_list = [by_type[t] for t in types_sorted]
        
        if all(len(v) >= 3 for v in values_list):
            # Kruskal-Wallis test (non-parametric)
            stat, pvalue = stats.kruskal(*values_list)
            comparison['kruskal_wallis'] = {
                'statistic': stat,
                'p_value': pvalue,
                'significant_005': pvalue < 0.05,
            }
    
    return comparison


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    instances: List[Dict],
    output_dir: Path,
    min_n: int = 30,
    js_threshold: float = 0.5,
    survey_type: str = 'generic',
    gender_values: List[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate complete evaluation report.
    
    Parameters
    ----------
    instances : list[dict]
        Prediction instances with results
    output_dir : Path
        Directory for output files
    min_n : int
        Minimum sample size for comparisons
    js_threshold : float
        Threshold for flagging outliers
    survey_type : str
        Survey type for feature mapping ('ess', 'wvs', 'generic')
    gender_values : list[str]
        Gender categories for intersectional analysis
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Complete report data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Distribution Comparison Evaluation")
        print(f"{'='*60}")
        print(f"Loaded {len(instances)} instances")
        print(f"Minimum sample size: {min_n}")
        print(f"Output directory: {output_dir}")
    
    # Setup comparator
    comparator = DistributionComparator(
        instances=instances,
        min_sample_size=min_n
    )
    
    # Get feature mapper for survey type
    if survey_type == 'ess':
        feature_mapper = FeatureMapper.for_ess()
    elif survey_type == 'wvs':
        feature_mapper = FeatureMapper.for_wvs()
    else:
        feature_mapper = FeatureMapper()
    
    # Summary stats
    n_countries = len(comparator.by_country)
    n_targets = len(comparator.by_target)
    n_profile_types = len(comparator.by_profile_type)
    
    if verbose:
        print(f"\nData summary:")
        print(f"  Countries: {n_countries}")
        print(f"  Targets: {n_targets}")
        print(f"  Profile types: {n_profile_types}")
    
    # Run analyses
    country_cond = run_country_analysis(comparator, conditioned=True, verbose=verbose)
    country_uncond = run_country_analysis(comparator, conditioned=False, verbose=verbose)
    
    intersect_cond = run_intersectional_analysis(
        comparator, 
        gender_values=gender_values,
        conditioned=True, 
        verbose=verbose
    )
    intersect_uncond = run_intersectional_analysis(
        comparator,
        gender_values=gender_values,
        conditioned=False,
        verbose=verbose
    )
    
    # Summaries
    country_cond_summaries = summarize_by_dimensions(country_cond, comparator)
    country_uncond_summaries = summarize_by_dimensions(country_uncond, comparator)
    
    # Profile type comparison
    profile_comparison = compare_across_profile_types(country_uncond)
    
    # Outliers
    outliers_cond = identify_outliers(country_cond, js_threshold)
    outliers_uncond = identify_outliers(country_uncond, js_threshold)
    
    # Overall metrics
    overall = compute_overall_metrics(
        country_cond, country_uncond, intersect_cond, intersect_uncond
    )
    
    # Build report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_instances': len(instances),
            'n_countries': n_countries,
            'n_targets': n_targets,
            'n_profile_types': n_profile_types,
            'min_sample_size': min_n,
            'js_outlier_threshold': js_threshold,
            'survey_type': survey_type,
        },
        'overall_summary': overall,
        'profile_type_comparison': profile_comparison,
        'country_conditioned': {
            'summary': country_cond.to_dict()['summary'],
            'by_country': country_cond_summaries['by_country'],
            'by_target': country_cond_summaries['by_target'],
            'by_profile_type': country_cond_summaries['by_profile_type'],
            'outliers': outliers_cond[:20],  # Top 20
        },
        'country_unconditioned': {
            'summary': country_uncond.to_dict()['summary'],
            'by_country': country_uncond_summaries['by_country'],
            'by_target': country_uncond_summaries['by_target'],
            'by_profile_type': country_uncond_summaries['by_profile_type'],
            'outliers': outliers_uncond[:20],
        },
    }
    
    # Add intersectional if we have results
    if intersect_cond.results:
        report['intersectional_conditioned'] = {
            'exploratory': True,
            'summary': intersect_cond.to_dict()['summary'],
            'n_comparisons': len(intersect_cond.results),
        }
    
    if intersect_uncond.results:
        report['intersectional_unconditioned'] = {
            'exploratory': True,
            'summary': intersect_uncond.to_dict()['summary'],
            'n_comparisons': len(intersect_uncond.results),
        }
    
    # Save outputs
    save_results_json(report, output_dir / 'summary_report.json')
    
    # Save detailed results
    save_results_json(
        country_cond.to_dict(), 
        output_dir / 'country_conditioned_full.json'
    )
    save_results_json(
        country_uncond.to_dict(),
        output_dir / 'country_unconditioned_full.json'
    )
    
    if intersect_uncond.results:
        save_results_json(
            intersect_uncond.to_dict(),
            output_dir / 'intersectional_unconditioned_full.json'
        )
    
    if verbose:
        print(f"\n{'='*60}")
        print("Evaluation Complete")
        print(f"{'='*60}")
        print(f"Reports saved to: {output_dir}")
        print(f"  - summary_report.json")
        print(f"  - country_conditioned_full.json")
        print(f"  - country_unconditioned_full.json")
        if intersect_uncond.results:
            print(f"  - intersectional_unconditioned_full.json")
    
    return report


# =============================================================================
# Pretty Printing
# =============================================================================

def print_detailed_summary(report: Dict[str, Any]):
    """Print a detailed human-readable summary."""
    print("\n" + "="*70)
    print("DETAILED EVALUATION SUMMARY")
    print("="*70)
    
    meta = report['metadata']
    print(f"\nDataset: {meta['n_instances']} instances")
    print(f"  Countries: {meta['n_countries']}")
    print(f"  Targets: {meta['n_targets']}")
    print(f"  Profile types: {meta['n_profile_types']}")
    
    overall = report['overall_summary']
    
    print("\n" + "-"*70)
    print("COUNTRY-LEVEL ANALYSIS")
    print("-"*70)
    
    for analysis_type in ['country_conditioned', 'country_unconditioned']:
        data = overall.get(analysis_type, {})
        print(f"\n{analysis_type.replace('_', ' ').title()}:")
        print(f"  Comparisons: {data.get('n_comparisons', 0)}")
        print(f"  Mean JS:     {data.get('mean_js', 0):.4f}")
        print(f"  Median JS:   {data.get('median_js', 0):.4f}")
        print(f"  Std JS:      {data.get('std_js', 0):.4f}")
    
    # Profile comparison
    profile_comp = report.get('profile_type_comparison', {})
    if profile_comp:
        print("\n" + "-"*70)
        print("PROFILE RICHNESS COMPARISON")
        print("-"*70)
        
        for ptype in sorted(profile_comp.keys()):
            if ptype == 'kruskal_wallis':
                continue
            data = profile_comp[ptype]
            print(f"\n  {ptype}: Mean JS = {data['mean_js']:.4f} (n={data['n']})")
        
        if 'kruskal_wallis' in profile_comp:
            kw = profile_comp['kruskal_wallis']
            sig = "significant" if kw['significant_005'] else "not significant"
            print(f"\n  Kruskal-Wallis test: p={kw['p_value']:.4f} ({sig} at α=0.05)")
    
    # Top outliers
    print("\n" + "-"*70)
    print("TOP OUTLIERS (High Divergence)")
    print("-"*70)
    
    outliers = report.get('country_unconditioned', {}).get('outliers', [])[:5]
    if outliers:
        for o in outliers:
            print(f"\n  {o['subgroup']} × {o['target_code']} ({o['profile_type']})")
            print(f"    JS = {o['jensen_shannon']:.4f}, n = {o['n_instances']}")
    else:
        print("  No major outliers found.")
    
    # Intersectional (if present)
    if 'intersectional_unconditioned' in report:
        print("\n" + "-"*70)
        print("INTERSECTIONAL ANALYSIS (EXPLORATORY)")
        print("-"*70)
        
        data = report['intersectional_unconditioned']
        summary = data.get('summary', {})
        print(f"\n  Comparisons: {data.get('n_comparisons', 0)}")
        print(f"  Mean JS:     {summary.get('mean_jensen_shannon', 0):.4f}")
        print(f"  Median JS:   {summary.get('median_jensen_shannon', 0):.4f}")
        print("  (Note: Smaller sample sizes - interpret with caution)")
    
    print("\n" + "="*70)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLM survey predictions via distribution comparison'
    )
    parser.add_argument(
        '--predictions', type=str, required=True,
        help='Path to predictions JSONL file'
    )
    parser.add_argument(
        '--output', type=str, default='eval_results',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--min_n', type=int, default=30,
        help='Minimum sample size for comparisons'
    )
    parser.add_argument(
        '--js_threshold', type=float, default=0.5,
        help='JS divergence threshold for outlier detection'
    )
    parser.add_argument(
        '--survey_type', type=str, default='generic',
        choices=['ess', 'wvs', 'generic'],
        help='Survey type for feature mapping'
    )
    parser.add_argument(
        '--gender_values', type=str, nargs='+', default=['Male', 'Female'],
        help='Gender values for intersectional analysis'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    instances = load_instances_jsonl(args.predictions)
    
    # Run evaluation
    report = generate_report(
        instances=instances,
        output_dir=Path(args.output),
        min_n=args.min_n,
        js_threshold=args.js_threshold,
        survey_type=args.survey_type,
        gender_values=args.gender_values,
        verbose=not args.quiet
    )
    
    # Print detailed summary
    if not args.quiet:
        print_detailed_summary(report)


if __name__ == '__main__':
    main()