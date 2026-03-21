#!/usr/bin/env python3
"""
Heterogeneity Analysis Script

Run three-layer heterogeneity analysis:
1. Full-sample: Per-question variance ratios
2. Feature-presence: Effect of profile features on heterogeneity  
3. Traditional subgroups: Demographics-based analysis

Usage:
    # Layer 1 only (just results)
    python analyze_heterogeneity.py results/ --output analysis/
    
    # Layers 1 + 2 (with input data for feature analysis)
    python analyze_heterogeneity.py results/ --inputs inputs/ --output analysis/
    
    # All three layers (with survey data for demographics)
    python analyze_heterogeneity.py results/ --inputs inputs/ --surveys surveys/ --output analysis/

Examples:
    # Basic analysis
    python analyze_heterogeneity.py ../results --output ../analysis
    
    # With feature-presence analysis
    python analyze_heterogeneity.py ../results --inputs outputs/main_data --output ../analysis
    
    # Full analysis with demographics
    python analyze_heterogeneity.py ../results \\
        --inputs outputs/main_data \\
        --surveys ../data/surveys \\
        --demographics demographics.json \\
        --output ../analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from synthetic_sampling.evaluation import (
    load_results,
    HeterogeneityAnalyzer,
    load_input_data_for_heterogeneity,
    load_survey_data,
)


def find_files(directory: Path, pattern: str = "*.jsonl") -> List[Path]:
    """Find files matching pattern in directory."""
    if directory.is_file():
        return [directory]
    return sorted(directory.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description='Run three-layer heterogeneity analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'results',
        type=str,
        help='Path to results directory or JSONL file'
    )
    
    parser.add_argument(
        '--inputs',
        type=str,
        default=None,
        help='Path to input data directory (for Layer 2 feature analysis)'
    )
    
    parser.add_argument(
        '--surveys',
        type=str,
        default=None,
        help='Path to survey data directory (for Layer 3 demographics)'
    )
    
    parser.add_argument(
        '--demographics',
        type=str,
        default=None,
        help='JSON file mapping surveys to demographic variables'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--min-n',
        type=int,
        default=30,
        help='Minimum instances per question (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Load results
    print("=" * 60)
    print("LOADING RESULTS")
    print("=" * 60)
    
    results_path = Path(args.results)
    result_files = find_files(results_path)
    
    if not result_files:
        print(f"No results files found in {results_path}")
        sys.exit(1)
    
    print(f"Found {len(result_files)} result files")
    
    all_instances = []
    for f in result_files:
        instances = load_results(str(f))
        all_instances.extend(instances)
        print(f"  {f.name}: {len(instances):,} instances")
    
    print(f"\nTotal: {len(all_instances):,} instances")
    
    # Load input data for Layer 2 (optional)
    input_data = None
    if args.inputs:
        print("\n" + "=" * 60)
        print("LOADING INPUT DATA (for Layer 2)")
        print("=" * 60)
        
        input_path = Path(args.inputs)
        input_files = find_files(input_path)
        
        if input_files:
            input_data = load_input_data_for_heterogeneity([str(f) for f in input_files])
            print(f"Loaded {len(input_data):,} input records")
        else:
            print(f"No input files found in {input_path}")
    
    # Load survey data for Layer 3 (optional)
    survey_data = None
    demographic_mapping = None
    
    if args.surveys:
        print("\n" + "=" * 60)
        print("LOADING SURVEY DATA (for Layer 3)")
        print("=" * 60)
        
        surveys_path = Path(args.surveys)
        
        # Find survey data files
        survey_files = {}
        for ext in ['.dta', '.sav', '.csv']:
            for f in surveys_path.glob(f'*{ext}'):
                # Infer survey name from filename
                survey_name = f.stem.lower()
                # Common mappings
                if 'ess' in survey_name and '11' in survey_name:
                    survey_name = 'ess_wave_11'
                elif 'ess' in survey_name and '10' in survey_name:
                    survey_name = 'ess_wave_10'
                elif 'wvs' in survey_name:
                    survey_name = 'wvs'
                elif 'afro' in survey_name:
                    survey_name = 'afrobarometer'
                elif 'arab' in survey_name:
                    survey_name = 'arabbarometer'
                elif 'asian' in survey_name or 'abs' in survey_name:
                    survey_name = 'asianbarometer'
                elif 'latino' in survey_name:
                    survey_name = 'latinobarometer'
                
                survey_files[survey_name] = str(f)
                print(f"  Found: {survey_name} -> {f.name}")
        
        if survey_files:
            survey_data = load_survey_data(survey_files)
        
        # Load demographic mapping
        if args.demographics:
            with open(args.demographics, 'r') as f:
                demographic_mapping = json.load(f)
        else:
            # Default demographic mappings
            demographic_mapping = {
                'ess_wave_10': {
                    'gender': 'gndr',
                    'education': 'eisced',
                    'age_group': 'agea',
                },
                'ess_wave_11': {
                    'gender': 'gndr',
                    'education': 'eisced',
                    'age_group': 'agea',
                },
                'wvs': {
                    'gender': 'Q260',
                    'education': 'Q275',
                    'age': 'Q262',
                },
                'afrobarometer': {
                    'gender': 'Q101',
                    'education': 'Q97',
                    'age': 'Q1',
                },
                'arabbarometer': {
                    'gender': 'Q1002',
                    'education': 'Q1003',
                    'age': 'Q1001',
                },
                'latinobarometer': {
                    'gender': 'sexo',
                    'education': 'S12',
                    'age': 'edad',
                },
                'asianbarometer': {
                    'gender': 'SE2',
                    'education': 'SE5',
                    'age': 'SE3_1',
                },
            }
            print("\nUsing default demographic mappings")
    
    # Create analyzer
    print("\n" + "=" * 60)
    print("RUNNING HETEROGENEITY ANALYSIS")
    print("=" * 60)
    
    analyzer = HeterogeneityAnalyzer(
        instances=all_instances,
        input_data=input_data,
        survey_data=survey_data,
    )
    
    # Run full report
    output_path = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'heterogeneity_analysis.json')
    
    report = analyzer.full_report(
        demographic_mapping=demographic_mapping,
        save_path=output_path,
    )
    
    # Print detailed summaries
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    
    # Layer 1 summary
    print("\n### Layer 1: Full-Sample Analysis ###")
    layer1 = report['layer1_full_sample']
    summary = layer1['summary']
    
    print(f"\nVariance Ratio (soft predictions):")
    vr = summary['variance_ratio_soft']
    if vr['mean'] is not None:
        print(f"  Mean:   {vr['mean']:.4f}")
        print(f"  Median: {vr['median']:.4f}")
        print(f"  Std:    {vr['std']:.4f}")
        print(f"  Range:  [{vr['min']:.4f}, {vr['max']:.4f}]")
        print(f"  IQR:    [{vr['q25']:.4f}, {vr['q75']:.4f}]")
        print(f"  % flattening (VR < 1): {vr['pct_flattening']:.1%}")
    
    print(f"\nVariance Ratio (hard predictions):")
    vr_hard = summary['variance_ratio_hard']
    if vr_hard['mean'] is not None:
        print(f"  Mean:   {vr_hard['mean']:.4f}")
        print(f"  % flattening (VR < 1): {vr_hard['pct_flattening']:.1%}")
    
    print(f"\nJensen-Shannon Divergence:")
    js = summary['js_divergence_soft']
    if js['mean'] is not None:
        print(f"  Soft - Mean: {js['mean']:.4f}, Median: {js['median']:.4f}")
    js_hard = summary['js_divergence_hard']
    if js_hard['mean'] is not None:
        print(f"  Hard - Mean: {js_hard['mean']:.4f}, Median: {js_hard['median']:.4f}")
    
    # Layer 2 summary
    print("\n### Layer 2: Feature-Presence Analysis ###")
    layer2 = report['layer2_feature_presence']
    
    if layer2.get('status') == 'complete':
        summary2 = layer2['summary']
        print(f"\nFeatures analyzed: {layer2['n_features_analyzed']}")
        
        if summary2['mean_vr_difference'] is not None:
            print(f"\nVariance Ratio Difference (with - without feature):")
            print(f"  Mean:   {summary2['mean_vr_difference']:+.4f}")
            print(f"  Median: {summary2['median_vr_difference']:+.4f}")
            print(f"  % positive (feature helps): {summary2['pct_positive_effect']:.1%}")
            
            if summary2.get('ttest_vs_zero'):
                tt = summary2['ttest_vs_zero']
                print(f"\n  t-test vs 0: t={tt['statistic']:.2f}, p={tt['pvalue']:.4f}")
            
            print(f"\nInterpretation:")
            print(f"  Positive difference → Feature presence reduces flattening")
            print(f"  Negative difference → Feature presence increases flattening")
        
        # Top effects
        if layer2.get('top_positive_effects'):
            print(f"\nTop 5 features that REDUCE flattening:")
            for i, e in enumerate(layer2['top_positive_effects'][:5], 1):
                print(f"  {i}. Δ={e['vr_difference']:+.4f}: {e['feature'][:60]}...")
        
        if layer2.get('top_negative_effects'):
            print(f"\nTop 5 features that INCREASE flattening:")
            for i, e in enumerate(layer2['top_negative_effects'][:5], 1):
                print(f"  {i}. Δ={e['vr_difference']:+.4f}: {e['feature'][:60]}...")
    else:
        print(f"  Status: {layer2.get('status', 'unknown')}")
        print(f"  {layer2.get('message', '')}")
    
    # Layer 3 summary
    print("\n### Layer 3: Traditional Subgroup Analysis ###")
    layer3 = report['layer3_subgroups']
    
    if layer3.get('status') == 'complete':
        for survey, demo_results in layer3.get('by_survey', {}).items():
            print(f"\n{survey}:")
            
            for demo_name, results in demo_results.items():
                if isinstance(results, dict) and results.get('status') == 'complete':
                    print(f"\n  {demo_name} ({results['variable']}):")
                    
                    for subgroup, metrics in results.get('subgroups', {}).items():
                        vr = metrics.get('variance_ratio_soft')
                        vr_str = f"{vr:.3f}" if vr else "N/A"
                        print(f"    {subgroup}: n={metrics['n']:,}, acc={metrics['accuracy']:.1%}, VR={vr_str}")
                elif isinstance(results, dict):
                    print(f"  {demo_name}: {results.get('status', 'unknown')}")
    else:
        print(f"  Status: {layer3.get('status', 'unknown')}")
        print(f"  {layer3.get('message', '')}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if output_path:
        print(f"\nFull results saved to: {output_path}")


if __name__ == '__main__':
    main()