#!/usr/bin/env python
"""
Quick analysis script for LLM survey prediction results.

Usage:
    # Single file
    python analyze_results.py results.jsonl
    
    # Multiple files
    python analyze_results.py file1.jsonl file2.jsonl file3.jsonl
    
    # All JSONL files in a directory
    python analyze_results.py results_folder/
    
    # With metadata enrichment from input files
    python analyze_results.py results_folder/ --inputs input_folder/
    
    # With options
    python analyze_results.py results_folder/ --output analysis_output/ --detailed --csv
"""

import argparse
import json
from pathlib import Path
from typing import List

# Import the evaluation module
from synthetic_sampling.evaluation import (
    ResultsAnalyzer, 
    analyze_errors, 
    load_results, 
    ParsedInstance,
    enrich_instances_with_metadata,
)


def load_multiple_files(paths: List[str]) -> List[ParsedInstance]:
    """Load and combine results from multiple JSONL files."""
    all_instances = []
    
    for path in paths:
        p = Path(path)
        
        if p.is_dir():
            # Load all .jsonl files in directory
            jsonl_files = sorted(p.glob("*.jsonl"))
            if not jsonl_files:
                print(f"Warning: No .jsonl files found in {p}")
                continue
            
            for f in jsonl_files:
                print(f"  Loading: {f.name}")
                instances = load_results(str(f))
                all_instances.extend(instances)
                print(f"    → {len(instances):,} instances")
        
        elif p.is_file() and p.suffix == '.jsonl':
            print(f"  Loading: {p.name}")
            instances = load_results(str(p))
            all_instances.extend(instances)
            print(f"    → {len(instances):,} instances")
        
        else:
            print(f"Warning: Skipping {path} (not a .jsonl file or directory)")
    
    return all_instances


def find_input_files(paths: List[str]) -> List[str]:
    """Find all input JSONL files from paths."""
    all_files = []
    
    for path in paths:
        p = Path(path)
        
        if p.is_dir():
            all_files.extend([str(f) for f in p.glob("*.jsonl")])
        elif p.is_file() and p.suffix == '.jsonl':
            all_files.append(str(p))
    
    return all_files


def main():
    parser = argparse.ArgumentParser(
        description='Analyze LLM survey prediction results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.jsonl                    # Single file
  %(prog)s *.jsonl                          # Multiple files (shell glob)
  %(prog)s results_folder/                  # All .jsonl in directory
  %(prog)s results_folder/ --output out/    # Save analysis outputs
  %(prog)s results_folder/ --inputs input/  # Enrich with metadata
  %(prog)s results_folder/ --detailed --csv # Full analysis with CSV export
        """
    )
    parser.add_argument('results', type=str, nargs='+', 
                       help='Path(s) to results JSONL file(s) or directory containing them')
    parser.add_argument('--inputs', type=str, nargs='*', default=None,
                       help='Path(s) to input JSONL file(s) for metadata enrichment')
    parser.add_argument('--output', type=str, default=None, help='Output directory for analysis files')
    parser.add_argument('--detailed', action='store_true', help='Include detailed breakdowns')
    parser.add_argument('--csv', action='store_true', help='Export data to CSV for further analysis')
    parser.add_argument('--skip-heterogeneity', action='store_true', 
                       help='Skip heterogeneity analysis (can be slow for large datasets)')
    parser.add_argument('--skip-profile-effect', action='store_true',
                       help='Skip profile richness effect analysis (can be slow)')
    
    args = parser.parse_args()
    
    # Load results from all sources
    print(f"Loading results...")
    all_instances = load_multiple_files(args.results)
    
    if not all_instances:
        print("Error: No instances loaded!")
        return
    
    print(f"\nTotal instances loaded: {len(all_instances):,}")
    
    # Enrich with metadata if input files provided
    if args.inputs:
        print(f"\nEnriching with metadata from input files...")
        input_files = find_input_files(args.inputs)
        print(f"  Found {len(input_files)} input files")
        all_instances = enrich_instances_with_metadata(all_instances, input_files)
    
    # Create analyzer
    analyzer = ResultsAnalyzer(all_instances)
    
    # Report metadata coverage
    if args.inputs or analyzer.has_metadata():
        coverage = analyzer.metadata_coverage()
        print(f"\nMetadata coverage:")
        for field, pct in coverage.items():
            print(f"  {field}: {pct:.1%}")
    
    # Basic summary
    analyzer.print_summary()
    
    # Baselines
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)
    baselines = analyzer.compute_baselines()
    overall = analyzer.overall_metrics()
    print(f"\nRandom baseline:      {baselines['random_baseline']:.2%} (1/n_options)")
    print(f"Majority baseline:    {baselines['majority_baseline']:.2%} (most common answer)")
    print(f"Model accuracy:       {baselines['model_accuracy']:.2%}")
    print(f"\nModel vs random:      {baselines['model_vs_random']:+.2%}")
    print(f"Model vs majority:    {baselines['model_vs_majority']:+.2%}")
    
    # Additional metrics from paper
    if overall.macro_f1 is not None:
        print(f"\nMacro F1:             {overall.macro_f1:.3f}")
    if overall.brier_score is not None:
        print(f"Brier Score:          {overall.brier_score:.3f}")
    
    # Profile richness effect
    if not args.skip_profile_effect:
        print("\n" + "=" * 60)
        print("PROFILE RICHNESS EFFECT")
        print("=" * 60)
        print("Computing profile richness effect (this may take a while)...")
        effect = analyzer.test_profile_richness_effect()
        
        if effect['status'] == 'complete':
            print(f"\nComplete sets: {effect['n_complete_sets']:,}")
            print(f"Profile types: {effect['profile_types']}")
            
            print("\nAccuracy by profile:")
            for profile, acc in effect['accuracy'].items():
                # Try to map to human-readable name
                name_map = {'s3m2': 'sparse (6 feat)', 's4m3': 'medium (12 feat)', 's6m4': 'rich (24 feat)'}
                name = name_map.get(profile, profile)
                print(f"  {name}: {acc:.2%}")
            
            print("\nLog-loss by profile (lower = better):")
            for profile, ll in effect['log_loss'].items():
                name_map = {'s3m2': 'sparse', 's4m3': 'medium', 's6m4': 'rich'}
                name = name_map.get(profile, profile)
                print(f"  {name}: {ll:.3f}")
            
            print("\nStatistical tests (sparse vs rich):")
            mcnemar = effect['mcnemar_sparse_vs_rich']
            ttest = effect['ttest_sparse_vs_rich']
            
            if mcnemar.get('p_value') is not None:
                print(f"  McNemar (accuracy): χ²={mcnemar['statistic']:.2f}, p={mcnemar['p_value']:.4f}")
            print(f"  Paired t-test (log-loss): t={ttest['statistic']:.2f}, p={ttest['p_value']:.4f}")
            
            # Get profile keys dynamically
            log_loss_data = effect['log_loss']
            profile_keys = sorted(log_loss_data.keys())
            
            if len(profile_keys) >= 2:
                sparse_key = profile_keys[0]  # First (sparsest)
                rich_key = profile_keys[-1]   # Last (richest)
                
                sig = "✓ Significant" if ttest['p_value'] < 0.05 else "✗ Not significant"
                direction = "improves" if log_loss_data[sparse_key] > log_loss_data[rich_key] else "worsens"
                print(f"\n  → Richer profile {direction} predictions ({sig} at α=0.05)")
        else:
            print(f"\nInsufficient data for paired analysis.")
            print(f"  Status: {effect.get('status', 'unknown')}")
            if 'complete_sets' in effect:
                print(f"  Complete sets: {effect.get('complete_sets', 0)} (need ≥30)")
            if 'profile_types' in effect:
                print(f"  Profile types found: {effect.get('profile_types', [])}")
    else:
        print("\n" + "=" * 60)
        print("PROFILE RICHNESS EFFECT")
        print("=" * 60)
        print("\nSkipped (use without --skip-profile-effect to include)")
        effect = None
    
    # Calibration
    print("\n" + "=" * 60)
    print("CALIBRATION")
    print("=" * 60)
    cal = analyzer.calibration_curve(n_bins=10)
    print(f"\nExpected Calibration Error (ECE): {cal['ece']:.4f}")
    print("\nCalibration bins:")
    print(f"{'Bin':<12} {'N':>8} {'Pred P':>10} {'Actual Acc':>12}")
    print("-" * 44)
    for b in cal['bins']:
        if b['actual_accuracy'] is not None:
            print(f"{b['bin_start']:.1f}-{b['bin_end']:.1f}     {b['n']:>8,} {b['mean_predicted_prob']:>10.3f} {b['actual_accuracy']:>11.1%}")
        else:
            print(f"{b['bin_start']:.1f}-{b['bin_end']:.1f}     {b['n']:>8,}        -           -")
    
    # Heterogeneity analysis
    if not args.skip_heterogeneity:
        print("\n" + "=" * 60)
        print("HETEROGENEITY PRESERVATION")
        print("=" * 60)
        print("Computing heterogeneity metrics (this may take a while)...")
        hetero = analyzer.heterogeneity_analysis()
    
        print("\nVariance Ratio (predicted/empirical variance):")
        print("  Values < 1 indicate diversity flattening")
        vr_soft = hetero['variance_ratio_soft']
        vr_hard = hetero['variance_ratio_hard']
        
        if vr_soft['mean'] is not None:
            print(f"\n  Soft predictions (probability-based):")
            print(f"    Mean: {vr_soft['mean']:.3f}, Median: {vr_soft['median']:.3f}")
            print(f"    Range: [{vr_soft['min']:.3f}, {vr_soft['max']:.3f}]")
            print(f"    Questions showing flattening: {hetero['flattening_rate_soft']:.1%}")
        
        if vr_hard['mean'] is not None:
            print(f"\n  Hard predictions (argmax):")
            print(f"    Mean: {vr_hard['mean']:.3f}, Median: {vr_hard['median']:.3f}")
            print(f"    Range: [{vr_hard['min']:.3f}, {vr_hard['max']:.3f}]")
            print(f"    Questions showing flattening: {hetero['flattening_rate_hard']:.1%}")
        
        print("\nJensen-Shannon Divergence (predicted vs empirical distribution):")
        js_soft = hetero['js_divergence_soft']
        js_hard = hetero['js_divergence_hard']
        
        if js_soft['mean'] is not None:
            print(f"  Soft: Mean={js_soft['mean']:.4f}, Median={js_soft['median']:.4f} (n={js_soft['n']} questions)")
        if js_hard['mean'] is not None:
            print(f"  Hard: Mean={js_hard['mean']:.4f}, Median={js_hard['median']:.4f}")
    else:
        hetero = None
        print("\n" + "=" * 60)
        print("HETEROGENEITY PRESERVATION")
        print("=" * 60)
        print("\nSkipped (use without --skip-heterogeneity to include)")
    
    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    errors = analyzer.error_analysis(max_examples=5)
    print(f"\nTotal errors: {errors['total_errors']:,} / {errors['total_errors'] + errors['total_correct']:,} ({errors['error_rate']:.1%})")
    
    print("\nTop error patterns (ground truth → predicted):")
    for i, pattern in enumerate(errors['top_error_patterns'][:10], 1):
        print(f"  {i}. '{pattern['ground_truth']}' → '{pattern['predicted']}' ({pattern['count']})")
    
    # =========================================================================
    # DISAGGREGATED ANALYSES
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("ANALYSIS BY NUMBER OF OPTIONS")
    print("=" * 60)
    by_n_opts = analyzer.metrics_by_n_options()
    print(f"\n{'Option Type':<15} {'N':>10} {'Acc':>8} {'Top2':>8} {'LogLoss':>10}")
    print("-" * 55)
    for opt_type, m in by_n_opts.items():
        print(f"{opt_type:<15} {m.n:>10,} {m.accuracy:>7.1%} {m.top2_accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
    
    # Detailed breakdown
    print("\nDetailed (by exact number of options):")
    by_n_detailed = analyzer.metrics_by_n_options_detailed()
    print(f"{'N_opts':>6} {'Count':>10} {'Acc':>8} {'LogLoss':>10}")
    print("-" * 38)
    for n_opts, m in sorted(by_n_detailed.items()):
        print(f"{n_opts:>6} {m.n:>10,} {m.accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
    
    print("\n" + "=" * 60)
    print("RESPONSE TYPE ANALYSIS")
    print("=" * 60)
    response_types = analyzer.response_type_analysis()
    print(f"\n{'Response Type':<25} {'N':>10} {'Acc':>8} {'LogLoss':>10}")
    print("-" * 58)
    for rtype, data in sorted(response_types.items(), key=lambda x: -x[1]['n']):
        m = data['metrics']
        print(f"{rtype:<25} {m['n']:>10,} {m['accuracy']:>7.1%} {m['mean_log_loss']:>10.3f}")
    
    print("\n" + "=" * 60)
    print("YES/NO BIAS ANALYSIS")
    print("=" * 60)
    yn_bias = analyzer.yes_no_bias_analysis()
    if yn_bias.get('status') != 'no_yes_no_questions_found':
        print(f"\nTotal yes/no questions: {yn_bias['n_instances']:,}")
        print(f"\nGround truth distribution:")
        print(f"  Yes: {yn_bias['ground_truth']['yes']:,} ({yn_bias['ground_truth']['yes_rate']:.1%})")
        print(f"  No:  {yn_bias['ground_truth']['no']:,} ({1-yn_bias['ground_truth']['yes_rate']:.1%})")
        print(f"\nModel predictions:")
        print(f"  Yes: {yn_bias['predictions']['yes']:,} ({yn_bias['predictions']['yes_rate']:.1%})")
        print(f"  No:  {yn_bias['predictions']['no']:,} ({1-yn_bias['predictions']['yes_rate']:.1%})")
        print(f"\nConfusion matrix:")
        cm = yn_bias['confusion_matrix']
        print(f"                  Predicted")
        print(f"                  Yes      No")
        print(f"  Actual Yes   {cm['yes_to_yes']:>6,}  {cm['yes_to_no']:>6,}")
        print(f"  Actual No    {cm['no_to_yes']:>6,}  {cm['no_to_no']:>6,}")
        print(f"\nAccuracy:")
        print(f"  Overall: {yn_bias['accuracy']['overall']:.1%}")
        print(f"  When GT=Yes: {yn_bias['accuracy']['when_gt_yes']:.1%}")
        print(f"  When GT=No:  {yn_bias['accuracy']['when_gt_no']:.1%}")
        print(f"\nBias: {yn_bias['bias']['yes_bias']:+.1%} ({yn_bias['bias']['interpretation']})")
        
        # Statistical tests
        if 'statistical_tests' in yn_bias:
            tests = yn_bias['statistical_tests']
            print(f"\nStatistical Tests:")
            chi2 = tests['chi2_independence']
            print(f"  χ² independence test: χ²={chi2['statistic']:.2f}, p={chi2['p_value']:.2e}")
            binom = tests['binomial_bias']
            print(f"  Binomial bias test: p={binom['p_value']:.2e}")
            mcnemar = tests['mcnemar_asymmetry']
            if mcnemar['statistic'] is not None:
                print(f"  McNemar asymmetry test: χ²={mcnemar['statistic']:.2f}, p={mcnemar['p_value']:.2e}")
            print(f"\n  → Yes-bias is {'statistically significant' if binom['p_value'] < 0.001 else 'not significant'} (p < 0.001)")
    else:
        print("\nNo yes/no questions found in dataset.")
    
    # Metadata-based analyses (if available)
    if analyzer.has_metadata():
        print("\n" + "=" * 60)
        print("ANALYSIS BY SECTION")
        print("=" * 60)
        by_section = analyzer.metrics_by_section(min_n=100)
        print(f"\n{'Section':<30} {'N':>10} {'Acc':>8} {'Top2':>8} {'LogLoss':>10}")
        print("-" * 70)
        for section, m in sorted(by_section.items(), key=lambda x: -x[1].n):
            print(f"{section:<30} {m.n:>10,} {m.accuracy:>7.1%} {m.top2_accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
        
        print("\n" + "=" * 60)
        print("ANALYSIS BY TOPIC TAG")
        print("=" * 60)
        by_tag = analyzer.metrics_by_topic_tag(min_n=100)
        print(f"\n{'Topic Tag':<35} {'N':>10} {'Acc':>8} {'LogLoss':>10}")
        print("-" * 68)
        for tag, m in sorted(by_tag.items(), key=lambda x: -x[1].n):
            print(f"{tag:<35} {m.n:>10,} {m.accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
        
        print("\n" + "=" * 60)
        print("ANALYSIS BY RESPONSE FORMAT")
        print("=" * 60)
        by_format = analyzer.metrics_by_response_format(min_n=100)
        print(f"\n{'Format':<20} {'N':>10} {'Acc':>8} {'Top2':>8} {'LogLoss':>10}")
        print("-" * 60)
        for fmt, m in sorted(by_format.items(), key=lambda x: -x[1].n):
            print(f"{fmt:<20} {m.n:>10,} {m.accuracy:>7.1%} {m.top2_accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS BY COUNTRY (Top 20)")
    print("=" * 60)
    by_country = analyzer.metrics_by_country(min_n=500)
    # Sort by count
    sorted_countries = sorted(by_country.items(), key=lambda x: -x[1].n)[:20]
    print(f"\n{'Country':<10} {'N':>10} {'Acc':>8} {'Top2':>8} {'LogLoss':>10}")
    print("-" * 50)
    for country, m in sorted_countries:
        print(f"{country:<10} {m.n:>10,} {m.accuracy:>7.1%} {m.top2_accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
    
    # Detailed breakdowns
    if args.detailed:
        print("\n" + "=" * 60)
        print("DETAILED SURVEY × PROFILE BREAKDOWN")
        print("=" * 60)
        
        by_survey_profile = analyzer.metrics_by_survey_and_profile()
        for survey, profiles in sorted(by_survey_profile.items()):
            print(f"\n{survey}:")
            for profile, m in sorted(profiles.items()):
                name = {'s3m2': 'sparse', 's4m3': 'medium', 's6m4': 'rich'}.get(profile, profile)
                print(f"  {name:>8}: n={m.n:>5,} acc={m.accuracy:>6.1%} ll={m.mean_log_loss:>6.3f}")
    
    # Save outputs
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary JSON
        analyzer.save_summary(output_dir / 'summary.json')
        print(f"\n✓ Saved summary to {output_dir / 'summary.json'}")
        
        # Save error analysis
        with open(output_dir / 'error_analysis.json', 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"✓ Saved error analysis to {output_dir / 'error_analysis.json'}")
        
        # Save calibration
        with open(output_dir / 'calibration.json', 'w') as f:
            json.dump(cal, f, indent=2)
        print(f"✓ Saved calibration to {output_dir / 'calibration.json'}")
        
        # Save disaggregated analyses
        disaggregated = {
            'baselines': baselines,
            'by_n_options': {k: v.to_dict() for k, v in by_n_opts.items()},
            'by_n_options_detailed': {str(k): v.to_dict() for k, v in by_n_detailed.items()},
            'response_types': response_types,
            'yes_no_bias': yn_bias,
            'by_country': {k: v.to_dict() for k, v in by_country.items()},
        }
        
        if hetero is not None:
            disaggregated['heterogeneity'] = hetero
        
        # Add metadata-based analyses if available
        if analyzer.has_metadata():
            disaggregated['by_section'] = {k: v.to_dict() for k, v in by_section.items()}
            disaggregated['by_topic_tag'] = {k: v.to_dict() for k, v in by_tag.items()}
            disaggregated['by_response_format'] = {k: v.to_dict() for k, v in by_format.items()}
            disaggregated['metadata_coverage'] = analyzer.metadata_coverage()
        
        with open(output_dir / 'disaggregated_analysis.json', 'w') as f:
            json.dump(disaggregated, f, indent=2, default=str)
        print(f"✓ Saved disaggregated analysis to {output_dir / 'disaggregated_analysis.json'}")
    
    # Export to CSV
    if args.csv:
        output_dir = Path(args.output) if args.output else Path('.')
        csv_path = output_dir / 'results_data.csv'
        print(f"\nExporting to CSV (this may take a while for large datasets)...")
        df = analyzer.to_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"✓ Exported {len(df):,} instances to {csv_path}")


if __name__ == '__main__':
    main()