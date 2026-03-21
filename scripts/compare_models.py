#!/usr/bin/env python3
"""
Multi-Model Comparison Analysis

Compare LLM survey prediction performance across multiple models.
Analyzes model family effects, size effects, and consistency of findings.

Usage:
    python compare_models.py model1_results/ model2_results/ model3_results/ --output comparison/
    
    # With model metadata
    python compare_models.py results/*/ --model-info models.json --output comparison/
    
    # With input data for full analysis
    python compare_models.py results/*/ --inputs inputs/ --output comparison/

Model naming convention (auto-parsed from folder names):
    - claude-3-5-sonnet-20241022
    - gpt-4o-2024-08-06  
    - llama-3.1-70b-instruct
    - mistral-large-2411
    
Or provide models.json:
    {
        "claude-3-5-sonnet": {"family": "claude", "size": "medium", "params_b": null},
        "llama-3.1-70b": {"family": "llama", "size": "70b", "params_b": 70}
    }
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from synthetic_sampling.evaluation import (
    load_results,
    ResultsAnalyzer,
    enrich_instances_with_metadata,
)


@dataclass
class ModelInfo:
    """Metadata about a model."""
    name: str
    family: str = "unknown"
    size: str = "unknown"
    params_b: Optional[float] = None  # Parameters in billions
    
    @classmethod
    def from_name(cls, name: str) -> 'ModelInfo':
        """Parse model info from folder/model name."""
        name_lower = name.lower().replace('_', '-').replace('.', '-')
        
        # Detect family
        if 'claude' in name_lower:
            family = 'claude'
        elif 'gpt' in name_lower:
            family = 'gpt'
        elif 'llama' in name_lower:
            family = 'llama'
        elif 'mistral' in name_lower:
            family = 'mistral'
        elif 'gemini' in name_lower:
            family = 'gemini'
        elif 'qwen' in name_lower:
            family = 'qwen'
        elif 'deepseek' in name_lower:
            family = 'deepseek'
        elif 'olmo' in name_lower:
            family = 'olmo'
        elif 'phi' in name_lower:
            family = 'phi'
        else:
            family = 'other'
        
        # Detect size - try multiple patterns
        params_b = None
        size = 'unknown'
        
        # Pattern: 70b, 8b, 32b, 4b, 7b etc (with or without hyphen/underscore before)
        size_match = re.search(r'[-_]?(\d+)b', name_lower)
        if size_match:
            params_b = float(size_match.group(1))
            size = f"{int(params_b)}B"
        # Pattern for DeepSeek v3 (estimated ~670B MoE, but active params ~37B)
        elif 'deepseek' in name_lower and 'v3' in name_lower:
            size = 'large'
            params_b = 685  # Total params (MoE)
        # GPT patterns
        elif 'gpt-4o-mini' in name_lower or '4o-mini' in name_lower:
            size = 'small'
            params_b = None
        elif 'gpt-4o' in name_lower or 'gpt-4' in name_lower or 'gpt-oss' in name_lower:
            size = 'large'
            params_b = None
        elif 'gpt-3-5' in name_lower or 'turbo' in name_lower:
            size = 'medium'
            params_b = None
        # Claude patterns
        elif 'haiku' in name_lower:
            size = 'small'
            params_b = None
        elif 'sonnet' in name_lower:
            size = 'medium'
            params_b = None
        elif 'opus' in name_lower:
            size = 'large'
            params_b = None
        # Generic size keywords
        elif 'small' in name_lower or 'mini' in name_lower:
            size = 'small'
        elif 'large' in name_lower:
            size = 'large'
        
        return cls(name=name, family=family, size=size, params_b=params_b)


@dataclass
class ModelResults:
    """Results for one model."""
    model_info: ModelInfo
    analyzer: ResultsAnalyzer
    metrics: Dict[str, Any] = field(default_factory=dict)


def find_result_files(path: Path) -> List[Path]:
    """Find JSONL result files in a directory."""
    if path.is_file() and path.suffix == '.jsonl':
        return [path]
    return list(path.glob('*.jsonl'))


def load_model_results(
    model_paths: List[Path],
    model_info_file: Optional[Path] = None,
    input_paths: Optional[List[Path]] = None,
) -> Dict[str, ModelResults]:
    """
    Load results for multiple models.
    
    Parameters
    ----------
    model_paths : list[Path]
        Paths to model result directories or files
    model_info_file : Path, optional
        JSON file with model metadata
    input_paths : list[Path], optional
        Paths to input files for metadata enrichment
        
    Returns
    -------
    dict
        Mapping of model_name -> ModelResults
    """
    # Load model info if provided
    model_info_map = {}
    if model_info_file and model_info_file.exists():
        with open(model_info_file) as f:
            info_data = json.load(f)
        for name, info in info_data.items():
            model_info_map[name] = ModelInfo(
                name=name,
                family=info.get('family', 'unknown'),
                size=info.get('size', 'unknown'),
                params_b=info.get('params_b'),
            )
    
    # Build list of input file paths for metadata enrichment
    input_file_paths = []
    if input_paths:
        for path in input_paths:
            if path.is_dir():
                input_file_paths.extend([str(f) for f in path.glob('*.jsonl')])
            elif path.suffix == '.jsonl':
                input_file_paths.append(str(path))
    
    results = {}
    
    for model_path in model_paths:
        # Determine model name from path
        if model_path.is_dir():
            model_name = model_path.name
        else:
            model_name = model_path.stem
        
        # Clean up model name
        model_name = model_name.replace('_results', '').replace('_output', '')
        
        print(f"\nLoading {model_name}...")
        
        # Find and load result files
        result_files = find_result_files(model_path)
        if not result_files:
            print(f"  Warning: No result files found in {model_path}")
            continue
        
        all_instances = []
        for f in result_files:
            instances = load_results(str(f))
            all_instances.extend(instances)
        
        print(f"  Loaded {len(all_instances):,} instances from {len(result_files)} files")
        
        # Enrich with metadata if available
        if input_file_paths:
            all_instances = enrich_instances_with_metadata(all_instances, input_file_paths)
        
        # Get model info
        if model_name in model_info_map:
            model_info = model_info_map[model_name]
        else:
            model_info = ModelInfo.from_name(model_name)
        
        print(f"  Family: {model_info.family}, Size: {model_info.size}")
        
        # Create analyzer
        analyzer = ResultsAnalyzer(all_instances)
        
        results[model_name] = ModelResults(
            model_info=model_info,
            analyzer=analyzer,
        )
    
    return results


def compute_all_metrics(model_results: Dict[str, ModelResults]) -> Dict[str, Dict[str, Any]]:
    """Compute comprehensive metrics for all models."""
    all_metrics = {}
    
    for model_name, results in model_results.items():
        print(f"\nComputing metrics for {model_name}...")
        analyzer = results.analyzer
        
        # Core metrics
        overall = analyzer.overall_metrics()
        baselines = analyzer.compute_baselines()
        
        metrics = {
            'n_instances': overall.n,
            'accuracy': overall.accuracy,
            'top2_accuracy': overall.top2_accuracy,
            'top3_accuracy': overall.top3_accuracy,
            'mean_log_loss': overall.mean_log_loss,
            'mean_prob_correct': overall.mean_prob_correct,
            'macro_f1': overall.macro_f1,
            'brier_score': overall.brier_score,
            
            # Baselines
            'random_baseline': baselines['random_baseline'],
            'majority_baseline': baselines['majority_baseline'],
            'accuracy_vs_random': baselines['model_vs_random'],
            'accuracy_vs_majority': baselines['model_vs_majority'],
            
            # Model info
            'family': results.model_info.family,
            'size': results.model_info.size,
            'params_b': results.model_info.params_b,
        }
        
        # Calibration
        cal = analyzer.calibration_curve(n_bins=10)
        metrics['ece'] = cal['ece']
        
        # Yes/No bias
        yn = analyzer.yes_no_bias_analysis()
        if 'status' in yn and yn['status'] == 'no_yes_no_questions_found':
            # Skip if no yes/no questions found
            pass
        elif 'bias' in yn:
            # Process yes/no bias results
            metrics['yes_bias'] = yn['bias']['yes_bias']
            metrics['yes_no_accuracy'] = yn['accuracy']['overall']
        
        # Heterogeneity
        hetero = analyzer.heterogeneity_analysis()
        if hetero['variance_ratio_soft']['mean'] is not None:
            metrics['variance_ratio_soft'] = hetero['variance_ratio_soft']['mean']
            metrics['flattening_rate'] = hetero['flattening_rate_soft']
            metrics['js_divergence'] = hetero['js_divergence_soft']['mean']
        
        # Profile richness effect
        profile_effect = analyzer.test_profile_richness_effect()
        if profile_effect['status'] == 'complete':
            acc_data = profile_effect['accuracy']
            profile_keys = sorted(acc_data.keys())
            if len(profile_keys) >= 2:
                metrics['sparse_accuracy'] = acc_data[profile_keys[0]]
                metrics['rich_accuracy'] = acc_data[profile_keys[-1]]
                metrics['profile_effect'] = acc_data[profile_keys[-1]] - acc_data[profile_keys[0]]
                metrics['profile_effect_pvalue'] = profile_effect['ttest_sparse_vs_rich']['p_value']
        
        # By survey breakdown
        by_survey = analyzer.metrics_by_survey()
        metrics['by_survey'] = {s: m.accuracy for s, m in by_survey.items()}
        
        all_metrics[model_name] = metrics
        results.metrics = metrics
    
    return all_metrics


def compare_models_statistical(
    model_results: Dict[str, ModelResults],
    metric: str = 'correct',
) -> Dict[str, Any]:
    """
    Statistical comparison across models using paired tests.
    
    Since all models are evaluated on the same instances,
    we can do paired comparisons.
    """
    model_names = list(model_results.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        return {'status': 'need_multiple_models'}
    
    # Build instance-level data for each model
    model_instance_data = {}
    
    for model_name, results in model_results.items():
        # Create dict of example_id -> metric value
        instance_metrics = {}
        for inst in results.analyzer.instances:
            if metric == 'correct':
                instance_metrics[inst.example_id] = 1 if inst.correct else 0
            elif metric == 'log_loss':
                instance_metrics[inst.example_id] = inst.log_loss
            elif metric == 'prob_correct':
                instance_metrics[inst.example_id] = inst.prob_correct
        
        model_instance_data[model_name] = instance_metrics
    
    # Find common instances across all models
    common_ids = set.intersection(*[set(d.keys()) for d in model_instance_data.values()])
    
    if len(common_ids) < 100:
        return {
            'status': 'insufficient_overlap',
            'common_instances': len(common_ids),
        }
    
    print(f"\nComparing models on {len(common_ids):,} common instances")
    
    # Pairwise comparisons
    pairwise_results = []
    
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            # Get paired data
            values_a = [model_instance_data[model_a][id_] for id_ in common_ids]
            values_b = [model_instance_data[model_b][id_] for id_ in common_ids]
            
            # Paired t-test
            ttest = stats.ttest_rel(values_a, values_b)
            
            # McNemar test (for binary metrics)
            if metric == 'correct':
                # Count disagreements
                a_correct_b_wrong = sum(1 for a, b in zip(values_a, values_b) if a == 1 and b == 0)
                a_wrong_b_correct = sum(1 for a, b in zip(values_a, values_b) if a == 0 and b == 1)
                
                if a_correct_b_wrong + a_wrong_b_correct > 0:
                    mcnemar = stats.binom_test(
                        min(a_correct_b_wrong, a_wrong_b_correct),
                        a_correct_b_wrong + a_wrong_b_correct,
                        0.5
                    ) if hasattr(stats, 'binom_test') else None
                    
                    # Compute exact McNemar
                    n = a_correct_b_wrong + a_wrong_b_correct
                    if n >= 25:
                        chi2 = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1)**2 / n
                        mcnemar_p = 1 - stats.chi2.cdf(chi2, 1)
                    else:
                        mcnemar_p = None
                else:
                    mcnemar_p = 1.0
            else:
                mcnemar_p = None
            
            pairwise_results.append({
                'model_a': model_a,
                'model_b': model_b,
                'mean_a': np.mean(values_a),
                'mean_b': np.mean(values_b),
                'difference': np.mean(values_a) - np.mean(values_b),
                'ttest_statistic': ttest.statistic,
                'ttest_pvalue': ttest.pvalue,
                'mcnemar_pvalue': mcnemar_p,
                'significant_005': ttest.pvalue < 0.05,
                'winner': model_a if np.mean(values_a) > np.mean(values_b) else model_b,
            })
    
    # Overall ANOVA-style test
    all_groups = [[model_instance_data[m][id_] for id_ in common_ids] for m in model_names]
    
    # Friedman test (non-parametric repeated measures)
    if len(model_names) >= 3:
        friedman = stats.friedmanchisquare(*all_groups)
    else:
        friedman = None
    
    return {
        'status': 'complete',
        'n_common_instances': len(common_ids),
        'metric': metric,
        'pairwise_comparisons': pairwise_results,
        'friedman_test': {
            'statistic': friedman.statistic if friedman else None,
            'pvalue': friedman.pvalue if friedman else None,
        } if friedman else None,
    }


def analyze_size_effect(
    model_results: Dict[str, ModelResults],
    all_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze effect of model size on performance."""
    # Get models with known parameter counts
    models_with_size = [
        (name, metrics['params_b'], metrics['accuracy'])
        for name, metrics in all_metrics.items()
        if metrics.get('params_b') is not None
    ]
    
    if len(models_with_size) < 3:
        # Try categorical size analysis
        size_groups = defaultdict(list)
        for name, metrics in all_metrics.items():
            size = metrics.get('size', 'unknown')
            if size != 'unknown':
                size_groups[size].append(metrics['accuracy'])
        
        if len(size_groups) >= 2:
            return {
                'status': 'categorical',
                'by_size': {s: {'mean': np.mean(accs), 'n': len(accs)} 
                           for s, accs in size_groups.items()},
            }
        
        return {'status': 'insufficient_size_data'}
    
    # Continuous analysis
    sizes = [s for _, s, _ in models_with_size]
    accs = [a for _, _, a in models_with_size]
    
    # Correlation
    corr, corr_p = stats.pearsonr(sizes, accs)
    
    # Log-linear fit (accuracy often scales with log of parameters)
    log_sizes = np.log10(sizes)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, accs)
    
    return {
        'status': 'complete',
        'n_models': len(models_with_size),
        'correlation': {
            'pearson_r': corr,
            'pvalue': corr_p,
        },
        'log_linear_fit': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'pvalue': p_value,
        },
        'interpretation': f"Each 10x increase in parameters → {slope*100:.1f}pp accuracy change",
        'models': [(n, s, a) for n, s, a in models_with_size],
    }


def check_conclusion_consistency(
    all_metrics: Dict[str, Dict[str, Any]],
    reference_conclusions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check if key conclusions hold across all models.
    
    Key conclusions from initial analysis:
    1. Accuracy > random baseline
    2. Accuracy < majority baseline (underperformance on common answers)
    3. Yes-bias exists
    4. Richer profiles don't help much / hurt
    5. Diversity flattening occurs (variance ratio < 1)
    6. Model is underconfident (ECE pattern)
    """
    conclusions = {
        'beats_random': [],
        'beats_majority': [],
        'has_yes_bias': [],
        'profile_helps': [],
        'shows_flattening': [],
        'is_underconfident': [],
    }
    
    for model_name, metrics in all_metrics.items():
        # 1. Beats random baseline
        conclusions['beats_random'].append({
            'model': model_name,
            'holds': metrics.get('accuracy_vs_random', 0) > 0,
            'margin': metrics.get('accuracy_vs_random'),
        })
        
        # 2. Beats majority baseline
        conclusions['beats_majority'].append({
            'model': model_name,
            'holds': metrics.get('accuracy_vs_majority', 0) > 0,
            'margin': metrics.get('accuracy_vs_majority'),
        })
        
        # 3. Yes-bias
        yes_bias = metrics.get('yes_bias')
        if yes_bias is not None:
            conclusions['has_yes_bias'].append({
                'model': model_name,
                'holds': yes_bias > 10,  # >10% bias
                'bias': yes_bias,
            })
        
        # 4. Profile effect
        profile_effect = metrics.get('profile_effect')
        if profile_effect is not None:
            conclusions['profile_helps'].append({
                'model': model_name,
                'holds': profile_effect > 0.01,  # >1pp improvement
                'effect': profile_effect,
                'pvalue': metrics.get('profile_effect_pvalue'),
            })
        
        # 5. Flattening
        vr = metrics.get('variance_ratio_soft')
        if vr is not None:
            conclusions['shows_flattening'].append({
                'model': model_name,
                'holds': vr < 1.0,
                'variance_ratio': vr,
            })
        
        # 6. Underconfidence (ECE > 0.1 and mean_prob < accuracy)
        ece = metrics.get('ece')
        prob = metrics.get('mean_prob_correct')
        acc = metrics.get('accuracy')
        if ece is not None and prob is not None:
            conclusions['is_underconfident'].append({
                'model': model_name,
                'holds': prob < acc,
                'ece': ece,
                'prob_vs_acc': prob - acc,
            })
    
    # Summarize consistency
    summary = {}
    for conclusion_name, results in conclusions.items():
        if results:
            n_holds = sum(1 for r in results if r['holds'])
            summary[conclusion_name] = {
                'consistent': n_holds == len(results),
                'n_holds': n_holds,
                'n_total': len(results),
                'percentage': n_holds / len(results),
                'details': results,
            }
    
    return summary


def create_comparison_table(all_metrics: Dict[str, Dict[str, Any]]) -> str:
    """Create formatted comparison table."""
    # Sort by accuracy
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
    
    lines = []
    lines.append("=" * 120)
    lines.append("MODEL COMPARISON TABLE")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    header = f"{'Model':<35} {'Family':<10} {'Size':<8} {'Acc':>7} {'Top2':>7} {'F1':>6} {'ECE':>6} {'VR':>6} {'YesBias':>8}"
    lines.append(header)
    lines.append("-" * 120)
    
    for model_name, metrics in sorted_models:
        acc = metrics.get('accuracy', 0) * 100
        top2 = metrics.get('top2_accuracy', 0) * 100
        f1 = metrics.get('macro_f1', 0)
        ece = metrics.get('ece', 0)
        vr = metrics.get('variance_ratio_soft', 0)
        yes_bias = metrics.get('yes_bias', 0)
        
        line = f"{model_name:<35} {metrics.get('family', '?'):<10} {metrics.get('size', '?'):<8} "
        line += f"{acc:>6.1f}% {top2:>6.1f}% {f1:>5.3f} {ece:>5.3f} {vr:>5.3f} {yes_bias:>+7.1f}%"
        lines.append(line)
    
    lines.append("-" * 120)
    
    # Baselines
    first_metrics = list(all_metrics.values())[0]
    random_bl = first_metrics.get('random_baseline', 0) * 100
    majority_bl = first_metrics.get('majority_baseline', 0) * 100
    lines.append(f"\nBaselines: Random={random_bl:.1f}%, Majority={majority_bl:.1f}%")
    
    return "\n".join(lines)


def create_by_survey_comparison(all_metrics: Dict[str, Dict[str, Any]]) -> str:
    """Create comparison table by survey."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("ACCURACY BY SURVEY")
    lines.append("=" * 100)
    
    # Get all surveys
    all_surveys = set()
    for metrics in all_metrics.values():
        all_surveys.update(metrics.get('by_survey', {}).keys())
    all_surveys = sorted(all_surveys)
    
    # Header
    header = f"{'Model':<30} " + " ".join(f"{s[:12]:>12}" for s in all_surveys)
    lines.append(header)
    lines.append("-" * 100)
    
    for model_name, metrics in sorted(all_metrics.items()):
        by_survey = metrics.get('by_survey', {})
        row = f"{model_name:<30} "
        row += " ".join(f"{by_survey.get(s, 0)*100:>11.1f}%" for s in all_surveys)
        lines.append(row)
    
    return "\n".join(lines)


def save_results(
    output_dir: Path,
    all_metrics: Dict[str, Dict[str, Any]],
    statistical_comparison: Dict[str, Any],
    size_effect: Dict[str, Any],
    consistency: Dict[str, Any],
):
    """Save all comparison results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main metrics JSON
    with open(output_dir / 'model_comparison.json', 'w') as f:
        json.dump({
            'metrics': all_metrics,
            'statistical_comparison': statistical_comparison,
            'size_effect': size_effect,
            'conclusion_consistency': consistency,
        }, f, indent=2, default=str)
    
    # CSV for easy plotting
    import csv
    with open(output_dir / 'model_metrics.csv', 'w', newline='') as f:
        if all_metrics:
            fieldnames = ['model'] + list(list(all_metrics.values())[0].keys())
            # Remove nested dicts
            fieldnames = [f for f in fieldnames if f != 'by_survey']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for model_name, metrics in all_metrics.items():
                row = {'model': model_name, **metrics}
                writer.writerow(row)
    
    print(f"\n✓ Saved results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple LLM models on survey prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'model_paths',
        nargs='+',
        type=str,
        help='Paths to model result directories'
    )
    
    parser.add_argument(
        '--model-info',
        type=str,
        default=None,
        help='JSON file with model metadata'
    )
    
    parser.add_argument(
        '--inputs',
        type=str,
        default=None,
        help='Path to input data for metadata enrichment'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./comparison',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-MODEL COMPARISON")
    print("=" * 60)
    
    # Parse paths and expand glob patterns (needed for Windows compatibility)
    expanded_paths = []
    for pattern in args.model_paths:
        # Check if pattern contains glob characters
        if '*' in pattern or '?' in pattern or '[' in pattern:
            # Expand glob pattern
            matches = glob.glob(pattern)
            if not matches:
                print(f"Warning: No matches found for pattern: {pattern}")
            expanded_paths.extend(matches)
        else:
            # Regular path, use as-is
            expanded_paths.append(pattern)
    
    model_paths = [Path(p) for p in expanded_paths]
    model_info_file = Path(args.model_info) if args.model_info else None
    input_paths = [Path(args.inputs)] if args.inputs else None
    output_dir = Path(args.output)
    
    # Load all model results
    print("\n[1/5] Loading model results...")
    model_results = load_model_results(model_paths, model_info_file, input_paths)
    
    if len(model_results) == 0:
        print("No models loaded!")
        sys.exit(1)
    
    print(f"\nLoaded {len(model_results)} models")
    
    # Compute metrics
    print("\n[2/5] Computing metrics for all models...")
    all_metrics = compute_all_metrics(model_results)
    
    # Statistical comparison
    print("\n[3/5] Running statistical comparisons...")
    statistical_comparison = compare_models_statistical(model_results, metric='correct')
    
    # Size effect
    print("\n[4/5] Analyzing size effects...")
    size_effect = analyze_size_effect(model_results, all_metrics)
    
    # Consistency check
    print("\n[5/5] Checking conclusion consistency...")
    consistency = check_conclusion_consistency(all_metrics)
    
    # Print results
    print("\n" + create_comparison_table(all_metrics))
    print(create_by_survey_comparison(all_metrics))
    
    # Statistical comparison summary
    if statistical_comparison.get('status') == 'complete':
        print("\n" + "=" * 60)
        print("STATISTICAL COMPARISONS (Paired tests on common instances)")
        print("=" * 60)
        print(f"\nCommon instances: {statistical_comparison['n_common_instances']:,}")
        
        print("\nPairwise comparisons:")
        for comp in statistical_comparison['pairwise_comparisons']:
            sig = "***" if comp['ttest_pvalue'] < 0.001 else "**" if comp['ttest_pvalue'] < 0.01 else "*" if comp['ttest_pvalue'] < 0.05 else ""
            print(f"  {comp['model_a']} vs {comp['model_b']}: Δ={comp['difference']*100:+.2f}pp (p={comp['ttest_pvalue']:.4f}){sig}")
    
    # Size effect summary
    if size_effect.get('status') == 'complete':
        print("\n" + "=" * 60)
        print("MODEL SIZE EFFECT")
        print("=" * 60)
        print(f"\nCorrelation (size vs accuracy): r={size_effect['correlation']['pearson_r']:.3f}, p={size_effect['correlation']['pvalue']:.4f}")
        print(f"Log-linear fit R²: {size_effect['log_linear_fit']['r_squared']:.3f}")
        print(f"Interpretation: {size_effect['interpretation']}")
    elif size_effect.get('status') == 'categorical':
        print("\n" + "=" * 60)
        print("MODEL SIZE EFFECT (Categorical)")
        print("=" * 60)
        for size, data in size_effect['by_size'].items():
            print(f"  {size}: mean accuracy = {data['mean']*100:.1f}% (n={data['n']})")
    
    # Consistency summary
    print("\n" + "=" * 60)
    print("CONCLUSION CONSISTENCY ACROSS MODELS")
    print("=" * 60)
    
    for conclusion, data in consistency.items():
        status = "✓ CONSISTENT" if data['consistent'] else "✗ INCONSISTENT"
        print(f"\n{conclusion}: {status} ({data['n_holds']}/{data['n_total']} models)")
        
        if not data['consistent']:
            for detail in data['details']:
                if not detail['holds']:
                    print(f"    Exception: {detail['model']}")
    
    # Save
    save_results(output_dir, all_metrics, statistical_comparison, size_effect, consistency)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()