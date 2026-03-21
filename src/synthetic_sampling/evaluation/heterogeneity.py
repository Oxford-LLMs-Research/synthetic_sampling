"""
Heterogeneity Analysis Module

Three-layer analysis of whether LLM predictions preserve within-group diversity:

1. Full-Sample Analysis: Per-question variance ratios across all respondents
2. Feature-Presence Analysis: Compare heterogeneity when profile contains feature vs not
3. Traditional Subgroup Analysis: Demographics-based subgroup distributions

Key metric: Variance Ratio = Var(predicted) / Var(empirical)
- Values < 1 indicate diversity flattening
- Values ≈ 1 indicate heterogeneity preservation

Usage:
    from synthetic_sampling.evaluation.heterogeneity import HeterogeneityAnalyzer
    
    # Layer 1 only (minimal setup)
    analyzer = HeterogeneityAnalyzer(instances)
    full_sample = analyzer.full_sample_analysis()
    
    # With Layer 2 (need input data)
    input_data = load_input_data_for_heterogeneity(input_paths)
    analyzer = HeterogeneityAnalyzer(instances, input_data=input_data)
    feature_effects = analyzer.feature_presence_analysis()
    
    # With Layer 3 (need survey data)
    survey_data = load_survey_data({'ess_wave_11': 'path/to/ess11.dta'})
    analyzer = HeterogeneityAnalyzer(instances, survey_data=survey_data)
    subgroups = analyzer.traditional_subgroup_analysis({'gender': 'gndr'})
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from scipy import stats

from .evaluation import (
    ParsedInstance, 
    jensen_shannon_divergence,
)


@dataclass
class DistributionComparison:
    """Results of comparing predicted vs empirical distributions."""
    n_instances: int
    n_options: int
    
    # Distributions
    empirical_dist: np.ndarray
    predicted_dist_soft: np.ndarray  # probability-weighted
    predicted_dist_hard: np.ndarray  # argmax counts
    
    # Metrics
    js_divergence_soft: float
    js_divergence_hard: float
    variance_ratio_soft: Optional[float]
    variance_ratio_hard: Optional[float]
    
    # Raw variances
    empirical_variance: float
    predicted_variance_soft: float
    predicted_variance_hard: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_instances': self.n_instances,
            'n_options': self.n_options,
            'js_divergence_soft': round(self.js_divergence_soft, 6),
            'js_divergence_hard': round(self.js_divergence_hard, 6),
            'variance_ratio_soft': round(self.variance_ratio_soft, 4) if self.variance_ratio_soft else None,
            'variance_ratio_hard': round(self.variance_ratio_hard, 4) if self.variance_ratio_hard else None,
            'empirical_variance': round(self.empirical_variance, 6),
            'predicted_variance_soft': round(self.predicted_variance_soft, 6),
            'predicted_variance_hard': round(self.predicted_variance_hard, 6),
        }


class HeterogeneityAnalyzer:
    """
    Three-layer heterogeneity analysis.
    
    Parameters
    ----------
    instances : list[ParsedInstance]
        Parsed prediction instances
    input_data : dict[str, dict], optional
        Mapping of example_id -> input data (with 'questions' field for profile features)
    survey_data : dict[str, pd.DataFrame], optional
        Mapping of survey_name -> DataFrame with full respondent data
    """
    
    def __init__(
        self,
        instances: List[ParsedInstance],
        input_data: Optional[Dict[str, Dict]] = None,
        survey_data: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.instances = instances
        self.input_data = input_data or {}
        self.survey_data = survey_data or {}
        
        # Build indices
        self._by_question: Dict[str, List[ParsedInstance]] = defaultdict(list)
        self._by_survey: Dict[str, List[ParsedInstance]] = defaultdict(list)
        self._by_respondent: Dict[str, List[ParsedInstance]] = defaultdict(list)
        
        for inst in instances:
            key = f"{inst.survey}_{inst.target_code}"
            self._by_question[key].append(inst)
            self._by_survey[inst.survey].append(inst)
            self._by_respondent[f"{inst.survey}_{inst.respondent_id}"].append(inst)
    
    # =========================================================================
    # Core Distribution Computation
    # =========================================================================
    
    def _compute_distribution_comparison(
        self, 
        instances: List[ParsedInstance]
    ) -> Optional[DistributionComparison]:
        """Compute distribution comparison for a set of instances (same question)."""
        if len(instances) < 5:
            return None
        
        options = instances[0].options
        n_opts = len(options)
        opt_to_idx = {o: i for i, o in enumerate(options)}
        
        # Empirical distribution (ground truth)
        emp_counts = np.zeros(n_opts)
        for inst in instances:
            idx = opt_to_idx.get(inst.ground_truth)
            if idx is not None:
                emp_counts[idx] += 1
        
        if emp_counts.sum() == 0:
            return None
        
        emp_dist = emp_counts / emp_counts.sum()
        
        # Soft predicted distribution (probability-weighted)
        pred_soft = np.zeros(n_opts)
        for inst in instances:
            for opt, prob in inst.probs.items():
                idx = opt_to_idx.get(opt)
                if idx is not None:
                    pred_soft[idx] += prob
        pred_soft = pred_soft / len(instances)
        
        # Hard predicted distribution (argmax counts)
        pred_hard = np.zeros(n_opts)
        for inst in instances:
            idx = opt_to_idx.get(inst.predicted)
            if idx is not None:
                pred_hard[idx] += 1
        pred_hard = pred_hard / pred_hard.sum() if pred_hard.sum() > 0 else pred_hard
        
        # Compute variances
        emp_var = np.var(emp_dist)
        pred_var_soft = np.var(pred_soft)
        pred_var_hard = np.var(pred_hard)
        
        vr_soft = pred_var_soft / emp_var if emp_var > 1e-10 else None
        vr_hard = pred_var_hard / emp_var if emp_var > 1e-10 else None
        
        return DistributionComparison(
            n_instances=len(instances),
            n_options=n_opts,
            empirical_dist=emp_dist,
            predicted_dist_soft=pred_soft,
            predicted_dist_hard=pred_hard,
            js_divergence_soft=jensen_shannon_divergence(pred_soft, emp_dist),
            js_divergence_hard=jensen_shannon_divergence(pred_hard, emp_dist),
            variance_ratio_soft=vr_soft,
            variance_ratio_hard=vr_hard,
            empirical_variance=emp_var,
            predicted_variance_soft=pred_var_soft,
            predicted_variance_hard=pred_var_hard,
        )
    
    def _compute_aggregate_metrics(
        self, 
        instances: List[ParsedInstance],
        min_per_question: int = 10,
    ) -> Dict[str, Optional[float]]:
        """Compute average variance ratio and JS divergence across questions."""
        by_question = defaultdict(list)
        for inst in instances:
            key = f"{inst.survey}_{inst.target_code}"
            by_question[key].append(inst)
        
        vr_soft_list = []
        vr_hard_list = []
        js_soft_list = []
        js_hard_list = []
        
        for q_instances in by_question.values():
            if len(q_instances) < min_per_question:
                continue
            
            comp = self._compute_distribution_comparison(q_instances)
            if comp:
                if comp.variance_ratio_soft is not None:
                    vr_soft_list.append(comp.variance_ratio_soft)
                if comp.variance_ratio_hard is not None:
                    vr_hard_list.append(comp.variance_ratio_hard)
                js_soft_list.append(comp.js_divergence_soft)
                js_hard_list.append(comp.js_divergence_hard)
        
        return {
            'variance_ratio_soft': np.mean(vr_soft_list) if vr_soft_list else None,
            'variance_ratio_hard': np.mean(vr_hard_list) if vr_hard_list else None,
            'js_divergence_soft': np.mean(js_soft_list) if js_soft_list else None,
            'js_divergence_hard': np.mean(js_hard_list) if js_hard_list else None,
            'n_questions': len(vr_soft_list),
        }
    
    # =========================================================================
    # Layer 1: Full-Sample Analysis
    # =========================================================================
    
    def full_sample_analysis(self, min_n: int = 30) -> Dict[str, Any]:
        """
        Layer 1: Full-sample heterogeneity analysis.
        
        For each question, compare predicted vs empirical distribution
        across ALL respondents (regardless of profile content).
        
        Parameters
        ----------
        min_n : int
            Minimum instances per question
            
        Returns
        -------
        dict
            Contains 'by_question' (detailed) and 'summary' (aggregated)
        """
        results_by_question = {}
        
        for question_key, insts in self._by_question.items():
            if len(insts) < min_n:
                continue
            
            comparison = self._compute_distribution_comparison(insts)
            if comparison is not None:
                results_by_question[question_key] = comparison
        
        # Collect metrics
        vr_soft = [r.variance_ratio_soft for r in results_by_question.values() 
                   if r.variance_ratio_soft is not None]
        vr_hard = [r.variance_ratio_hard for r in results_by_question.values()
                   if r.variance_ratio_hard is not None]
        js_soft = [r.js_divergence_soft for r in results_by_question.values()]
        js_hard = [r.js_divergence_hard for r in results_by_question.values()]
        
        def summarize(arr, name=""):
            if not arr:
                return {'mean': None, 'median': None, 'std': None, 'n': 0}
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
                'n': len(arr),
                'pct_flattening': sum(1 for v in arr if v < 1.0) / len(arr) if 'variance' in name.lower() or name == '' else None,
            }
        
        return {
            'n_questions': len(results_by_question),
            'n_instances': sum(r.n_instances for r in results_by_question.values()),
            'summary': {
                'variance_ratio_soft': summarize(vr_soft, 'variance'),
                'variance_ratio_hard': summarize(vr_hard, 'variance'),
                'js_divergence_soft': summarize(js_soft, 'js'),
                'js_divergence_hard': summarize(js_hard, 'js'),
            },
            'by_question': {k: v.to_dict() for k, v in results_by_question.items()},
        }
    
    # =========================================================================
    # Layer 2: Feature-Presence Analysis  
    # =========================================================================
    
    def get_profile_features(self) -> Dict[str, Set[str]]:
        """
        Extract which feature questions appear in each instance's profile.
        
        Returns
        -------
        dict
            Mapping of feature_question -> set of example_ids containing that feature
        """
        feature_to_ids = defaultdict(set)
        
        for example_id, data in self.input_data.items():
            questions = data.get('questions', {})
            for question_text in questions.keys():
                # Truncate long questions for cleaner keys
                feature_key = question_text[:150] if len(question_text) > 150 else question_text
                feature_to_ids[feature_key].add(example_id)
        
        return dict(feature_to_ids)
    
    def feature_presence_analysis(
        self,
        min_instances_per_group: int = 500,
        min_feature_freq: float = 0.05,
        max_feature_freq: float = 0.95,
        top_n: int = 50,
    ) -> Dict[str, Any]:
        """
        Layer 2: Feature-presence heterogeneity analysis.
        
        For each feature (question topic) that appears in profiles:
        - Split instances into: feature present vs absent
        - Compare variance ratios between splits
        - Test: Does knowing a feature improve heterogeneity preservation?
        
        Parameters
        ----------
        min_instances_per_group : int
            Minimum instances in both present/absent groups
        min_feature_freq : float
            Minimum frequency for feature to be analyzed
        max_feature_freq : float
            Maximum frequency (exclude near-universal features)
        top_n : int
            Number of top features to return in detail
            
        Returns
        -------
        dict
            Feature effects and summary statistics
        """
        if not self.input_data:
            return {
                'status': 'no_input_data',
                'message': 'Load input data with input_data parameter for feature-presence analysis.',
            }
        
        # Get feature -> example_ids mapping
        feature_to_ids = self.get_profile_features()
        
        if not feature_to_ids:
            return {
                'status': 'no_features_found',
                'message': 'No features found in input data.',
            }
        
        # Build id -> instance lookup
        id_to_instance = {inst.example_id: inst for inst in self.instances}
        all_ids = set(id_to_instance.keys())
        n_total = len(all_ids)
        
        # Filter features by frequency
        valid_features = []
        for feature, ids in feature_to_ids.items():
            ids_in_results = ids & all_ids
            freq = len(ids_in_results) / n_total if n_total > 0 else 0
            
            if min_feature_freq <= freq <= max_feature_freq:
                n_with = len(ids_in_results)
                n_without = n_total - n_with
                if n_with >= min_instances_per_group and n_without >= min_instances_per_group:
                    valid_features.append((feature, ids_in_results))
        
        if not valid_features:
            return {
                'status': 'no_valid_features',
                'message': f'No features meet criteria. Total features: {len(feature_to_ids)}',
                'criteria': {
                    'min_freq': min_feature_freq,
                    'max_freq': max_feature_freq,
                    'min_per_group': min_instances_per_group,
                }
            }
        
        # Analyze each feature
        results = []
        
        for feature, with_ids in valid_features:
            without_ids = all_ids - with_ids
            
            instances_with = [id_to_instance[i] for i in with_ids]
            instances_without = [id_to_instance[i] for i in without_ids]
            
            metrics_with = self._compute_aggregate_metrics(instances_with)
            metrics_without = self._compute_aggregate_metrics(instances_without)
            
            vr_with = metrics_with['variance_ratio_soft']
            vr_without = metrics_without['variance_ratio_soft']
            
            vr_diff = None
            if vr_with is not None and vr_without is not None:
                vr_diff = vr_with - vr_without
            
            results.append({
                'feature': feature,
                'n_with': len(instances_with),
                'n_without': len(instances_without),
                'vr_with': vr_with,
                'vr_without': vr_without,
                'vr_difference': vr_diff,
                'js_with': metrics_with['js_divergence_soft'],
                'js_without': metrics_without['js_divergence_soft'],
                'n_questions_with': metrics_with['n_questions'],
                'n_questions_without': metrics_without['n_questions'],
            })
        
        # Sort by effect size (positive = feature helps)
        results_with_diff = [r for r in results if r['vr_difference'] is not None]
        results_with_diff.sort(key=lambda x: x['vr_difference'], reverse=True)
        
        # Summary statistics
        vr_diffs = [r['vr_difference'] for r in results_with_diff]
        
        # Statistical test: are VR differences significantly different from 0?
        if len(vr_diffs) >= 5:
            ttest = stats.ttest_1samp(vr_diffs, 0)
            wilcoxon = stats.wilcoxon(vr_diffs) if len(vr_diffs) >= 10 else None
        else:
            ttest = None
            wilcoxon = None
        
        return {
            'status': 'complete',
            'n_features_analyzed': len(results_with_diff),
            'summary': {
                'mean_vr_difference': float(np.mean(vr_diffs)) if vr_diffs else None,
                'median_vr_difference': float(np.median(vr_diffs)) if vr_diffs else None,
                'std_vr_difference': float(np.std(vr_diffs)) if vr_diffs else None,
                'pct_positive_effect': sum(1 for d in vr_diffs if d > 0) / len(vr_diffs) if vr_diffs else None,
                'ttest_vs_zero': {
                    'statistic': float(ttest.statistic) if ttest else None,
                    'pvalue': float(ttest.pvalue) if ttest else None,
                } if ttest else None,
                'wilcoxon_vs_zero': {
                    'statistic': float(wilcoxon.statistic) if wilcoxon else None,
                    'pvalue': float(wilcoxon.pvalue) if wilcoxon else None,
                } if wilcoxon else None,
            },
            'interpretation': {
                'positive_vr_difference': 'Feature presence INCREASES predicted variance relative to empirical (less flattening)',
                'negative_vr_difference': 'Feature presence DECREASES predicted variance relative to empirical (more flattening)',
                'hypothesis': 'If models use information effectively, VR should be higher (less flattening) when relevant features are present.',
            },
            'top_positive_effects': results_with_diff[:top_n],
            'top_negative_effects': results_with_diff[-top_n:][::-1],
            'all_effects': results_with_diff,
        }
    
    # =========================================================================
    # Layer 3: Traditional Subgroup Analysis
    # =========================================================================
    
    def traditional_subgroup_analysis(
        self,
        demographic_mapping: Dict[str, Dict[str, str]],
        min_subgroup_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Layer 3: Traditional demographic subgroup analysis.
        
        Compare heterogeneity preservation within canonical demographic subgroups,
        using ground-truth demographics from original survey data.
        
        Parameters
        ----------
        demographic_mapping : dict
            Nested mapping of survey -> {demo_name: variable_name}
            Example:
            {
                'ess_wave_11': {'gender': 'gndr', 'education': 'eisced'},
                'wvs': {'gender': 'Q260', 'education': 'Q275'},
            }
        min_subgroup_size : int
            Minimum instances per subgroup
            
        Returns
        -------
        dict
            Subgroup-level heterogeneity results
        """
        if not self.survey_data:
            return {
                'status': 'no_survey_data',
                'message': 'Load survey data with survey_data parameter.',
            }
        
        results = {}
        
        for survey, demo_vars in demographic_mapping.items():
            if survey not in self.survey_data:
                results[survey] = {'status': 'survey_not_loaded'}
                continue
            
            if survey not in self._by_survey:
                results[survey] = {'status': 'no_instances_for_survey'}
                continue
            
            survey_instances = self._by_survey[survey]
            survey_df = self.survey_data[survey]
            
            results[survey] = self._analyze_survey_demographics(
                survey_instances, survey_df, demo_vars, min_subgroup_size
            )
        
        return {
            'status': 'complete',
            'by_survey': results,
        }
    
    def _analyze_survey_demographics(
        self,
        instances: List[ParsedInstance],
        df: pd.DataFrame,
        demo_vars: Dict[str, str],
        min_subgroup_size: int,
    ) -> Dict[str, Any]:
        """Analyze demographics for one survey."""
        # Try to find ID column
        id_col = self._find_id_column(df)
        if id_col is None:
            return {'status': 'no_id_column_found', 'columns': list(df.columns)[:20]}
        
        # Build lookup: respondent_id -> demographics
        resp_demographics = {}
        for _, row in df.iterrows():
            resp_id = str(row[id_col])
            resp_demographics[resp_id] = {
                demo_name: row.get(var_name) 
                for demo_name, var_name in demo_vars.items()
                if var_name in df.columns
            }
        
        # Analyze each demographic variable
        demo_results = {}
        
        for demo_name, var_name in demo_vars.items():
            if var_name not in df.columns:
                demo_results[demo_name] = {'status': 'variable_not_found', 'variable': var_name}
                continue
            
            # Group instances by demographic value
            by_value = defaultdict(list)
            missing = 0
            
            for inst in instances:
                resp_demo = resp_demographics.get(inst.respondent_id, {})
                value = resp_demo.get(demo_name)
                
                if value is not None and pd.notna(value):
                    by_value[str(value)].append(inst)
                else:
                    missing += 1
            
            # Filter to subgroups with enough data
            valid_subgroups = {
                k: v for k, v in by_value.items() 
                if len(v) >= min_subgroup_size
            }
            
            if not valid_subgroups:
                demo_results[demo_name] = {
                    'status': 'insufficient_subgroups',
                    'n_missing': missing,
                    'subgroup_sizes': {k: len(v) for k, v in by_value.items()},
                }
                continue
            
            # Compute metrics per subgroup
            subgroup_metrics = {}
            for value, insts in valid_subgroups.items():
                metrics = self._compute_aggregate_metrics(insts)
                acc = sum(1 for i in insts if i.correct) / len(insts)
                
                subgroup_metrics[value] = {
                    'n': len(insts),
                    'accuracy': round(acc, 4),
                    'variance_ratio_soft': round(metrics['variance_ratio_soft'], 4) if metrics['variance_ratio_soft'] else None,
                    'js_divergence_soft': round(metrics['js_divergence_soft'], 4) if metrics['js_divergence_soft'] else None,
                    'n_questions': metrics['n_questions'],
                }
            
            demo_results[demo_name] = {
                'status': 'complete',
                'variable': var_name,
                'n_subgroups': len(subgroup_metrics),
                'n_missing': missing,
                'subgroups': subgroup_metrics,
            }
        
        return demo_results
    
    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the respondent ID column in a dataframe."""
        candidates = ['idno', 'IDNO', 'id', 'ID', 'respondent_id', 'respid', 'caseid']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    # =========================================================================
    # Combined Analysis
    # =========================================================================
    
    def full_report(
        self,
        demographic_mapping: Optional[Dict[str, Dict[str, str]]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run all three layers of analysis.
        
        Parameters
        ----------
        demographic_mapping : dict, optional
            For Layer 3. Maps survey -> {demo_name: var_name}
        save_path : str, optional
            Path to save JSON results
            
        Returns
        -------
        dict
            Complete three-layer analysis
        """
        print("=" * 60)
        print("HETEROGENEITY ANALYSIS")
        print("=" * 60)
        
        # Layer 1
        print("\n[Layer 1] Full-sample analysis...")
        layer1 = self.full_sample_analysis()
        summary = layer1['summary']
        print(f"  Questions: {layer1['n_questions']}")
        if summary['variance_ratio_soft']['mean'] is not None:
            print(f"  Mean VR (soft): {summary['variance_ratio_soft']['mean']:.3f}")
            print(f"  % showing flattening: {summary['variance_ratio_soft']['pct_flattening']:.1%}")
        
        # Layer 2
        print("\n[Layer 2] Feature-presence analysis...")
        layer2 = self.feature_presence_analysis()
        if layer2['status'] == 'complete':
            print(f"  Features analyzed: {layer2['n_features_analyzed']}")
            if layer2['summary']['mean_vr_difference'] is not None:
                print(f"  Mean VR difference: {layer2['summary']['mean_vr_difference']:+.4f}")
                print(f"  % with positive effect: {layer2['summary']['pct_positive_effect']:.1%}")
        else:
            print(f"  Status: {layer2['status']}")
        
        # Layer 3
        print("\n[Layer 3] Traditional subgroup analysis...")
        if demographic_mapping:
            layer3 = self.traditional_subgroup_analysis(demographic_mapping)
            for survey, results in layer3.get('by_survey', {}).items():
                if isinstance(results, dict):
                    for demo, demo_results in results.items():
                        if isinstance(demo_results, dict) and demo_results.get('status') == 'complete':
                            print(f"  {survey}/{demo}: {demo_results['n_subgroups']} subgroups")
        else:
            layer3 = {'status': 'demographic_mapping_not_provided'}
            print("  Skipped (no demographic_mapping provided)")
        
        report = {
            'layer1_full_sample': layer1,
            'layer2_feature_presence': layer2,
            'layer3_subgroups': layer3,
        }
        
        if save_path:
            # Clean for JSON serialization
            def clean_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(v) for v in obj]
                return obj
            
            with open(save_path, 'w') as f:
                json.dump(clean_for_json(report), f, indent=2)
            print(f"\n✓ Saved to {save_path}")
        
        return report


# =============================================================================
# Utility Functions
# =============================================================================

def load_input_data_for_heterogeneity(input_paths: List[str]) -> Dict[str, Dict]:
    """
    Load input JSONL files for feature-presence analysis.
    
    Parameters
    ----------
    input_paths : list[str]
        Paths to input JSONL files
        
    Returns
    -------
    dict
        Mapping of example_id -> full input data dict
    """
    all_data = {}
    
    for path in input_paths:
        path = Path(path)
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    example_id = data.get('example_id')
                    if example_id:
                        all_data[example_id] = data
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(all_data):,} input records")
    return all_data


def load_survey_data(
    survey_paths: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    """
    Load survey data files for traditional subgroup analysis.
    
    Parameters
    ----------
    survey_paths : dict
        Mapping of survey_name -> file path
        
    Returns
    -------
    dict
        Mapping of survey_name -> DataFrame
    """
    survey_data = {}
    
    for survey_name, path in survey_paths.items():
        path = Path(path)
        
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
        
        ext = path.suffix.lower()
        
        try:
            if ext == '.csv':
                df = pd.read_csv(path)
            elif ext == '.dta':
                df = pd.read_stata(path)
            elif ext in ['.sav', '.por']:
                try:
                    import pyreadstat
                    df, _ = pyreadstat.read_sav(str(path))
                except ImportError:
                    df = pd.read_spss(path)
            else:
                print(f"Warning: Unknown format {ext} for {path}")
                continue
            
            survey_data[survey_name] = df
            print(f"Loaded {survey_name}: {len(df):,} rows")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return survey_data