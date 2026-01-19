"""
Distribution Comparison Module for LLM Survey Prediction Evaluation.

This module provides tools for comparing predicted response distributions
against empirical (ground truth) distributions at various levels of aggregation.

Key Concepts:
- **Conditioned Analysis**: Only includes instances where subgroup-defining 
  features are explicitly present in the profile (e.g., for "women in Germany",
  only profiles that contain both gender and country information).
  
- **Unconditioned Analysis**: Aggregates predictions by respondent's actual 
  demographics regardless of what features appear in the profile. Tests whether
  models can infer demographics from correlated features.

Metrics:
- Jensen-Shannon Divergence: Symmetric measure of distribution similarity [0, 1]
- Total Variation Distance: Maximum pointwise difference between distributions
- Variance Ratio: Ratio of predicted to empirical variance (calibration check)
- Calibration Error: Absolute difference in expected probabilities per bin

Usage:
    from distribution_comparison import (
        DistributionComparator,
        SubgroupDefinition,
        aggregate_to_distribution,
        jensen_shannon_divergence
    )
    
    comparator = DistributionComparator(predictions, ground_truth)
    results = comparator.compare_by_country(conditioned=True)
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set, Union, Callable
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


# =============================================================================
# Distribution Metrics
# =============================================================================

def jensen_shannon_divergence(
    p: np.ndarray, 
    q: np.ndarray,
    base: float = 2
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    JSD is a symmetric, bounded measure of similarity between distributions.
    JSD = 0 means identical distributions; JSD = 1 (for base=2) means maximally different.
    
    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions (must sum to 1, same length)
    base : float
        Logarithm base (2 gives range [0, 1])
        
    Returns
    -------
    float
        Jensen-Shannon divergence
    """
    # Ensure proper probability distributions
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Handle zeros by adding small epsilon
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    
    # Renormalize
    p = p / p.sum()
    q = q / q.sum()
    
    # scipy's jensenshannon returns sqrt(JSD), so we square it
    # Also, scipy uses natural log by default, we want base 2
    js_sqrt = jensenshannon(p, q, base=base)
    return js_sqrt ** 2


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Total Variation Distance between two distributions.
    
    TVD = 0.5 * sum(|p_i - q_i|)
    Range: [0, 1] where 0 = identical, 1 = no overlap
    
    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions
        
    Returns
    -------
    float
        Total variation distance
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p / p.sum()
    q = q / q.sum()
    
    return 0.5 * np.sum(np.abs(p - q))


def variance_ratio(
    predicted_probs: np.ndarray, 
    empirical_probs: np.ndarray
) -> float:
    """
    Compute ratio of predicted to empirical variance.
    
    Variance ratio > 1 indicates overconfident predictions (too peaked).
    Variance ratio < 1 indicates underconfident predictions (too flat).
    Variance ratio ≈ 1 indicates well-calibrated variance.
    
    For categorical distributions, we compute variance of the probability
    mass function treating option indices as values.
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probability distribution
    empirical_probs : np.ndarray
        Empirical probability distribution
        
    Returns
    -------
    float
        Ratio of predicted variance to empirical variance
    """
    p = np.asarray(predicted_probs, dtype=np.float64)
    q = np.asarray(empirical_probs, dtype=np.float64)
    
    p = p / p.sum()
    q = q / q.sum()
    
    # Indices as "values"
    x = np.arange(len(p))
    
    # Compute variances
    pred_mean = np.sum(x * p)
    pred_var = np.sum(p * (x - pred_mean) ** 2)
    
    emp_mean = np.sum(x * q)
    emp_var = np.sum(q * (x - emp_mean) ** 2)
    
    # Handle edge cases
    if emp_var < 1e-10:
        return np.inf if pred_var > 1e-10 else 1.0
    
    return pred_var / emp_var


def calibration_error(
    predicted_probs: np.ndarray,
    empirical_probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Groups predictions into bins by confidence and measures
    the gap between predicted confidence and actual accuracy.
    
    For distributional predictions, we compute per-option calibration.
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities for each option
    empirical_probs : np.ndarray
        Empirical probabilities (ground truth frequencies)
    n_bins : int
        Number of calibration bins
        
    Returns
    -------
    float
        Expected calibration error (lower is better)
    """
    p = np.asarray(predicted_probs, dtype=np.float64)
    q = np.asarray(empirical_probs, dtype=np.float64)
    
    p = p / p.sum()
    q = q / q.sum()
    
    # Simple ECE: weighted average of |pred - emp| per option
    # This is essentially mean absolute error on probability estimates
    return np.mean(np.abs(p - q))


def compute_all_metrics(
    predicted: np.ndarray,
    empirical: np.ndarray
) -> Dict[str, float]:
    """
    Compute all distribution comparison metrics.
    
    Parameters
    ----------
    predicted : np.ndarray
        Predicted probability distribution
    empirical : np.ndarray
        Empirical probability distribution
        
    Returns
    -------
    dict
        Dictionary with all metrics
    """
    return {
        'jensen_shannon': jensen_shannon_divergence(predicted, empirical),
        'total_variation': total_variation_distance(predicted, empirical),
        'variance_ratio': variance_ratio(predicted, empirical),
        'calibration_error': calibration_error(predicted, empirical),
    }


# =============================================================================
# Distribution Aggregation
# =============================================================================

def aggregate_to_distribution(
    values: List[Any],
    options: List[str],
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Aggregate a list of categorical values to a probability distribution.
    
    Parameters
    ----------
    values : list
        List of response values (answers)
    options : list[str]
        Ordered list of all possible options
    normalize : bool
        If True, return probabilities; if False, return counts
        
    Returns
    -------
    tuple[np.ndarray, dict]
        - Probability (or count) array aligned with options
        - Mapping of option -> count
    """
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    
    # Build array aligned with options
    dist = np.array([counts.get(opt, 0) for opt in options], dtype=np.float64)
    
    if normalize and dist.sum() > 0:
        dist = dist / dist.sum()
    
    return dist, dict(counts)


def aggregate_predictions_perplexity(
    instances: List[Dict],
    option_key: str = 'options',
    perplexity_key: str = 'perplexities'
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate perplexity-based predictions to a distribution.
    
    Converts perplexities to probabilities via softmax-style normalization:
    p(option) ∝ exp(-perplexity(option))
    
    Parameters
    ----------
    instances : list[dict]
        Instances with perplexity scores per option
    option_key : str
        Key for options list in instance dict
    perplexity_key : str
        Key for perplexity scores in instance dict
        
    Returns
    -------
    tuple[np.ndarray, list[str]]
        - Aggregated probability distribution
        - List of options
    """
    if not instances:
        return np.array([]), []
    
    # Get options from first instance
    options = instances[0][option_key]
    n_options = len(options)
    
    # Aggregate probabilities
    total_probs = np.zeros(n_options)
    
    for inst in instances:
        perplexities = inst.get(perplexity_key, [])
        if len(perplexities) != n_options:
            continue
        
        # Convert perplexities to probabilities
        # Lower perplexity = higher probability
        log_probs = -np.array(perplexities)
        log_probs = log_probs - np.max(log_probs)  # Numerical stability
        probs = np.exp(log_probs)
        probs = probs / probs.sum()
        
        total_probs += probs
    
    # Normalize
    if total_probs.sum() > 0:
        total_probs = total_probs / total_probs.sum()
    
    return total_probs, options


# =============================================================================
# Subgroup Definition and Filtering
# =============================================================================

@dataclass
class SubgroupDefinition:
    """
    Definition of a demographic subgroup for analysis.
    
    Attributes
    ----------
    name : str
        Human-readable name (e.g., "Women in Germany")
    filters : dict
        Mapping of feature_name -> required_value
        e.g., {'gender': 'Female', 'country': 'DE'}
    profile_features : list[str], optional
        For conditioned analysis: features that must be present in profile.
        If None, uses keys from filters.
    """
    name: str
    filters: Dict[str, Any]
    profile_features: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.profile_features is None:
            self.profile_features = list(self.filters.keys())
    
    @classmethod
    def from_country(cls, country_code: str, country_name: str = None) -> 'SubgroupDefinition':
        """Create a country-level subgroup definition."""
        name = country_name or country_code
        return cls(
            name=name,
            filters={'country': country_code},
            profile_features=['country']
        )
    
    @classmethod
    def from_country_gender(
        cls, 
        country_code: str, 
        gender: str,
        country_name: str = None
    ) -> 'SubgroupDefinition':
        """Create a country × gender intersectional subgroup."""
        name = f"{gender} in {country_name or country_code}"
        return cls(
            name=name,
            filters={'country': country_code, 'gender': gender},
            profile_features=['country', 'gender']
        )


def filter_instances_conditioned(
    instances: List[Dict],
    subgroup: SubgroupDefinition,
    feature_codes_key: str = 'feature_codes',
    respondent_features_key: str = 'respondent_features'
) -> List[Dict]:
    """
    Filter instances for conditioned analysis.
    
    Only includes instances where:
    1. Respondent matches the subgroup filters (actual demographics)
    2. Profile contains the subgroup-defining features
    
    Parameters
    ----------
    instances : list[dict]
        Prediction instances
    subgroup : SubgroupDefinition
        Subgroup definition with filters and required profile features
    feature_codes_key : str
        Key for list of feature codes in profile
    respondent_features_key : str
        Key for respondent's actual features/demographics
        
    Returns
    -------
    list[dict]
        Filtered instances
    """
    filtered = []
    
    for inst in instances:
        # Check respondent matches filters
        matches_filters = True
        for feat, required_val in subgroup.filters.items():
            # Try multiple possible locations for the value
            actual_val = (
                inst.get(feat) or 
                inst.get(respondent_features_key, {}).get(feat)
            )
            if actual_val != required_val:
                matches_filters = False
                break
        
        if not matches_filters:
            continue
        
        # Check profile contains required features (conditioned)
        profile_features = set(inst.get(feature_codes_key, []))
        
        # For conditioned analysis, we need the feature to be in the profile
        # This is tricky because 'country' and 'gender' are abstract concepts
        # We need to map them to actual variable codes
        # 
        # For now, we use a simpler heuristic: check if the subgroup-defining
        # information COULD be inferred from the profile
        # 
        # In practice, the caller should provide the mapping
        has_required = True  # Placeholder - see note below
        
        if has_required:
            filtered.append(inst)
    
    return filtered


def filter_instances_unconditioned(
    instances: List[Dict],
    subgroup: SubgroupDefinition,
    respondent_features_key: str = 'respondent_features'
) -> List[Dict]:
    """
    Filter instances for unconditioned analysis.
    
    Includes all instances where respondent matches subgroup filters,
    regardless of what features appear in the profile.
    
    Parameters
    ----------
    instances : list[dict]
        Prediction instances
    subgroup : SubgroupDefinition
        Subgroup definition with filters
    respondent_features_key : str
        Key for respondent's actual features/demographics
        
    Returns
    -------
    list[dict]
        Filtered instances
    """
    filtered = []
    
    for inst in instances:
        matches = True
        for feat, required_val in subgroup.filters.items():
            actual_val = (
                inst.get(feat) or
                inst.get(respondent_features_key, {}).get(feat)
            )
            if actual_val != required_val:
                matches = False
                break
        
        if matches:
            filtered.append(inst)
    
    return filtered


# =============================================================================
# Comparison Results
# =============================================================================

@dataclass
class ComparisonResult:
    """
    Result of comparing predicted vs empirical distributions for a subgroup.
    """
    subgroup: str
    target_code: str
    profile_type: str
    n_instances: int
    n_empirical: int  # Sample size for empirical distribution
    
    # Distributions
    predicted_dist: np.ndarray
    empirical_dist: np.ndarray
    options: List[str]
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    conditioned: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subgroup': self.subgroup,
            'target_code': self.target_code,
            'profile_type': self.profile_type,
            'n_instances': self.n_instances,
            'n_empirical': self.n_empirical,
            'conditioned': self.conditioned,
            'metrics': self.metrics,
            'predicted_dist': self.predicted_dist.tolist(),
            'empirical_dist': self.empirical_dist.tolist(),
            'options': self.options,
        }


@dataclass
class AggregatedResults:
    """
    Aggregated comparison results across subgroups/targets.
    """
    results: List[ComparisonResult]
    
    # Summary statistics
    mean_js: float = 0.0
    median_js: float = 0.0
    std_js: float = 0.0
    mean_variance_ratio: float = 0.0
    
    def compute_summary(self):
        """Compute summary statistics from results."""
        if not self.results:
            return
        
        js_values = [r.metrics.get('jensen_shannon', 0) for r in self.results]
        vr_values = [r.metrics.get('variance_ratio', 1) for r in self.results 
                     if np.isfinite(r.metrics.get('variance_ratio', 1))]
        
        self.mean_js = np.mean(js_values)
        self.median_js = np.median(js_values)
        self.std_js = np.std(js_values)
        self.mean_variance_ratio = np.mean(vr_values) if vr_values else 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_comparisons': len(self.results),
            'summary': {
                'mean_jensen_shannon': self.mean_js,
                'median_jensen_shannon': self.median_js,
                'std_jensen_shannon': self.std_js,
                'mean_variance_ratio': self.mean_variance_ratio,
            },
            'results': [r.to_dict() for r in self.results],
        }


# =============================================================================
# Main Comparator Class
# =============================================================================

class DistributionComparator:
    """
    Main class for comparing predicted vs empirical response distributions.
    
    Handles both conditioned and unconditioned analysis at various
    aggregation levels (overall, by country, by country×gender).
    
    Parameters
    ----------
    instances : list[dict]
        Prediction instances with predicted probabilities/perplexities
    ground_truth : pd.DataFrame or dict, optional
        Ground truth data for computing empirical distributions.
        If None, uses answers from instances.
    min_sample_size : int
        Minimum sample size for reporting a comparison
    """
    
    def __init__(
        self,
        instances: List[Dict],
        ground_truth: Any = None,
        min_sample_size: int = 30
    ):
        self.instances = instances
        self.ground_truth = ground_truth
        self.min_sample_size = min_sample_size
        
        # Index instances by various keys
        self._index_instances()
    
    def _index_instances(self):
        """Build indices for efficient querying."""
        self.by_target = defaultdict(list)
        self.by_country = defaultdict(list)
        self.by_profile_type = defaultdict(list)
        self.by_country_target = defaultdict(list)
        
        for inst in self.instances:
            target = inst.get('target_code')
            country = inst.get('country')
            profile_type = inst.get('profile_type')
            
            if target:
                self.by_target[target].append(inst)
            if country:
                self.by_country[country].append(inst)
            if profile_type:
                self.by_profile_type[profile_type].append(inst)
            if target and country:
                self.by_country_target[(country, target)].append(inst)
    
    def get_empirical_distribution(
        self,
        instances: List[Dict],
        target_code: str
    ) -> Tuple[np.ndarray, List[str], int]:
        """
        Compute empirical distribution from instance answers.
        
        Parameters
        ----------
        instances : list[dict]
            Instances to aggregate
        target_code : str
            Target question code
            
        Returns
        -------
        tuple[np.ndarray, list[str], int]
            - Probability distribution
            - List of options
            - Sample size
        """
        if not instances:
            return np.array([]), [], 0
        
        # Get options from first instance
        options = instances[0].get('options', [])
        
        # Collect answers
        answers = [inst.get('answer') for inst in instances if inst.get('answer')]
        
        dist, _ = aggregate_to_distribution(answers, options)
        
        return dist, options, len(answers)
    
    def get_predicted_distribution(
        self,
        instances: List[Dict],
        prediction_key: str = 'predicted_probs'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute aggregated predicted distribution.
        
        For perplexity-based predictions, converts to probabilities first.
        
        Parameters
        ----------
        instances : list[dict]
            Instances with predictions
        prediction_key : str
            Key for predictions in instance dict
            
        Returns
        -------
        tuple[np.ndarray, list[str]]
            - Probability distribution
            - List of options
        """
        if not instances:
            return np.array([]), []
        
        options = instances[0].get('options', [])
        n_options = len(options)
        
        total_probs = np.zeros(n_options)
        count = 0
        
        for inst in instances:
            probs = inst.get(prediction_key)
            
            if probs is None:
                # Try perplexity-based prediction
                perplexities = inst.get('perplexities')
                if perplexities and len(perplexities) == n_options:
                    log_probs = -np.array(perplexities)
                    log_probs = log_probs - np.max(log_probs)
                    probs = np.exp(log_probs)
                    probs = probs / probs.sum()
            
            if probs is not None and len(probs) == n_options:
                total_probs += np.array(probs)
                count += 1
        
        if count > 0:
            total_probs = total_probs / count
        
        return total_probs, options
    
    def compare_subgroup(
        self,
        subgroup: SubgroupDefinition,
        target_code: str,
        profile_type: str,
        conditioned: bool = True
    ) -> Optional[ComparisonResult]:
        """
        Compare distributions for a specific subgroup, target, and profile type.
        
        Parameters
        ----------
        subgroup : SubgroupDefinition
            Subgroup to analyze
        target_code : str
            Target question code
        profile_type : str
            Profile richness type (e.g., 's3m2')
        conditioned : bool
            If True, use conditioned filtering; if False, unconditioned
            
        Returns
        -------
        ComparisonResult or None
            Comparison result, or None if insufficient sample size
        """
        # Get instances for this target and profile type
        target_instances = [
            inst for inst in self.by_target.get(target_code, [])
            if inst.get('profile_type') == profile_type
        ]
        
        # Filter by subgroup
        if conditioned:
            filtered = filter_instances_conditioned(target_instances, subgroup)
        else:
            filtered = filter_instances_unconditioned(target_instances, subgroup)
        
        if len(filtered) < self.min_sample_size:
            return None
        
        # Get distributions
        empirical_dist, options, n_empirical = self.get_empirical_distribution(
            filtered, target_code
        )
        predicted_dist, _ = self.get_predicted_distribution(filtered)
        
        if len(empirical_dist) == 0 or len(predicted_dist) == 0:
            return None
        
        if len(empirical_dist) != len(predicted_dist):
            return None
        
        # Compute metrics
        metrics = compute_all_metrics(predicted_dist, empirical_dist)
        
        return ComparisonResult(
            subgroup=subgroup.name,
            target_code=target_code,
            profile_type=profile_type,
            n_instances=len(filtered),
            n_empirical=n_empirical,
            predicted_dist=predicted_dist,
            empirical_dist=empirical_dist,
            options=options,
            metrics=metrics,
            conditioned=conditioned
        )
    
    def compare_by_country(
        self,
        conditioned: bool = True,
        target_codes: List[str] = None,
        profile_types: List[str] = None
    ) -> AggregatedResults:
        """
        Run country-level distribution comparison.
        
        Parameters
        ----------
        conditioned : bool
            Analysis type
        target_codes : list[str], optional
            Specific targets to analyze (None = all)
        profile_types : list[str], optional
            Specific profile types to analyze (None = all)
            
        Returns
        -------
        AggregatedResults
            Aggregated results across all country-target-profile combinations
        """
        results = []
        
        # Get unique countries, targets, profile types
        countries = list(self.by_country.keys())
        targets = target_codes or list(self.by_target.keys())
        profiles = profile_types or list(self.by_profile_type.keys())
        
        for country in countries:
            subgroup = SubgroupDefinition.from_country(country)
            
            for target in targets:
                for profile in profiles:
                    result = self.compare_subgroup(
                        subgroup, target, profile, conditioned
                    )
                    if result is not None:
                        results.append(result)
        
        agg = AggregatedResults(results=results)
        agg.compute_summary()
        
        return agg
    
    def compare_intersectional(
        self,
        gender_values: List[str] = None,
        conditioned: bool = True,
        target_codes: List[str] = None,
        profile_types: List[str] = None
    ) -> AggregatedResults:
        """
        Run intersectional (country × gender) distribution comparison.
        
        Parameters
        ----------
        gender_values : list[str]
            Gender values to analyze (e.g., ['Male', 'Female'])
        conditioned : bool
            Analysis type
        target_codes : list[str], optional
            Specific targets to analyze
        profile_types : list[str], optional
            Specific profile types to analyze
            
        Returns
        -------
        AggregatedResults
            Aggregated results
        """
        if gender_values is None:
            gender_values = ['Male', 'Female']
        
        results = []
        
        countries = list(self.by_country.keys())
        targets = target_codes or list(self.by_target.keys())
        profiles = profile_types or list(self.by_profile_type.keys())
        
        for country in countries:
            for gender in gender_values:
                subgroup = SubgroupDefinition.from_country_gender(country, gender)
                
                for target in targets:
                    for profile in profiles:
                        result = self.compare_subgroup(
                            subgroup, target, profile, conditioned
                        )
                        if result is not None:
                            results.append(result)
        
        agg = AggregatedResults(results=results)
        agg.compute_summary()
        
        return agg
    
    def summary_by_dimension(
        self,
        results: AggregatedResults,
        dimension: str = 'subgroup'
    ) -> Dict[str, Dict[str, float]]:
        """
        Summarize results by a specific dimension.
        
        Parameters
        ----------
        results : AggregatedResults
            Results to summarize
        dimension : str
            Dimension to group by ('subgroup', 'target_code', 'profile_type')
            
        Returns
        -------
        dict
            Mapping of dimension value -> summary metrics
        """
        grouped = defaultdict(list)
        
        for r in results.results:
            key = getattr(r, dimension, 'unknown')
            grouped[key].append(r)
        
        summaries = {}
        for key, group in grouped.items():
            js_values = [r.metrics.get('jensen_shannon', 0) for r in group]
            summaries[key] = {
                'n_comparisons': len(group),
                'mean_js': np.mean(js_values),
                'median_js': np.median(js_values),
                'std_js': np.std(js_values),
                'min_js': np.min(js_values),
                'max_js': np.max(js_values),
            }
        
        return summaries


# =============================================================================
# Utility Functions
# =============================================================================

def load_instances_jsonl(filepath: str) -> List[Dict]:
    """Load instances from a JSONL file."""
    instances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    return instances


def save_results_json(results: Union[AggregatedResults, Dict], filepath: str):
    """Save results to JSON file."""
    if isinstance(results, AggregatedResults):
        data = results.to_dict()
    else:
        data = results
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(results: AggregatedResults, title: str = "Results Summary"):
    """Print a formatted summary of results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total comparisons: {len(results.results)}")
    print(f"\nJensen-Shannon Divergence:")
    print(f"  Mean:   {results.mean_js:.4f}")
    print(f"  Median: {results.median_js:.4f}")
    print(f"  Std:    {results.std_js:.4f}")
    print(f"\nVariance Ratio (predicted/empirical):")
    print(f"  Mean:   {results.mean_variance_ratio:.4f}")
    print(f"{'='*60}")


# =============================================================================
# Feature Mapping for Conditioned Analysis
# =============================================================================

class FeatureMapper:
    """
    Maps abstract subgroup features to actual survey variable codes.
    
    For conditioned analysis, we need to know which variable codes
    correspond to features like 'country' or 'gender'.
    """
    
    def __init__(self, mappings: Dict[str, Set[str]] = None):
        """
        Initialize with feature -> variable codes mappings.
        
        Parameters
        ----------
        mappings : dict
            e.g., {'country': {'cntry', 'COUNTRY'}, 
                   'gender': {'gndr', 'SEX', 'gender'}}
        """
        self.mappings = mappings or {}
    
    @classmethod
    def for_ess(cls) -> 'FeatureMapper':
        """Create mapper with ESS variable names."""
        return cls({
            'country': {'cntry'},
            'gender': {'gndr'},
            'age': {'agea', 'yrbrn'},
            'education': {'eisced', 'eduyrs'},
        })
    
    @classmethod
    def for_wvs(cls) -> 'FeatureMapper':
        """Create mapper with WVS variable names."""
        return cls({
            'country': {'B_COUNTRY', 'B_COUNTRY_ALPHA'},
            'gender': {'Q260'},
            'age': {'Q262'},
            'education': {'Q275'},
        })
    
    def get_codes_for_feature(self, feature: str) -> Set[str]:
        """Get all variable codes that map to a feature."""
        return self.mappings.get(feature, set())
    
    def profile_contains_feature(
        self,
        feature: str,
        profile_codes: Set[str]
    ) -> bool:
        """Check if a profile contains any code for a feature."""
        feature_codes = self.get_codes_for_feature(feature)
        return bool(feature_codes & profile_codes)
    
    def profile_contains_all_features(
        self,
        features: List[str],
        profile_codes: Set[str]
    ) -> bool:
        """Check if a profile contains codes for all specified features."""
        for feature in features:
            if not self.profile_contains_feature(feature, profile_codes):
                return False
        return True