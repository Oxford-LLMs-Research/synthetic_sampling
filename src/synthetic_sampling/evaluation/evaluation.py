"""
Evaluation Module for LLM Survey Prediction Results.

Designed to work with the actual results format containing:
- example_id: "{survey}_{respondent}_{target}_{profile_type}"
- ground_truth: actual answer
- predicted: model's prediction  
- option_logprobs: {option: logprob} for each option
- options: list of all options
- correct: boolean

Key Metrics:
1. Instance-level: accuracy, log-loss, top-k accuracy
2. Distribution-level: JS divergence, calibration
3. Comparative: by profile richness, survey, target

Usage:
    from synthetic_sampling.evaluation import ResultsAnalyzer
    
    analyzer = ResultsAnalyzer.from_jsonl("results.jsonl")
    
    # Quick summary
    analyzer.print_summary()
    
    # Detailed breakdowns
    by_profile = analyzer.metrics_by_profile_type()
    by_survey = analyzer.metrics_by_survey()
    
    # Distribution comparison
    js_by_target = analyzer.js_divergence_by_target()
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterator
import numpy as np
from scipy.spatial.distance import jensenshannon


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ParsedInstance:
    """Parsed result instance with extracted metadata."""
    example_id: str
    survey: str
    respondent_id: str
    target_code: str
    profile_type: str  # e.g., 's3m2', 's4m3', 's6m4'
    
    ground_truth: str
    predicted: str
    correct: bool
    
    options: List[str]
    logprobs: Dict[str, float]  # option -> logprob
    probs: Dict[str, float]     # option -> probability (computed)
    
    # Optional metadata (enriched from input files)
    country: Optional[str] = None
    target_section: Optional[str] = None
    target_topic_tag: Optional[str] = None
    target_response_format: Optional[str] = None
    
    @property
    def profile_name(self) -> str:
        """Convert profile type code to name."""
        mapping = {'s3m2': 'sparse', 's4m3': 'medium', 's6m4': 'rich'}
        return mapping.get(self.profile_type, self.profile_type)
    
    @property
    def n_features(self) -> int:
        """Estimate feature count from profile type."""
        match = re.match(r's(\d+)m(\d+)', self.profile_type)
        if match:
            return int(match.group(1)) * int(match.group(2))
        return 0
    
    @property
    def log_loss(self) -> float:
        """Negative log probability of ground truth (lower = better)."""
        logp = self.logprobs.get(self.ground_truth)
        if logp is None:
            return float('inf')
        return -logp
    
    @property
    def prob_correct(self) -> float:
        """Probability assigned to correct answer."""
        return self.probs.get(self.ground_truth, 0.0)
    
    def rank_of_correct(self) -> int:
        """Rank of correct answer (1 = highest prob)."""
        sorted_opts = sorted(self.probs.keys(), key=lambda x: -self.probs[x])
        try:
            return sorted_opts.index(self.ground_truth) + 1
        except ValueError:
            return len(sorted_opts) + 1


@dataclass 
class MetricsSummary:
    """Summary statistics for a group of instances."""
    n: int
    accuracy: float
    mean_log_loss: float
    median_log_loss: float
    top2_accuracy: float
    top3_accuracy: float
    mean_prob_correct: float
    
    # Additional metrics
    macro_f1: Optional[float] = None
    brier_score: Optional[float] = None
    
    # Distribution metrics (if computed)
    mean_js_divergence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'n': self.n,
            'accuracy': round(self.accuracy, 4),
            'mean_log_loss': round(self.mean_log_loss, 4),
            'median_log_loss': round(self.median_log_loss, 4),
            'top2_accuracy': round(self.top2_accuracy, 4),
            'top3_accuracy': round(self.top3_accuracy, 4),
            'mean_prob_correct': round(self.mean_prob_correct, 4),
        }
        if self.macro_f1 is not None:
            d['macro_f1'] = round(self.macro_f1, 4)
        if self.brier_score is not None:
            d['brier_score'] = round(self.brier_score, 4)
        if self.mean_js_divergence is not None:
            d['mean_js_divergence'] = round(self.mean_js_divergence, 4)
        return d


# =============================================================================
# Parsing and Loading
# =============================================================================

def parse_example_id(example_id: str) -> Tuple[str, str, str, str]:
    """
    Parse example_id into components.
    
    Format: "{survey}_{respondent}_{target}_{profile_type}"
    Example: "afrobarometer_BEN1123_Q27B_s3m2"
    Example with underscore in target: "arabbarometer_700512_Q725_4_s3m2"
    
    Returns: (survey, respondent_id, target_code, profile_type)
    """
    # Profile type is always at the end: s\dm\d
    match = re.match(r'^(.+)_(s\d+m\d+)$', example_id)
    if not match:
        raise ValueError(f"Cannot parse example_id: {example_id}")
    
    prefix, profile_type = match.groups()
    
    # Known surveys (check longer ones first to avoid partial matches)
    known_surveys = [
        'ess_wave_11', 'ess_wave_10',  # Multi-word surveys first
        'afrobarometer', 'arabbarometer', 'asianbarometer', 'latinobarometer',
        'wvs'
    ]
    
    # Find which survey this is
    survey = None
    survey_prefix = None
    for s in known_surveys:
        if prefix.startswith(s + '_'):
            survey = s
            survey_prefix = s + '_'
            break
    
    if survey_prefix:
        # Remove survey prefix: remainder is "{respondent_id}_{target_code}"
        remainder = prefix[len(survey_prefix):]
        
        # Target codes typically start with Q, P, S, or are lowercase identifiers
        # Find the split point by looking for where target code likely starts
        # Strategy: work backwards from the end, looking for target code patterns
        
        # Try to find where target code starts
        # Target codes often start with Q/P/S followed by numbers, or are lowercase
        # Respondent IDs are often numeric or alphanumeric but don't start with Q/P/S
        
        # Split on all underscores and work backwards
        parts = remainder.split('_')
        
        if len(parts) == 1:
            # Only one part - must be the target (no respondent ID)
            return (survey, '', remainder, profile_type)
        elif len(parts) == 2:
            # Two parts: likely {respondent}_{target}
            # But target might have underscore, so check if first part looks like respondent
            # If second part starts with Q/P/S or is lowercase, it's likely the target
            if parts[1] and (parts[1][0] in 'QPS' or parts[1][0].islower()):
                return (survey, parts[0], parts[1], profile_type)
            else:
                # Both parts might be target, or first is respondent
                # Assume first is respondent, rest is target
                return (survey, parts[0], parts[1], profile_type)
        else:
            # Multiple parts: need to find where target code starts
            # Target codes typically start with Q/P/S or are lowercase
            # Work backwards to find the first part that looks like a target code start
            target_start_idx = len(parts) - 1
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] and (parts[i][0] in 'QPS' or parts[i][0].islower()):
                    target_start_idx = i
                    break
            
            # Everything before target_start_idx is respondent_id
            # Everything from target_start_idx onwards is target_code
            if target_start_idx == 0:
                # All parts are target code
                target_code = '_'.join(parts)
                return (survey, '', target_code, profile_type)
            else:
                respondent_id = '_'.join(parts[:target_start_idx])
                target_code = '_'.join(parts[target_start_idx:])
                return (survey, respondent_id, target_code, profile_type)
    
    # Fallback: try to parse without known survey
    parts = prefix.rsplit('_', 2)
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2], profile_type)
    elif len(parts) == 2:
        return (parts[0], '', parts[1], profile_type)
    else:
        return (prefix, '', '', profile_type)


def logprobs_to_probs(logprobs: Dict[str, float]) -> Dict[str, float]:
    """
    Convert log probabilities to probabilities via softmax.
    
    Parameters
    ----------
    logprobs : dict
        Mapping of option -> log probability
        
    Returns
    -------
    dict
        Mapping of option -> probability (sums to 1)
    """
    if not logprobs:
        return {}
    
    options = list(logprobs.keys())
    lps = np.array([logprobs[o] for o in options])
    
    # Softmax with numerical stability
    lps_shifted = lps - np.max(lps)
    probs = np.exp(lps_shifted)
    probs = probs / probs.sum()
    
    return {o: float(p) for o, p in zip(options, probs)}


def load_results(filepath: str) -> List[ParsedInstance]:
    """
    Load results from JSONL file.
    
    Parameters
    ----------
    filepath : str
        Path to JSONL results file
        
    Returns
    -------
    list[ParsedInstance]
        Parsed instances
    """
    instances = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue
            
            # Parse example_id
            try:
                survey, resp_id, target, profile_type = parse_example_id(
                    data['example_id']
                )
            except (ValueError, KeyError) as e:
                print(f"Warning: Cannot parse line {line_num}: {e}")
                continue
            
            # Get logprobs (handle both formats)
            logprobs = data.get('option_logprobs', {})
            
            # Convert to probs
            probs = logprobs_to_probs(logprobs)
            
            instance = ParsedInstance(
                example_id=data['example_id'],
                survey=survey,
                respondent_id=resp_id,
                target_code=target,
                profile_type=profile_type,
                ground_truth=data.get('ground_truth', ''),
                predicted=data.get('predicted', ''),
                correct=data.get('correct', False),
                options=data.get('options', []),
                logprobs=logprobs,
                probs=probs,
            )
            
            instances.append(instance)
    
    return instances


def load_input_metadata(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load input JSONL file and extract metadata for each example.
    
    Parameters
    ----------
    filepath : str
        Path to input JSONL file
        
    Returns
    -------
    dict
        Mapping of example_id -> metadata dict
    """
    metadata = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            example_id = data.get('example_id')
            if example_id:
                metadata[example_id] = {
                    'country': data.get('country'),
                    'target_section': data.get('target_section'),
                    'target_topic_tag': data.get('target_topic_tag'),
                    'target_response_format': data.get('target_response_format'),
                }
    
    return metadata


def enrich_instances_with_metadata(
    instances: List[ParsedInstance],
    input_paths: List[str],
) -> List[ParsedInstance]:
    """
    Enrich parsed instances with metadata from input files.
    
    Parameters
    ----------
    instances : list[ParsedInstance]
        List of result instances
    input_paths : list[str]
        Paths to input JSONL files containing metadata
        
    Returns
    -------
    list[ParsedInstance]
        Instances enriched with metadata
    """
    # Load all metadata
    all_metadata = {}
    for path in input_paths:
        all_metadata.update(load_input_metadata(path))
    
    # Enrich instances
    enriched_count = 0
    for inst in instances:
        meta = all_metadata.get(inst.example_id)
        if meta:
            inst.country = meta.get('country')
            inst.target_section = meta.get('target_section')
            inst.target_topic_tag = meta.get('target_topic_tag')
            inst.target_response_format = meta.get('target_response_format')
            enriched_count += 1
    
    print(f"Enriched {enriched_count}/{len(instances)} instances with metadata")
    return instances


# Country code mappings for various surveys
COUNTRY_CODES = {
    # ISO 3166-1 numeric codes (used by WVS, Latinobarometer)
    '4': 'AFG', '8': 'ALB', '12': 'DZA', '20': 'AND', '24': 'AGO',
    '32': 'ARG', '36': 'AUS', '40': 'AUT', '48': 'BHR', '50': 'BGD',
    '51': 'ARM', '56': 'BEL', '68': 'BOL', '70': 'BIH', '72': 'BWA',
    '76': 'BRA', '100': 'BGR', '104': 'MMR', '116': 'KHM', '120': 'CMR',
    '124': 'CAN', '144': 'LKA', '152': 'CHL', '156': 'CHN', '158': 'TWN',
    '170': 'COL', '188': 'CRI', '191': 'HRV', '196': 'CYP', '203': 'CZE',
    '208': 'DNK', '214': 'DOM', '218': 'ECU', '222': 'SLV', '226': 'GNQ',
    '231': 'ETH', '233': 'EST', '246': 'FIN', '250': 'FRA', '268': 'GEO',
    '276': 'DEU', '288': 'GHA', '300': 'GRC', '320': 'GTM', '332': 'HTI',
    '340': 'HND', '344': 'HKG', '348': 'HUN', '352': 'ISL', '356': 'IND',
    '360': 'IDN', '364': 'IRN', '368': 'IRQ', '372': 'IRL', '376': 'ISR',
    '380': 'ITA', '392': 'JPN', '398': 'KAZ', '400': 'JOR', '404': 'KEN',
    '410': 'KOR', '414': 'KWT', '417': 'KGZ', '422': 'LBN', '426': 'LSO',
    '428': 'LVA', '430': 'LBR', '434': 'LBY', '440': 'LTU', '442': 'LUX',
    '454': 'MWI', '458': 'MYS', '466': 'MLI', '484': 'MEX', '496': 'MNG',
    '504': 'MAR', '508': 'MOZ', '512': 'OMN', '516': 'NAM', '524': 'NPL',
    '528': 'NLD', '554': 'NZL', '558': 'NIC', '562': 'NER', '566': 'NGA',
    '578': 'NOR', '586': 'PAK', '591': 'PAN', '600': 'PRY', '604': 'PER',
    '608': 'PHL', '616': 'POL', '620': 'PRT', '630': 'PRI', '634': 'QAT',
    '642': 'ROU', '643': 'RUS', '646': 'RWA', '682': 'SAU', '686': 'SEN',
    '688': 'SRB', '702': 'SGP', '703': 'SVK', '704': 'VNM', '705': 'SVN',
    '710': 'ZAF', '716': 'ZWE', '724': 'ESP', '729': 'SDN', '740': 'SUR',
    '752': 'SWE', '756': 'CHE', '760': 'SYR', '762': 'TJK', '764': 'THA',
    '780': 'TTO', '784': 'ARE', '788': 'TUN', '792': 'TUR', '800': 'UGA',
    '804': 'UKR', '818': 'EGY', '826': 'GBR', '834': 'TZA', '840': 'USA',
    '854': 'BFA', '858': 'URY', '860': 'UZB', '862': 'VEN', '887': 'YEM',
    '894': 'ZMB',
    
    # Afrobarometer country codes (survey-specific numeric)
    # These are internal codes, may differ from ISO
    '1': 'Benin', '2': 'Botswana', '3': 'Ghana', '4': 'Lesotho', 
    '5': 'Malawi', '6': 'Mali', '7': 'Namibia', '8': 'Nigeria',
    '9': 'South Africa', '10': 'Tanzania', '11': 'Uganda', '12': 'Zambia',
    '13': 'Zimbabwe', '14': 'Cape Verde', '15': 'Kenya', '16': 'Mozambique',
    '17': 'Senegal', '18': 'Burkina Faso', '19': 'Liberia', '20': 'Madagascar',
    '21': 'Sierra Leone', '22': 'Benin', '23': 'Mauritius', '24': 'Niger',
    '25': 'Algeria', '26': 'Cameroon', '27': 'Egypt', '28': 'Gabon',
    '29': 'Guinea', '30': 'Ivory Coast', '31': 'Morocco', '32': 'Sudan',
    '33': 'Swaziland', '34': 'Togo', '35': 'Tunisia', '36': 'Burundi',
    
    # Arab Barometer country codes
    # Note: these may conflict with Afrobarometer codes, use survey context
    
    # Common variations with decimals
    '76.0': 'BRA', '170.0': 'COL', '218.0': 'ECU', '604.0': 'PER',
    '32.0': 'ARG', '152.0': 'CHL', '484.0': 'MEX', '858.0': 'URY',
}


def normalize_country(country_raw: Optional[str], survey: str) -> Optional[str]:
    """
    Normalize country codes to ISO-2 or country names.
    
    Parameters
    ----------
    country_raw : str or None
        Raw country value from input data
    survey : str
        Survey name (for survey-specific handling)
        
    Returns
    -------
    str or None
        Normalized country code/name
    """
    if country_raw is None:
        return None
    
    country_str = str(country_raw).strip()
    
    # Remove decimal point for numeric codes
    if country_str.endswith('.0'):
        country_str = country_str[:-2]
    
    # If it's already a country name, return it
    if country_str.isalpha() and len(country_str) > 2:
        return country_str
    
    # If it's already ISO-2, return uppercase
    if len(country_str) == 2 and country_str.isalpha():
        return country_str.upper()
    
    # Check mapping
    if country_str in COUNTRY_CODES:
        return COUNTRY_CODES[country_str]
    
    # Return as-is if no mapping found
    return country_str
    
    return instances


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_instance_metrics(instances: List[ParsedInstance]) -> MetricsSummary:
    """
    Compute aggregate metrics for a list of instances.
    
    Includes:
    - Accuracy, Top-k accuracy
    - Log loss (mean, median)
    - Macro F1 (average F1 across classes)
    - Brier score (calibration metric)
    """
    if not instances:
        return MetricsSummary(
            n=0, accuracy=0, mean_log_loss=0, median_log_loss=0,
            top2_accuracy=0, top3_accuracy=0, mean_prob_correct=0
        )
    
    n = len(instances)
    
    # Accuracy
    correct_count = sum(1 for i in instances if i.correct)
    accuracy = correct_count / n
    
    # Log loss
    log_losses = [i.log_loss for i in instances if i.log_loss != float('inf')]
    mean_ll = np.mean(log_losses) if log_losses else float('inf')
    median_ll = np.median(log_losses) if log_losses else float('inf')
    
    # Top-k accuracy
    ranks = [i.rank_of_correct() for i in instances]
    top2 = sum(1 for r in ranks if r <= 2) / n
    top3 = sum(1 for r in ranks if r <= 3) / n
    
    # Mean probability of correct
    probs_correct = [i.prob_correct for i in instances]
    mean_prob = np.mean(probs_correct)
    
    # Macro F1 (average F1 across all response categories)
    # Group by ground truth, compute precision/recall for each
    from collections import Counter
    gt_counts = Counter(i.ground_truth for i in instances)
    pred_counts = Counter(i.predicted for i in instances)
    
    # For each class, compute TP, FP, FN
    all_classes = set(gt_counts.keys()) | set(pred_counts.keys())
    f1_scores = []
    
    for cls in all_classes:
        tp = sum(1 for i in instances if i.ground_truth == cls and i.predicted == cls)
        fp = sum(1 for i in instances if i.ground_truth != cls and i.predicted == cls)
        fn = sum(1 for i in instances if i.ground_truth == cls and i.predicted != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores) if f1_scores else 0
    
    # Brier score: mean squared error of predicted probabilities
    # For each instance: (1 - p(correct))^2 if correct, p(predicted)^2 if incorrect
    # Simplified: sum over all options of (p_pred - indicator)^2
    # But we'll use: mean((prob_correct - 1)^2) for correct, mean(prob_correct^2) for incorrect
    # Actually, multiclass Brier: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    # Simplified approximation: (1/N) * sum_i (1 - prob_correct_i)^2 + sum_k!=correct (prob_k)^2
    # Even simpler: just use (1 - prob_correct) for correct class contribution
    brier_scores = []
    for inst in instances:
        # Brier score contribution: (1 - p_correct)^2 + sum over incorrect classes of p_k^2
        p_correct = inst.prob_correct
        other_probs = [p for opt, p in inst.probs.items() if opt != inst.ground_truth]
        brier = (1 - p_correct)**2 + sum(p**2 for p in other_probs)
        brier_scores.append(brier)
    
    brier_score = np.mean(brier_scores) if brier_scores else 1.0
    
    return MetricsSummary(
        n=n,
        accuracy=accuracy,
        mean_log_loss=mean_ll,
        median_log_loss=median_ll,
        top2_accuracy=top2,
        top3_accuracy=top3,
        mean_prob_correct=mean_prob,
        macro_f1=macro_f1,
        brier_score=brier_score,
    )


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Returns value in [0, 1] where 0 = identical.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Handle zeros
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # scipy returns sqrt(JSD), so square it
    return jensenshannon(p, q, base=2) ** 2


def compute_distribution_metrics(
    instances: List[ParsedInstance],
    options: List[str]
) -> Dict[str, float]:
    """
    Compute distribution-level metrics (predicted vs empirical).
    
    Parameters
    ----------
    instances : list
        Instances for a specific target question
    options : list
        Ordered list of answer options
        
    Returns
    -------
    dict
        Distribution metrics including JS divergence
    """
    if len(instances) < 5:
        return {'js_divergence': None, 'n': len(instances)}
    
    n_opts = len(options)
    opt_to_idx = {o: i for i, o in enumerate(options)}
    
    # Empirical distribution (ground truth)
    emp_counts = np.zeros(n_opts)
    for inst in instances:
        idx = opt_to_idx.get(inst.ground_truth)
        if idx is not None:
            emp_counts[idx] += 1
    emp_dist = emp_counts / emp_counts.sum() if emp_counts.sum() > 0 else emp_counts
    
    # Predicted distribution (average of model's probability distributions)
    pred_sum = np.zeros(n_opts)
    for inst in instances:
        for opt, prob in inst.probs.items():
            idx = opt_to_idx.get(opt)
            if idx is not None:
                pred_sum[idx] += prob
    pred_dist = pred_sum / len(instances)
    
    # Also compute "hard" predicted distribution (based on argmax predictions)
    hard_counts = np.zeros(n_opts)
    for inst in instances:
        idx = opt_to_idx.get(inst.predicted)
        if idx is not None:
            hard_counts[idx] += 1
    hard_dist = hard_counts / hard_counts.sum() if hard_counts.sum() > 0 else hard_counts
    
    # Compute variance ratio (for heterogeneity preservation)
    # Variance of empirical distribution vs variance of predicted distribution
    # Values < 1 indicate diversity flattening
    emp_variance = np.var(emp_dist) if len(emp_dist) > 1 else 0
    pred_soft_variance = np.var(pred_dist) if len(pred_dist) > 1 else 0
    pred_hard_variance = np.var(hard_dist) if len(hard_dist) > 1 else 0
    
    variance_ratio_soft = pred_soft_variance / emp_variance if emp_variance > 1e-10 else None
    variance_ratio_hard = pred_hard_variance / emp_variance if emp_variance > 1e-10 else None
    
    return {
        'js_divergence_soft': jensen_shannon_divergence(pred_dist, emp_dist),
        'js_divergence_hard': jensen_shannon_divergence(hard_dist, emp_dist),
        'variance_ratio_soft': variance_ratio_soft,
        'variance_ratio_hard': variance_ratio_hard,
        'empirical_variance': emp_variance,
        'predicted_variance_soft': pred_soft_variance,
        'predicted_variance_hard': pred_hard_variance,
        'n': len(instances),
        'empirical_dist': emp_dist.tolist(),
        'predicted_dist_soft': pred_dist.tolist(),
        'predicted_dist_hard': hard_dist.tolist(),
    }


# =============================================================================
# Main Analyzer Class
# =============================================================================

class ResultsAnalyzer:
    """
    Main class for analyzing LLM survey prediction results.
    
    Examples
    --------
    >>> analyzer = ResultsAnalyzer.from_jsonl("results.jsonl")
    >>> analyzer.print_summary()
    >>> 
    >>> # Breakdown by profile type
    >>> by_profile = analyzer.metrics_by_profile_type()
    >>> for profile, metrics in by_profile.items():
    ...     print(f"{profile}: {metrics.accuracy:.2%} accuracy")
    >>>
    >>> # Test if richer profiles help
    >>> analyzer.test_profile_richness_effect()
    """
    
    def __init__(self, instances: List[ParsedInstance]):
        """
        Initialize analyzer with parsed instances.
        
        Parameters
        ----------
        instances : list[ParsedInstance]
            Parsed result instances
        """
        self.instances = instances
        
        # Build indices for fast lookup
        self._by_survey: Dict[str, List[ParsedInstance]] = defaultdict(list)
        self._by_profile: Dict[str, List[ParsedInstance]] = defaultdict(list)
        self._by_target: Dict[str, List[ParsedInstance]] = defaultdict(list)
        self._by_respondent: Dict[str, List[ParsedInstance]] = defaultdict(list)
        
        for inst in instances:
            self._by_survey[inst.survey].append(inst)
            self._by_profile[inst.profile_type].append(inst)
            self._by_target[inst.target_code].append(inst)
            self._by_respondent[inst.respondent_id].append(inst)
    
    @classmethod
    def from_jsonl(cls, filepath: str) -> 'ResultsAnalyzer':
        """Load results from JSONL file."""
        instances = load_results(filepath)
        return cls(instances)
    
    # -------------------------------------------------------------------------
    # Basic Properties
    # -------------------------------------------------------------------------
    
    @property
    def n_instances(self) -> int:
        return len(self.instances)
    
    @property
    def surveys(self) -> List[str]:
        return sorted(self._by_survey.keys())
    
    @property
    def profile_types(self) -> List[str]:
        return sorted(self._by_profile.keys())
    
    @property
    def targets(self) -> List[str]:
        return sorted(self._by_target.keys())
    
    # -------------------------------------------------------------------------
    # Overall Metrics
    # -------------------------------------------------------------------------
    
    def overall_metrics(self) -> MetricsSummary:
        """Compute overall metrics across all instances."""
        return compute_instance_metrics(self.instances)
    
    # -------------------------------------------------------------------------
    # Grouped Metrics
    # -------------------------------------------------------------------------
    
    def metrics_by_profile_type(self) -> Dict[str, MetricsSummary]:
        """Compute metrics grouped by profile richness level."""
        return {
            profile: compute_instance_metrics(insts)
            for profile, insts in sorted(self._by_profile.items())
        }
    
    def metrics_by_survey(self) -> Dict[str, MetricsSummary]:
        """Compute metrics grouped by survey."""
        return {
            survey: compute_instance_metrics(insts)
            for survey, insts in sorted(self._by_survey.items())
        }
    
    def metrics_by_target(self, min_n: int = 30) -> Dict[str, MetricsSummary]:
        """Compute metrics grouped by target question."""
        return {
            target: compute_instance_metrics(insts)
            for target, insts in sorted(self._by_target.items())
            if len(insts) >= min_n
        }
    
    def metrics_by_survey_and_profile(self) -> Dict[str, Dict[str, MetricsSummary]]:
        """Compute metrics grouped by survey × profile type."""
        result = {}
        for survey in self.surveys:
            survey_insts = self._by_survey[survey]
            by_profile = defaultdict(list)
            for inst in survey_insts:
                by_profile[inst.profile_type].append(inst)
            
            result[survey] = {
                profile: compute_instance_metrics(insts)
                for profile, insts in sorted(by_profile.items())
            }
        return result
    
    # -------------------------------------------------------------------------
    # Distribution Analysis
    # -------------------------------------------------------------------------
    
    def js_divergence_by_target(
        self, 
        min_n: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute JS divergence for each target question.
        
        Returns dict of {target: {metrics...}}
        """
        results = {}
        
        for target, insts in self._by_target.items():
            if len(insts) < min_n:
                continue
            
            # Get options from first instance
            options = insts[0].options
            
            metrics = compute_distribution_metrics(insts, options)
            results[target] = metrics
        
        return results
    
    def js_divergence_by_profile_type(
        self, 
        min_n: int = 30
    ) -> Dict[str, float]:
        """
        Compute mean JS divergence per profile type.
        
        Averages across targets within each profile type.
        """
        profile_js = defaultdict(list)
        
        for profile_type, insts in self._by_profile.items():
            # Group by target within this profile
            by_target = defaultdict(list)
            for inst in insts:
                by_target[inst.target_code].append(inst)
            
            for target, target_insts in by_target.items():
                if len(target_insts) < min_n:
                    continue
                
                options = target_insts[0].options
                metrics = compute_distribution_metrics(target_insts, options)
                
                if metrics.get('js_divergence_soft') is not None:
                    profile_js[profile_type].append(metrics['js_divergence_soft'])
        
        return {
            profile: np.mean(js_list) if js_list else None
            for profile, js_list in sorted(profile_js.items())
        }
    
    # -------------------------------------------------------------------------
    # Statistical Tests
    # -------------------------------------------------------------------------
    
    def test_profile_richness_effect(self) -> Dict[str, Any]:
        """
        Test whether richer profiles improve predictions.
        
        Uses paired comparisons across respondent × target combinations
        that have all profile levels present.
        
        Returns
        -------
        dict
            Statistical test results and effect sizes
        """
        from scipy import stats
        
        # Group by base_id (respondent × target)
        by_base = defaultdict(dict)
        for inst in self.instances:
            base_id = f"{inst.survey}_{inst.respondent_id}_{inst.target_code}"
            by_base[base_id][inst.profile_type] = inst
        
        # Detect profile types from data and sort by feature count
        all_profile_types = set()
        for profiles in by_base.values():
            all_profile_types.update(profiles.keys())
        
        # Sort by estimated feature count (parse s{n}m{m} -> n*m)
        def profile_feature_count(p):
            match = re.match(r's(\d+)m(\d+)', p)
            if match:
                return int(match.group(1)) * int(match.group(2))
            return 0
        
        profile_order = sorted(all_profile_types, key=profile_feature_count)
        
        if len(profile_order) < 2:
            return {
                'status': 'insufficient_profile_types',
                'profile_types_found': list(all_profile_types),
            }
        
        # Find complete sets (instances that have all profile types)
        complete = [
            base for base, profiles in by_base.items()
            if all(p in profiles for p in profile_order)
        ]
        
        if len(complete) < 30:
            return {
                'status': 'insufficient_data',
                'complete_sets': len(complete),
                'required': 30,
                'profile_types': profile_order,
            }
        
        # Extract paired data for each profile level
        profile_data = {p: {'correct': [], 'log_loss': []} for p in profile_order}
        
        for base_id in complete:
            profiles = by_base[base_id]
            for p in profile_order:
                profile_data[p]['correct'].append(int(profiles[p].correct))
                profile_data[p]['log_loss'].append(profiles[p].log_loss)
        
        # McNemar tests for accuracy differences
        def mcnemar_test(a, b):
            """Test if accuracy differs significantly between two conditions."""
            a_not_b = sum(1 for i in range(len(a)) if a[i] and not b[i])
            b_not_a = sum(1 for i in range(len(a)) if b[i] and not a[i])
            
            if a_not_b + b_not_a < 10:
                return {'statistic': None, 'p_value': None, 'note': 'too_few_discordant'}
            
            chi2 = (abs(a_not_b - b_not_a) - 1) ** 2 / (a_not_b + b_not_a)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
            
            return {
                'statistic': chi2,
                'p_value': p_value,
                'a_better': a_not_b,
                'b_better': b_not_a,
            }
        
        # Get sparse and rich (first and last in sorted order)
        sparse_key = profile_order[0]
        rich_key = profile_order[-1]
        
        sparse_correct = profile_data[sparse_key]['correct']
        rich_correct = profile_data[rich_key]['correct']
        sparse_ll = profile_data[sparse_key]['log_loss']
        rich_ll = profile_data[rich_key]['log_loss']
        
        # Paired t-test for sparse vs rich
        sparse_rich_ttest = stats.ttest_rel(sparse_ll, rich_ll)
        
        # Build results
        result = {
            'status': 'complete',
            'n_complete_sets': len(complete),
            'profile_types': profile_order,
            
            # Accuracy by profile
            'accuracy': {
                p: np.mean(profile_data[p]['correct']) 
                for p in profile_order
            },
            
            # Log loss by profile
            'log_loss': {
                p: np.mean(profile_data[p]['log_loss'])
                for p in profile_order
            },
            
            # Statistical tests (sparse vs rich)
            'mcnemar_sparse_vs_rich': mcnemar_test(sparse_correct, rich_correct),
            'ttest_sparse_vs_rich': {
                'statistic': sparse_rich_ttest.statistic,
                'p_value': sparse_rich_ttest.pvalue,
            },
        }
        
        # Add pairwise tests if we have 3+ profile types
        if len(profile_order) >= 3:
            medium_key = profile_order[len(profile_order) // 2]
            medium_correct = profile_data[medium_key]['correct']
            medium_ll = profile_data[medium_key]['log_loss']
            
            result['mcnemar_sparse_vs_medium'] = mcnemar_test(sparse_correct, medium_correct)
            result['mcnemar_medium_vs_rich'] = mcnemar_test(medium_correct, rich_correct)
            
            result['ttest_sparse_vs_medium'] = {
                'statistic': stats.ttest_rel(sparse_ll, medium_ll).statistic,
                'p_value': stats.ttest_rel(sparse_ll, medium_ll).pvalue,
            }
            result['ttest_medium_vs_rich'] = {
                'statistic': stats.ttest_rel(medium_ll, rich_ll).statistic,
                'p_value': stats.ttest_rel(medium_ll, rich_ll).pvalue,
            }
        
        return result
    
    # -------------------------------------------------------------------------
    # Baselines and Heterogeneity Analysis
    # -------------------------------------------------------------------------
    
    def compute_baselines(self) -> Dict[str, Any]:
        """
        Compute baseline accuracies for comparison.
        
        Returns
        -------
        dict
            Contains:
            - random_baseline: Expected accuracy from random guessing (1/n_options per question)
            - majority_baseline: Accuracy from always predicting most common answer per question
            - stratified_random: Expected accuracy weighted by answer distribution
        """
        # Group by target question
        by_target = defaultdict(list)
        for inst in self.instances:
            key = f"{inst.survey}_{inst.target_code}"
            by_target[key].append(inst)
        
        # Random baseline: average of 1/n_options across all questions
        random_accs = []
        majority_accs = []
        majority_correct = 0
        total = 0
        
        for target_key, insts in by_target.items():
            n_opts = len(insts[0].options)
            n_insts = len(insts)
            
            # Random baseline for this question
            random_accs.extend([1.0 / n_opts] * n_insts)
            
            # Majority baseline: find most common ground truth
            from collections import Counter
            gt_counts = Counter(inst.ground_truth for inst in insts)
            majority_answer, majority_count = gt_counts.most_common(1)[0]
            
            # How many would be correct if we always predicted majority?
            majority_correct += majority_count
            total += n_insts
            
            # Per-question majority accuracy (for averaging)
            majority_accs.append(majority_count / n_insts)
        
        return {
            'random_baseline': np.mean(random_accs),
            'random_baseline_weighted': np.mean(random_accs),  # same, just explicit
            'majority_baseline': majority_correct / total if total > 0 else 0,
            'majority_baseline_by_question': np.mean(majority_accs),
            'n_questions': len(by_target),
            'model_accuracy': sum(1 for i in self.instances if i.correct) / len(self.instances),
            'model_vs_random': (sum(1 for i in self.instances if i.correct) / len(self.instances)) - np.mean(random_accs),
            'model_vs_majority': (sum(1 for i in self.instances if i.correct) / len(self.instances)) - (majority_correct / total if total > 0 else 0),
        }
    
    def heterogeneity_analysis(self, min_n: int = 30) -> Dict[str, Any]:
        """
        Analyze whether predictions preserve within-group heterogeneity.
        
        Computes variance ratio (predicted/empirical) across questions.
        Values < 1 indicate diversity flattening.
        
        Parameters
        ----------
        min_n : int
            Minimum instances per question for inclusion
            
        Returns
        -------
        dict
            Variance ratios and JS divergences aggregated across questions
        """
        # Group by target question
        by_target = defaultdict(list)
        for inst in self.instances:
            key = f"{inst.survey}_{inst.target_code}"
            by_target[key].append(inst)
        
        variance_ratios_soft = []
        variance_ratios_hard = []
        js_divergences_soft = []
        js_divergences_hard = []
        
        for target_key, insts in by_target.items():
            if len(insts) < min_n:
                continue
            
            options = insts[0].options
            metrics = compute_distribution_metrics(insts, options)
            
            if metrics.get('variance_ratio_soft') is not None:
                variance_ratios_soft.append(metrics['variance_ratio_soft'])
            if metrics.get('variance_ratio_hard') is not None:
                variance_ratios_hard.append(metrics['variance_ratio_hard'])
            if metrics.get('js_divergence_soft') is not None:
                js_divergences_soft.append(metrics['js_divergence_soft'])
            if metrics.get('js_divergence_hard') is not None:
                js_divergences_hard.append(metrics['js_divergence_hard'])
        
        # Compute summary statistics
        def summarize(arr):
            if not arr:
                return {'mean': None, 'median': None, 'std': None, 'n': 0}
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'n': len(arr),
            }
        
        # Count how many show flattening (ratio < 1)
        n_flattening_soft = sum(1 for v in variance_ratios_soft if v < 1.0)
        n_flattening_hard = sum(1 for v in variance_ratios_hard if v < 1.0)
        
        return {
            'variance_ratio_soft': summarize(variance_ratios_soft),
            'variance_ratio_hard': summarize(variance_ratios_hard),
            'js_divergence_soft': summarize(js_divergences_soft),
            'js_divergence_hard': summarize(js_divergences_hard),
            'flattening_rate_soft': n_flattening_soft / len(variance_ratios_soft) if variance_ratios_soft else None,
            'flattening_rate_hard': n_flattening_hard / len(variance_ratios_hard) if variance_ratios_hard else None,
            'interpretation': {
                'variance_ratio < 1': 'Predictions less diverse than reality (flattening)',
                'variance_ratio = 1': 'Predictions match empirical diversity',
                'variance_ratio > 1': 'Predictions more diverse than reality (unlikely)',
            }
        }
    
    # -------------------------------------------------------------------------
    # Output Methods
    # -------------------------------------------------------------------------
    
    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        overall = self.overall_metrics()
        print(f"\nTotal instances: {overall.n:,}")
        print(f"Surveys: {', '.join(self.surveys)}")
        print(f"Profile types: {', '.join(self.profile_types)}")
        print(f"Unique targets: {len(self.targets)}")
        
        print(f"\n--- Overall Metrics ---")
        print(f"Accuracy:         {overall.accuracy:.2%}")
        print(f"Top-2 Accuracy:   {overall.top2_accuracy:.2%}")
        print(f"Top-3 Accuracy:   {overall.top3_accuracy:.2%}")
        print(f"Mean Log Loss:    {overall.mean_log_loss:.3f}")
        print(f"Mean P(correct):  {overall.mean_prob_correct:.3f}")
        
        print(f"\n--- By Profile Type ---")
        print(f"{'Profile':<12} {'N':>8} {'Acc':>8} {'Top2':>8} {'LogLoss':>10}")
        print("-" * 50)
        for profile, m in self.metrics_by_profile_type().items():
            name = {'s3m2': 'sparse', 's4m3': 'medium', 's6m4': 'rich'}.get(profile, profile)
            print(f"{name:<12} {m.n:>8,} {m.accuracy:>7.1%} {m.top2_accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
        
        print(f"\n--- By Survey ---")
        print(f"{'Survey':<20} {'N':>8} {'Acc':>8} {'Top2':>8} {'LogLoss':>10}")
        print("-" * 58)
        for survey, m in self.metrics_by_survey().items():
            print(f"{survey:<20} {m.n:>8,} {m.accuracy:>7.1%} {m.top2_accuracy:>7.1%} {m.mean_log_loss:>10.3f}")
    
    def to_dataframe(self):
        """Convert instances to pandas DataFrame for further analysis."""
        import pandas as pd
        
        rows = []
        for inst in self.instances:
            rows.append({
                'example_id': inst.example_id,
                'survey': inst.survey,
                'respondent_id': inst.respondent_id,
                'target_code': inst.target_code,
                'profile_type': inst.profile_type,
                'profile_name': inst.profile_name,
                'n_features': inst.n_features,
                'ground_truth': inst.ground_truth,
                'predicted': inst.predicted,
                'correct': inst.correct,
                'log_loss': inst.log_loss,
                'prob_correct': inst.prob_correct,
                'rank_of_correct': inst.rank_of_correct(),
            })
        
        return pd.DataFrame(rows)
    
    def save_summary(self, filepath: str) -> None:
        """Save summary statistics to JSON file."""
        summary = {
            'total_instances': self.n_instances,
            'surveys': self.surveys,
            'profile_types': self.profile_types,
            'n_targets': len(self.targets),
            'overall': self.overall_metrics().to_dict(),
            'by_profile_type': {
                k: v.to_dict() for k, v in self.metrics_by_profile_type().items()
            },
            'by_survey': {
                k: v.to_dict() for k, v in self.metrics_by_survey().items()
            },
            'profile_richness_effect': self.test_profile_richness_effect(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def error_analysis(self, max_examples: int = 10) -> Dict[str, Any]:
        """Analyze error patterns in predictions."""
        return analyze_errors(self.instances, max_examples)
    
    def error_analysis_by_survey(self) -> Dict[str, Dict[str, Any]]:
        """Error analysis broken down by survey."""
        return {
            survey: analyze_errors(insts)
            for survey, insts in self._by_survey.items()
        }
    
    def calibration_curve(
        self, 
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute calibration curve data.
        
        Bins instances by predicted probability of correct answer
        and computes actual accuracy in each bin.
        """
        probs = [inst.prob_correct for inst in self.instances]
        correct = [int(inst.correct) for inst in self.instances]
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            n_in_bin = mask.sum()
            
            if n_in_bin > 0:
                mean_pred_prob = np.mean([p for p, m in zip(probs, mask) if m])
                actual_accuracy = np.mean([c for c, m in zip(correct, mask) if m])
            else:
                mean_pred_prob = (bins[i] + bins[i+1]) / 2
                actual_accuracy = None
            
            calibration_data.append({
                'bin_start': bins[i],
                'bin_end': bins[i+1],
                'n': int(n_in_bin),
                'mean_predicted_prob': mean_pred_prob,
                'actual_accuracy': actual_accuracy,
            })
        
        # Expected Calibration Error (ECE)
        ece = 0
        total = len(self.instances)
        for bin_data in calibration_data:
            if bin_data['actual_accuracy'] is not None:
                weight = bin_data['n'] / total
                gap = abs(bin_data['mean_predicted_prob'] - bin_data['actual_accuracy'])
                ece += weight * gap
        
        return {
            'bins': calibration_data,
            'ece': ece,
            'n_bins': n_bins,
        }
    
    # -------------------------------------------------------------------------
    # Disaggregated Analysis Methods
    # -------------------------------------------------------------------------
    
    def metrics_by_n_options(self) -> Dict[str, MetricsSummary]:
        """
        Compute metrics grouped by number of answer options.
        
        Useful for comparing binary vs. Likert scale questions.
        """
        by_n_opts = defaultdict(list)
        for inst in self.instances:
            n = len(inst.options)
            # Group into meaningful categories
            if n == 2:
                key = '2 (binary)'
            elif n <= 4:
                key = '3-4 options'
            elif n <= 6:
                key = '5-6 options'
            elif n <= 10:
                key = '7-10 options'
            else:
                key = '11+ options'
            by_n_opts[key].append(inst)
        
        return {
            k: compute_instance_metrics(insts)
            for k, insts in sorted(by_n_opts.items())
        }
    
    def metrics_by_n_options_detailed(self) -> Dict[int, MetricsSummary]:
        """
        Compute metrics for each exact number of options.
        """
        by_n_opts = defaultdict(list)
        for inst in self.instances:
            by_n_opts[len(inst.options)].append(inst)
        
        return {
            k: compute_instance_metrics(insts)
            for k, insts in sorted(by_n_opts.items())
        }
    
    # -------------------------------------------------------------------------
    # Metadata-based Analysis Methods
    # -------------------------------------------------------------------------
    
    def has_metadata(self) -> bool:
        """Check if instances have been enriched with metadata."""
        if not self.instances:
            return False
        # Check first few instances for metadata
        sample = self.instances[:min(10, len(self.instances))]
        has_any = any(
            inst.country is not None or 
            inst.target_section is not None or
            inst.target_topic_tag is not None
            for inst in sample
        )
        return has_any
    
    def metadata_coverage(self) -> Dict[str, float]:
        """Report what percentage of instances have each metadata field."""
        if not self.instances:
            return {}
        
        n = len(self.instances)
        return {
            'country': sum(1 for i in self.instances if i.country is not None) / n,
            'target_section': sum(1 for i in self.instances if i.target_section is not None) / n,
            'target_topic_tag': sum(1 for i in self.instances if i.target_topic_tag is not None) / n,
            'target_response_format': sum(1 for i in self.instances if i.target_response_format is not None) / n,
        }
    
    def metrics_by_country(self, min_n: int = 100) -> Dict[str, MetricsSummary]:
        """
        Compute metrics grouped by country.
        
        Uses metadata if available, otherwise falls back to parsing respondent_id.
        
        Parameters
        ----------
        min_n : int
            Minimum instances required for a country to be included
        """
        by_country = defaultdict(list)
        
        for inst in self.instances:
            country = None
            
            # First try metadata
            if inst.country is not None:
                country = normalize_country(inst.country, inst.survey)
            
            # Fall back to parsing respondent_id for ESS-style IDs
            if country is None:
                resp_id = str(inst.respondent_id)
                if len(resp_id) >= 2:
                    prefix = resp_id[:2].upper()
                    if prefix.isalpha() and len(resp_id) > 2:
                        if resp_id[2] == '_' or resp_id[2].isdigit():
                            country = prefix
            
            # Final fallback
            if country is None:
                country = f"{inst.survey}_unknown"
            
            by_country[country].append(inst)
        
        return {
            k: compute_instance_metrics(insts)
            for k, insts in sorted(by_country.items())
            if len(insts) >= min_n
        }
    
    def metrics_by_section(self, min_n: int = 100) -> Dict[str, MetricsSummary]:
        """
        Compute metrics grouped by target section (e.g., political_attitudes).
        
        Parameters
        ----------
        min_n : int
            Minimum instances required for a section to be included
        """
        by_section = defaultdict(list)
        
        for inst in self.instances:
            section = inst.target_section or 'unknown'
            by_section[section].append(inst)
        
        return {
            k: compute_instance_metrics(insts)
            for k, insts in sorted(by_section.items())
            if len(insts) >= min_n
        }
    
    def metrics_by_topic_tag(self, min_n: int = 100) -> Dict[str, MetricsSummary]:
        """
        Compute metrics grouped by topic tag (e.g., democratic_values).
        
        Parameters
        ----------
        min_n : int
            Minimum instances required for a tag to be included
        """
        by_tag = defaultdict(list)
        
        for inst in self.instances:
            tag = inst.target_topic_tag or 'unknown'
            by_tag[tag].append(inst)
        
        return {
            k: compute_instance_metrics(insts)
            for k, insts in sorted(by_tag.items())
            if len(insts) >= min_n
        }
    
    def metrics_by_response_format(self, min_n: int = 100) -> Dict[str, MetricsSummary]:
        """
        Compute metrics grouped by response format (e.g., likert_5, categorical).
        
        Parameters
        ----------
        min_n : int
            Minimum instances required for a format to be included
        """
        by_format = defaultdict(list)
        
        for inst in self.instances:
            fmt = inst.target_response_format or 'unknown'
            by_format[fmt].append(inst)
        
        return {
            k: compute_instance_metrics(insts)
            for k, insts in sorted(by_format.items())
            if len(insts) >= min_n
        }
    
    def metrics_by_survey_and_section(self, min_n: int = 50) -> Dict[str, Dict[str, MetricsSummary]]:
        """
        Compute metrics grouped by survey × section.
        """
        result = {}
        
        for survey in self.surveys:
            survey_insts = self._by_survey[survey]
            by_section = defaultdict(list)
            
            for inst in survey_insts:
                section = inst.target_section or 'unknown'
                by_section[section].append(inst)
            
            result[survey] = {
                k: compute_instance_metrics(insts)
                for k, insts in sorted(by_section.items())
                if len(insts) >= min_n
            }
        
        return result
    
    def metrics_by_survey_and_country(self, min_n: int = 100) -> Dict[str, Dict[str, MetricsSummary]]:
        """
        Compute metrics grouped by survey × country.
        """
        result = {}
        
        for survey in self.surveys:
            survey_insts = self._by_survey[survey]
            by_country = defaultdict(list)
            
            for inst in survey_insts:
                country = None
                
                # First try metadata
                if inst.country is not None:
                    country = normalize_country(inst.country, inst.survey)
                
                # Fall back to parsing respondent_id
                if country is None:
                    resp_id = str(inst.respondent_id)
                    parts = resp_id.split('_')
                    if len(parts) >= 1 and len(parts[0]) >= 2:
                        prefix = parts[0][:3] if len(parts[0]) >= 3 and parts[0][:3].isalpha() else parts[0][:2]
                        if prefix.isalpha():
                            country = prefix.upper()
                
                if country is None:
                    country = 'unknown'
                
                by_country[country].append(inst)
            
            result[survey] = {
                k: compute_instance_metrics(insts)
                for k, insts in sorted(by_country.items())
                if len(insts) >= min_n
            }
        
        return result
    
    def option_position_bias(self) -> Dict[str, Any]:
        """
        NOTE: This analysis is only meaningful if options are shown to the model in order.
        If using perplexity-based evaluation where each option is scored independently,
        this measures distribution of correct answers, not model bias.
        
        Returns statistics on where correct answers tend to fall in option lists.
        """
        correct_positions = []
        correct_is_first = []
        correct_is_last = []
        
        for inst in self.instances:
            n_opts = len(inst.options)
            if n_opts == 0:
                continue
            
            # Position of correct answer
            try:
                correct_pos = inst.options.index(inst.ground_truth)
                correct_positions.append(correct_pos / (n_opts - 1) if n_opts > 1 else 0.5)
                correct_is_first.append(correct_pos == 0)
                correct_is_last.append(correct_pos == n_opts - 1)
            except ValueError:
                pass
        
        # Analyze accuracy by position of correct answer
        accuracy_by_position = defaultdict(lambda: {'correct': 0, 'total': 0})
        for inst in self.instances:
            n_opts = len(inst.options)
            if n_opts == 0:
                continue
            try:
                correct_pos = inst.options.index(inst.ground_truth)
                # Normalize to quintiles
                if n_opts > 1:
                    rel_pos = correct_pos / (n_opts - 1)
                    if rel_pos < 0.2:
                        bucket = 'first_20%'
                    elif rel_pos < 0.4:
                        bucket = '20-40%'
                    elif rel_pos < 0.6:
                        bucket = '40-60%'
                    elif rel_pos < 0.8:
                        bucket = '60-80%'
                    else:
                        bucket = 'last_20%'
                else:
                    bucket = 'single_option'
                
                accuracy_by_position[bucket]['total'] += 1
                if inst.correct:
                    accuracy_by_position[bucket]['correct'] += 1
            except ValueError:
                pass
        
        position_accuracy = {
            k: v['correct'] / v['total'] if v['total'] > 0 else 0
            for k, v in accuracy_by_position.items()
        }
        position_counts = {k: v['total'] for k, v in accuracy_by_position.items()}
        
        return {
            'note': 'This shows accuracy by where correct answer falls in option list, NOT model position bias (model does not see option order)',
            'mean_correct_position': np.mean(correct_positions) if correct_positions else None,
            'correct_first_rate': np.mean(correct_is_first) if correct_is_first else None,
            'correct_last_rate': np.mean(correct_is_last) if correct_is_last else None,
            'accuracy_by_correct_position': position_accuracy,
            'counts_by_correct_position': position_counts,
        }
    
    def response_type_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance by response type (agree/disagree, yes/no, etc.).
        
        Detects common response patterns and groups questions accordingly.
        """
        # Define response type patterns
        patterns = {
            'binary_yes_no': {'yes', 'no'},
            'binary_agree_disagree': {'agree', 'disagree'},
            'likert_5': {'strongly agree', 'agree', 'neither agree nor disagree', 
                        'disagree', 'strongly disagree'},
            'likert_agreement': {'agree', 'disagree', 'strongly'},  # partial match
            'frequency': {'never', 'rarely', 'sometimes', 'often', 'always'},
            'satisfaction': {'very satisfied', 'satisfied', 'dissatisfied', 'very dissatisfied'},
            'trust': {'trust', 'distrust', 'completely'},
        }
        
        by_type = defaultdict(list)
        
        for inst in self.instances:
            opts_lower = {o.lower() for o in inst.options}
            
            classified = False
            
            # Check for yes/no
            if opts_lower == {'yes', 'no'} or opts_lower <= {'yes', 'no', "don't know", 'refused'}:
                by_type['binary_yes_no'].append(inst)
                classified = True
            
            # Check for agree/disagree variants
            elif any('agree' in o for o in opts_lower) and any('disagree' in o for o in opts_lower):
                if len(inst.options) <= 3:
                    by_type['binary_agree_disagree'].append(inst)
                elif len(inst.options) <= 5:
                    by_type['likert_4_5'].append(inst)
                else:
                    by_type['likert_6+'].append(inst)
                classified = True
            
            # Check for frequency scales
            elif any(w in ' '.join(opts_lower) for w in ['never', 'always', 'often', 'rarely']):
                by_type['frequency_scale'].append(inst)
                classified = True
            
            # Check for satisfaction
            elif any('satisf' in o for o in opts_lower):
                by_type['satisfaction_scale'].append(inst)
                classified = True
            
            # Check for trust
            elif any('trust' in o for o in opts_lower):
                by_type['trust_scale'].append(inst)
                classified = True
            
            # Numeric/quantity
            elif all(any(c.isdigit() for c in o) for o in inst.options if o.lower() not in {"don't know", 'refused', 'no answer'}):
                by_type['numeric'].append(inst)
                classified = True
            
            if not classified:
                by_type['other'].append(inst)
        
        return {
            response_type: {
                'n': len(insts),
                'metrics': compute_instance_metrics(insts).to_dict(),
                'example_options': insts[0].options if insts else [],
            }
            for response_type, insts in sorted(by_type.items())
        }
    
    def yes_no_bias_analysis(self) -> Dict[str, Any]:
        """
        Detailed analysis of yes/no question bias with statistical tests.
        
        This addresses the top error pattern: 'No' → 'Yes'
        """
        from scipy import stats
        
        yes_no_instances = []
        
        for inst in self.instances:
            opts_lower = {o.lower() for o in inst.options}
            if 'yes' in opts_lower and 'no' in opts_lower and len(opts_lower) <= 4:
                yes_no_instances.append(inst)
        
        if not yes_no_instances:
            return {'status': 'no_yes_no_questions_found'}
        
        # Count predictions and ground truths
        pred_yes = sum(1 for i in yes_no_instances if i.predicted.lower() == 'yes')
        pred_no = sum(1 for i in yes_no_instances if i.predicted.lower() == 'no')
        gt_yes = sum(1 for i in yes_no_instances if i.ground_truth.lower() == 'yes')
        gt_no = sum(1 for i in yes_no_instances if i.ground_truth.lower() == 'no')
        
        # Confusion matrix
        yes_to_yes = sum(1 for i in yes_no_instances 
                        if i.ground_truth.lower() == 'yes' and i.predicted.lower() == 'yes')
        yes_to_no = sum(1 for i in yes_no_instances 
                       if i.ground_truth.lower() == 'yes' and i.predicted.lower() == 'no')
        no_to_yes = sum(1 for i in yes_no_instances 
                       if i.ground_truth.lower() == 'no' and i.predicted.lower() == 'yes')
        no_to_no = sum(1 for i in yes_no_instances 
                      if i.ground_truth.lower() == 'no' and i.predicted.lower() == 'no')
        
        total = len(yes_no_instances)
        
        # Statistical tests for bias
        # Chi-square test: are predictions independent of ground truth?
        observed = np.array([[yes_to_yes, yes_to_no], [no_to_yes, no_to_no]])
        chi2_result = stats.chi2_contingency(observed)
        
        # Binomial test: is prediction rate different from ground truth rate?
        # H0: P(predict yes) = P(gt yes)
        gt_yes_rate = gt_yes / total if total > 0 else 0.5
        binom_result = stats.binomtest(pred_yes, total, gt_yes_rate, alternative='two-sided')
        
        # McNemar test for systematic bias
        # Tests if the model systematically flips yes→no differently than no→yes
        if yes_to_no + no_to_yes > 0:
            mcnemar_chi2 = (abs(yes_to_no - no_to_yes) - 1)**2 / (yes_to_no + no_to_yes)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, df=1)
        else:
            mcnemar_chi2 = None
            mcnemar_p = None
        
        return {
            'n_instances': total,
            'ground_truth': {
                'yes': gt_yes,
                'no': gt_no,
                'yes_rate': gt_yes / total if total > 0 else 0,
            },
            'predictions': {
                'yes': pred_yes,
                'no': pred_no,
                'yes_rate': pred_yes / total if total > 0 else 0,
            },
            'confusion_matrix': {
                'yes_to_yes': yes_to_yes,
                'yes_to_no': yes_to_no,
                'no_to_yes': no_to_yes,
                'no_to_no': no_to_no,
            },
            'accuracy': {
                'overall': (yes_to_yes + no_to_no) / total if total > 0 else 0,
                'when_gt_yes': yes_to_yes / gt_yes if gt_yes > 0 else 0,
                'when_gt_no': no_to_no / gt_no if gt_no > 0 else 0,
            },
            'bias': {
                'yes_bias': (pred_yes / total) - (gt_yes / total) if total > 0 else 0,
                'interpretation': 'Model predicts "Yes" more often than ground truth' 
                                 if pred_yes > gt_yes else 'Model predicts "No" more often than ground truth'
            },
            'statistical_tests': {
                'chi2_independence': {
                    'statistic': chi2_result.statistic,
                    'p_value': chi2_result.pvalue,
                    'interpretation': 'Tests if predictions are independent of ground truth'
                },
                'binomial_bias': {
                    'statistic': binom_result.statistic,
                    'p_value': binom_result.pvalue,
                    'interpretation': 'Tests if prediction yes-rate differs from ground truth yes-rate'
                },
                'mcnemar_asymmetry': {
                    'statistic': mcnemar_chi2,
                    'p_value': mcnemar_p,
                    'interpretation': 'Tests if yes→no errors differ systematically from no→yes errors'
                }
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_evaluate(filepath: str) -> None:
    """Quick evaluation: load results and print summary."""
    analyzer = ResultsAnalyzer.from_jsonl(filepath)
    analyzer.print_summary()
    
    print("\n--- Profile Richness Effect ---")
    effect = analyzer.test_profile_richness_effect()
    if effect['status'] == 'complete':
        print(f"Complete triplets: {effect['n_triplets']}")
        print(f"Accuracy by profile: {effect['accuracy']}")
        sparse_vs_rich = effect['ttest_sparse_vs_rich']
        print(f"Sparse vs Rich log-loss t-test: t={sparse_vs_rich['statistic']:.3f}, p={sparse_vs_rich['p_value']:.4f}")
    else:
        print(f"Insufficient data: {effect}")


def compare_models(
    results_paths: Dict[str, str],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare results across multiple models.
    
    Parameters
    ----------
    results_paths : dict
        Mapping of model_name -> results_jsonl_path
    output_path : str, optional
        Path to save comparison JSON
        
    Returns
    -------
    dict
        Comparison results
    """
    comparison = {}
    
    for model_name, path in results_paths.items():
        analyzer = ResultsAnalyzer.from_jsonl(path)
        comparison[model_name] = {
            'overall': analyzer.overall_metrics().to_dict(),
            'by_profile': {
                k: v.to_dict() for k, v in analyzer.metrics_by_profile_type().items()
            },
            'by_survey': {
                k: v.to_dict() for k, v in analyzer.metrics_by_survey().items()
            },
        }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    return comparison


def analyze_errors(
    instances: List[ParsedInstance],
    max_examples: int = 10
) -> Dict[str, Any]:
    """
    Analyze error patterns in predictions.
    
    Returns
    -------
    dict
        Error analysis including confusion patterns
    """
    errors = [i for i in instances if not i.correct]
    correct = [i for i in instances if i.correct]
    
    # Confusion: ground_truth -> predicted -> count
    confusion = defaultdict(lambda: defaultdict(int))
    for inst in errors:
        confusion[inst.ground_truth][inst.predicted] += 1
    
    # Convert to regular dict
    confusion_dict = {
        gt: dict(preds) for gt, preds in confusion.items()
    }
    
    # Most common error patterns
    error_patterns = []
    for gt, preds in confusion.items():
        for pred, count in preds.items():
            error_patterns.append((gt, pred, count))
    error_patterns.sort(key=lambda x: -x[2])
    
    # Sample error instances
    error_examples = []
    for inst in errors[:max_examples]:
        error_examples.append({
            'example_id': inst.example_id,
            'ground_truth': inst.ground_truth,
            'predicted': inst.predicted,
            'prob_correct': inst.prob_correct,
            'prob_predicted': inst.probs.get(inst.predicted, 0),
            'rank_of_correct': inst.rank_of_correct(),
        })
    
    return {
        'total_errors': len(errors),
        'total_correct': len(correct),
        'error_rate': len(errors) / len(instances) if instances else 0,
        'confusion_matrix': confusion_dict,
        'top_error_patterns': [
            {'ground_truth': gt, 'predicted': pred, 'count': count}
            for gt, pred, count in error_patterns[:20]
        ],
        'error_examples': error_examples,
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        quick_evaluate(sys.argv[1])
    else:
        print("Usage: python evaluation.py <results.jsonl>")