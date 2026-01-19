"""
Evaluation module for LLM survey prediction experiments.

Provides distribution comparison metrics and aggregation tools for comparing
predicted response distributions against empirical (ground truth) distributions.

Supports both conditioned analysis (subgroup features in profile) and 
unconditioned analysis (aggregate by respondent demographics).
"""

from .evaluation import (
    # Metrics
    jensen_shannon_divergence,
    total_variation_distance,
    variance_ratio,
    calibration_error,
    compute_all_metrics,
    
    # Distribution building
    aggregate_to_distribution,
    aggregate_predictions_perplexity,
    
    # Subgroup definitions
    SubgroupDefinition,
    filter_instances_conditioned,
    filter_instances_unconditioned,
    
    # Results
    ComparisonResult,
    AggregatedResults,
    
    # Main comparator
    DistributionComparator,
    
    # Feature mapping
    FeatureMapper,
    
    # I/O utilities
    load_instances_jsonl,
    save_results_json,
    print_summary,
)

__all__ = [
    # Metrics
    'jensen_shannon_divergence',
    'total_variation_distance',
    'variance_ratio',
    'calibration_error',
    'compute_all_metrics',
    
    # Distribution building
    'aggregate_to_distribution',
    'aggregate_predictions_perplexity',
    
    # Subgroup definitions
    'SubgroupDefinition',
    'filter_instances_conditioned',
    'filter_instances_unconditioned',
    
    # Results
    'ComparisonResult',
    'AggregatedResults',
    
    # Main comparator
    'DistributionComparator',
    
    # Feature mapping
    'FeatureMapper',
    
    # I/O utilities
    'load_instances_jsonl',
    'save_results_json',
    'print_summary',
]