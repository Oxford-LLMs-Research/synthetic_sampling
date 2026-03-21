"""
Evaluation module for LLM survey prediction results.

Main classes:
- ResultsAnalyzer: Comprehensive analysis of prediction results
- ParsedInstance: Parsed result instance with computed metrics
- MetricsSummary: Summary statistics for a group of instances
- HeterogeneityAnalyzer: Three-layer heterogeneity preservation analysis

Key functions:
- load_results: Load JSONL results file
- enrich_instances_with_metadata: Enrich results with metadata from input files
- quick_evaluate: Quick command-line evaluation
- jensen_shannon_divergence: Distribution comparison metric

Usage:
    from synthetic_sampling.evaluation import ResultsAnalyzer
    
    analyzer = ResultsAnalyzer.from_jsonl("results.jsonl")
    analyzer.print_summary()
    
    # With metadata enrichment
    from synthetic_sampling.evaluation import load_results, enrich_instances_with_metadata
    
    instances = load_results("results.jsonl")
    instances = enrich_instances_with_metadata(instances, ["inputs.jsonl"])
    analyzer = ResultsAnalyzer(instances)
    
    # Heterogeneity analysis
    from synthetic_sampling.evaluation import HeterogeneityAnalyzer
    
    hetero = HeterogeneityAnalyzer(instances)
    report = hetero.full_report()
"""

from .evaluation import (
    # Main classes
    ResultsAnalyzer,
    ParsedInstance,
    MetricsSummary,
    
    # Loading
    load_results,
    load_input_metadata,
    enrich_instances_with_metadata,
    parse_example_id,
    logprobs_to_probs,
    normalize_country,
    
    # Metrics
    jensen_shannon_divergence,
    compute_instance_metrics,
    compute_distribution_metrics,
    
    # Analysis
    analyze_errors,
    compare_models,
    quick_evaluate,
)

from .heterogeneity import (
    HeterogeneityAnalyzer,
    DistributionComparison,
    load_input_data_for_heterogeneity,
    load_survey_data,
)

__all__ = [
    # Main classes
    'ResultsAnalyzer',
    'ParsedInstance', 
    'MetricsSummary',
    'HeterogeneityAnalyzer',
    'DistributionComparison',
    
    # Loading
    'load_results',
    'load_input_metadata',
    'enrich_instances_with_metadata',
    'parse_example_id',
    'logprobs_to_probs',
    'normalize_country',
    'load_input_data_for_heterogeneity',
    'load_survey_data',
    
    # Metrics
    'jensen_shannon_divergence',
    'compute_instance_metrics',
    'compute_distribution_metrics',
    
    # Analysis
    'analyze_errors',
    'compare_models',
    'quick_evaluate',
]