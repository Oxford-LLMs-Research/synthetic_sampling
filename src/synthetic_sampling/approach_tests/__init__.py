"""
Approach Tests: Methodology validation for perplexity-based evaluation.

This module provides test generators to validate the soundness of the
perplexity-based prediction approach used in the main experiments.

Tests:
    1. Surface Form Sensitivity - Tests stability across answer option variations
    2. Options Context - Tests whether showing options in prompt affects predictions
    3. Feature Order Robustness - Tests sensitivity to profile feature ordering

Modules:
    variations: Answer variation rules and generator (synonym, reorder, pronoun)
    surface_form: Test 1 data generator
    options_context: Test 2 data generator
    feature_order: Test 3 data generator

Example:
    from synthetic_sampling.approach_tests import (
        SurfaceFormTestGenerator,
        OptionsContextTestGenerator,
        FeatureOrderTestGenerator,
        AnswerVariationGenerator,
    )
    
    # Generate surface form test data (Test 1)
    sf_gen = SurfaceFormTestGenerator(seed=42)
    sf_instances = sf_gen.generate_from_jsonl('base_instances.jsonl', n_samples=500)
    sf_gen.save_jsonl(sf_instances, 'surface_form_test.jsonl')
    
    # Generate options context test data (Test 2)
    oc_gen = OptionsContextTestGenerator(seed=42)
    oc_instances = oc_gen.generate_from_jsonl('base_instances.jsonl', n_samples=500)
    oc_gen.save_jsonl(oc_instances, 'options_context_test.jsonl')
    
    # Generate feature order test data (Test 3)
    fo_gen = FeatureOrderTestGenerator(seed=42)
    fo_instances = fo_gen.generate_from_jsonl('base_instances.jsonl', n_samples=500)
    fo_gen.save_jsonl(fo_instances, 'feature_order_test.jsonl')
"""

from .variations import (
    # Data structures
    AnswerVariation,
    
    # Generator class
    AnswerVariationGenerator,
    
    # Rule constants
    SPECIAL_VALUES,
    INELIGIBILITY_PATTERNS,
    INELIGIBLE_QUESTION_PATTERNS,
    SYNONYM_WORDS,
    SYNONYM_PHRASES,
    REORDER_PATTERNS,
    PRONOUN_EXACT,
    
    # Utility functions
    get_rule_statistics,
    print_rule_statistics,
)

from .surface_form import (
    # Data structures
    SurfaceFormTestInstance,
    
    # Generator class
    SurfaceFormTestGenerator,
    
    # Utility functions
    get_instance_stats as get_surface_form_stats,
    print_instance_stats as print_surface_form_stats,
)

from .options_context import (
    # Data structures
    OptionsContextTestInstance,
    
    # Generator class
    OptionsContextTestGenerator,
    
    # Option type detection
    detect_option_type,
    is_scale_option,
    
    # Utility functions
    get_options_context_stats,
    print_options_context_stats,
)

from .feature_order import (
    # Data structures
    FeatureOrderTestInstance,
    
    # Generator class
    FeatureOrderTestGenerator,
    
    # Utility functions
    get_feature_order_stats,
    print_feature_order_stats,
    compute_ordering_consistency,
)

__all__ = [
    # === variations module ===
    # Data structures
    "AnswerVariation",
    
    # Generator class
    "AnswerVariationGenerator",
    
    # Rule constants
    "SPECIAL_VALUES",
    "INELIGIBILITY_PATTERNS",
    "INELIGIBLE_QUESTION_PATTERNS",
    "SYNONYM_WORDS",
    "SYNONYM_PHRASES",
    "REORDER_PATTERNS",
    "PRONOUN_EXACT",
    
    # Utility functions
    "get_rule_statistics",
    "print_rule_statistics",
    
    # === surface_form module ===
    # Data structures
    "SurfaceFormTestInstance",
    
    # Generator class
    "SurfaceFormTestGenerator",
    
    # Utility functions
    "get_surface_form_stats",
    "print_surface_form_stats",
    
    # === options_context module ===
    # Data structures
    "OptionsContextTestInstance",
    
    # Generator class
    "OptionsContextTestGenerator",
    
    # Option type detection
    "detect_option_type",
    "is_scale_option",
    
    # Utility functions
    "get_options_context_stats",
    "print_options_context_stats",
    
    # === feature_order module ===
    # Data structures
    "FeatureOrderTestInstance",
    
    # Generator class
    "FeatureOrderTestGenerator",
    
    # Utility functions
    "get_feature_order_stats",
    "print_feature_order_stats",
    "compute_ordering_consistency",
]