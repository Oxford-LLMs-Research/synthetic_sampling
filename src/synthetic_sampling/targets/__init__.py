"""
Target sampling module for LLM survey prediction experiments.

Provides adaptive stratified sampling of target questions across survey sections,
with special handling for ESS country-specific variables via concept grouping.
"""

from .targets import (
    # Core sampling
    sample_targets_stratified,
    compute_section_stats,
    
    # Data structures
    SampledTarget,
    SectionStats,
    
    # Utilities
    targets_to_codes,
    is_concept_code,
    get_concept_id_from_code,
    
    # ESS-specific
    ESS_CONCEPT_CONFIGS,
    CONCEPT_MARKER,
    detect_country_specific_variables,
)

__all__ = [
    # Core sampling
    'sample_targets_stratified',
    'compute_section_stats',
    
    # Data structures
    'SampledTarget',
    'SectionStats',
    
    # Utilities
    'targets_to_codes',
    'is_concept_code',
    'get_concept_id_from_code',
    
    # ESS-specific
    'ESS_CONCEPT_CONFIGS',
    'CONCEPT_MARKER',
    'detect_country_specific_variables',
]