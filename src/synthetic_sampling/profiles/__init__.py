"""
Respondent Profile Generation Module

This package provides functionality for generating respondent profiles and
prediction instances for LLM survey prediction experiments.
"""

from .generator import RespondentProfileGenerator
from .dataclasses import (
    ProfileConfig,
    RespondentProfile,
    TargetQuestion,
    PredictionInstance
)
from .formats import (
    get_profile_formatter,
    list_profile_formats,
    PROFILE_FORMATS
)
from .utils import (
    get_bundled_metadata_dir,
    load_survey_metadata,
    load_metadata,
    verify_profile_nesting,
)

__all__ = [
    # Main generator class
    'RespondentProfileGenerator',
    # Data classes
    'ProfileConfig',
    'RespondentProfile',
    'TargetQuestion',
    'PredictionInstance',
    # Formatting functions
    'get_profile_formatter',
    'list_profile_formats',
    'PROFILE_FORMATS',
    # Utility functions
    'get_bundled_metadata_dir',
    'load_survey_metadata',
    'load_metadata',
    'verify_profile_nesting',
]

