"""
Synthetic Sampling -- profile generation and LLM scoring for survey prediction.

Minimal-core pipeline (rebuild/minimal-core):

    surveys   load + harmonise six survey sources
    profiles  instance generation (incl. ESS concept resolve)
    scoring   /completions arms (label_num primary; echo retained)
    analysis  normalized accuracy, AUC, prior-correct, bootstrap
    checks    smoke, coverage, number verification

Import from subpackages for heavy use::

    from synthetic_sampling.surveys import DataPaths, SurveyLoader, list_surveys
    from synthetic_sampling.profiles import DatasetBuilder
    from synthetic_sampling.scoring import run_scoring, DEFAULT_ARMS
"""

__version__ = "0.3.0"
__author__ = "Oxford LLMs Research"

# Light public surface (survey registry + paths). Heavy modules stay in
# subpackages so `ss-score` does not import the profile generator.
from .surveys import (
    SURVEY_REGISTRY,
    DataPaths,
    DatasetConfig,
    GeneratorConfig,
    SurveyLoader,
    get_survey_config,
    list_surveys,
    load_config,
)

__all__ = [
    "__version__",
    "SURVEY_REGISTRY",
    "DataPaths",
    "DatasetConfig",
    "GeneratorConfig",
    "SurveyLoader",
    "get_survey_config",
    "list_surveys",
    "load_config",
]
