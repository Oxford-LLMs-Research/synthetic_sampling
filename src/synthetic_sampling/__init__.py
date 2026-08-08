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

# The survey surface is exposed LAZILY (PEP 562): `surveys` imports pandas,
# which the cluster scoring venv does not carry, and `ss-score` /
# `ss-analyze` must run without it (the 8 Aug A1 jobs failed on exactly
# this). Anything here resolves on first attribute access, not at import.
_SURVEYS_EXPORTS = {
    "SURVEY_REGISTRY", "DataPaths", "DatasetConfig", "GeneratorConfig",
    "SurveyLoader", "get_survey_config", "list_surveys", "load_config",
}

__all__ = ["__version__", *sorted(_SURVEYS_EXPORTS)]


def __getattr__(name):
    if name in _SURVEYS_EXPORTS:
        from . import surveys
        return getattr(surveys, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
