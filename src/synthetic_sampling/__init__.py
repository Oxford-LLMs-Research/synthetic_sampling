"""
Synthetic Sampling -- respondent profile generation for LLM evaluation.

Quick Start::

    from synthetic_sampling import ProfileBuilder

    # Explore bundled survey metadata (no data required)
    builder = ProfileBuilder('wvs')
    builder.list_sections()
    builder.list_variables('demographics')

    # Load survey data and generate profiles
    builder.load_data('~/data/WVS/wvs.csv')
    profile = builder.generate_profile(respondent_id=12345, seed=42)

See ``ProfileBuilder`` for the full API.
"""

__version__ = '0.2.0'
__author__ = 'Oxford LLMs Research'

# -- High-level public API (recommended for external users) ----------------

from .profile_builder import ProfileBuilder

from .profiles.utils import get_bundled_metadata_dir, load_survey_metadata
from .profiles.formats import list_profile_formats, PROFILE_FORMATS

# -- Lower-level API (config, loaders, builder) ----------------------------

from .config import (
    DataPaths,
    DatasetConfig,
    GeneratorConfig,
    SURVEY_REGISTRY,
    get_survey_config,
    list_surveys,
    load_config,
)

from .loaders import (
    SurveyLoader,
    scan_survey_directory,
)

from .builder import DatasetBuilder

__all__ = [
    # High-level API
    'ProfileBuilder',
    'get_bundled_metadata_dir',
    'load_survey_metadata',
    'list_profile_formats',
    'PROFILE_FORMATS',
    # Version
    '__version__',
    # Config
    'DataPaths',
    'DatasetConfig',
    'GeneratorConfig',
    'SURVEY_REGISTRY',
    'get_survey_config',
    'list_surveys',
    'load_config',
    # Loaders
    'SurveyLoader',
    'scan_survey_directory',
    # Builder
    'DatasetBuilder',
]