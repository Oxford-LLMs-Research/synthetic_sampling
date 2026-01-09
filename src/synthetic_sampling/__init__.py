"""
Synthetic Sampling: LLM evaluation for survey response prediction.

This package provides tools for:
- Loading and preprocessing cross-national survey data
- Generating respondent profiles with configurable features
- Creating prediction instances for LLM evaluation
- Managing experimental configurations

Quick Start:
    from synthetic_sampling.config import DataPaths, DatasetConfig, GeneratorConfig
    from synthetic_sampling.builder import DatasetBuilder
    
    paths = DataPaths(
        raw_data_dir='~/data/surveys',
        metadata_dir='./src/synthetic_sampling/profiles/metadata',
        output_dir='./outputs'
    )
    
    builder = DatasetBuilder(paths, DatasetConfig(), GeneratorConfig())
    instances = builder.build_dataset(['wvs'])
    builder.save_jsonl(instances, 'dataset.jsonl')

Modules:
    config: Configuration management (paths, surveys, parameters)
    loaders: Data loading (survey files, metadata)
    profiles: Profile generation (RespondentProfileGenerator)
    builder: Dataset building (DatasetBuilder)
"""

__version__ = '0.2.0'
__author__ = 'Oxford LLMs Research'

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