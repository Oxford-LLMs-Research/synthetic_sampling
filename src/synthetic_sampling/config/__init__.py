"""
Configuration module for synthetic sampling pipeline.

This module provides configuration management for paths, survey definitions,
and generation parameters.

Example usage:
    from synthetic_sampling.config import (
        DataPaths, 
        DatasetConfig,
        SURVEY_REGISTRY,
        get_survey_config,
        load_config,
    )
    
    # Load all config from a YAML file
    config = load_config('configs/local.yaml')
    paths = config['paths']
    
    # Or load individual components
    paths = DataPaths.from_yaml('configs/local.yaml')
    
    # Or create directly without YAML
    paths = DataPaths(
        raw_data_dir='~/data/surveys',
        metadata_dir='./src/synthetic_sampling/profiles/metadata',
        output_dir='./outputs'
    )
    
    # Access survey configurations
    wvs_config = get_survey_config('wvs')
    print(wvs_config.respondent_id_col)  # 'D_INTERVIEW'
"""

from .base import (
    DataPaths,
    GeneratorConfig,
    DatasetConfig,
    load_config,
)

from .surveys import (
    SurveyConfig,
    SURVEY_REGISTRY,
    get_survey_config,
    list_surveys,
    list_surveys_detailed,
    BAROMETER_SURVEYS,
    ESS_SURVEYS,
    ALL_SURVEYS,
)

__all__ = [
    # Base config classes
    'DataPaths',
    'GeneratorConfig', 
    'DatasetConfig',
    'load_config',
    # Survey config
    'SurveyConfig',
    'SURVEY_REGISTRY',
    'get_survey_config',
    'list_surveys',
    'list_surveys_detailed',
    # Survey groups
    'BAROMETER_SURVEYS',
    'ESS_SURVEYS',
    'ALL_SURVEYS',
]