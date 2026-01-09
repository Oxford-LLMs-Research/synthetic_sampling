"""
Data loading module for synthetic sampling pipeline.

Example usage:
    from synthetic_sampling.config import DataPaths
    from synthetic_sampling.data import SurveyLoader
    
    paths = DataPaths(
        raw_data_dir='~/data/surveys',
        metadata_dir='./src/synthetic_sampling/profiles/metadata',
        output_dir='./outputs'
    )
    loader = SurveyLoader(paths)
    
    # Load a single survey
    df, metadata = loader.load_survey('wvs')
    
    # Load all available surveys
    all_data = loader.load_all()
"""

from .file_io import (
    load_csv,
    load_stata,
    load_spss,
    load_file,
    find_data_files,
    load_multiple_files,
)

from .survey_loader import (
    SurveyLoader,
    scan_survey_directory,
)

__all__ = [
    # Main class
    'SurveyLoader',
    'scan_survey_directory',
    # File loaders (rarely needed directly)
    'load_csv',
    'load_stata', 
    'load_spss',
    'load_file',
    'find_data_files',
    'load_multiple_files',
]