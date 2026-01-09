"""
Survey configuration registry.

This module defines the configuration for each survey source, including
column mappings, file patterns, and any survey-specific preprocessing.

The SURVEY_REGISTRY is the single source of truth for survey configurations.
To add a new survey, add an entry here with the appropriate settings.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass(frozen=True)
class SurveyConfig:
    """
    Immutable configuration for a single survey source.
    
    Attributes:
        name: Human-readable survey name (e.g., 'World Values Survey')
        survey_id: Unique identifier used in code and file paths (e.g., 'wvs')
        folder_name: Name of the folder in raw_data_dir containing this survey's files
        respondent_id_col: Column name containing unique respondent identifiers
        country_col: Column name containing country codes/identifiers
        metadata_path: Relative path from metadata_dir to the survey's metadata JSON
        file_patterns: Glob patterns to find data files (tried in order)
        multi_file: If True, combine all matching files (e.g., one file per country)
        id_columns_to_combine: If respondent ID needs to be constructed from multiple columns
        encoding: File encoding for CSV files
    """
    name: str
    survey_id: str
    folder_name: str
    respondent_id_col: str
    country_col: str
    metadata_path: str  # Relative path from metadata_dir
    
    # File loading options
    file_patterns: tuple = ('*.csv',)  # Using tuple for immutability
    multi_file: bool = False
    encoding: str = 'utf-8'
    
    # ID construction (if respondent ID needs to be built from multiple columns)
    id_columns_to_combine: Optional[tuple] = None  # e.g., ('IDENPA', 'NUMENTRE')
    id_separator: str = '_'
    
    def get_file_patterns(self) -> List[str]:
        """Return file patterns as a list."""
        return list(self.file_patterns)


# =============================================================================
# SURVEY REGISTRY
# =============================================================================

SURVEY_REGISTRY: Dict[str, SurveyConfig] = {
    
    'wvs': SurveyConfig(
        name='World Values Survey',
        survey_id='wvs',
        folder_name='WVS',
        respondent_id_col='D_INTERVIEW',
        country_col='B_COUNTRY',
        metadata_path='pulled_metadata/pulled_metadata_wvs.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
    ),
    
    'afrobarometer': SurveyConfig(
        name='Afrobarometer',
        survey_id='afrobarometer',
        folder_name='Afrobarometer',
        respondent_id_col='RESPNO',
        country_col='COUNTRY',
        metadata_path='pulled_metadata/pulled_metadata_afrobarometer.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
    ),
    
    'arabbarometer': SurveyConfig(
        name='Arab Barometer',
        survey_id='arabbarometer',
        folder_name='Arabbarometer',
        respondent_id_col='ID',
        country_col='COUNTRY',
        metadata_path='pulled_metadata/pulled_metadata_arabbarometer.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
    ),
    
    'asianbarometer': SurveyConfig(
        name='Asian Barometer',
        survey_id='asianbarometer',
        folder_name='Asianbarometer',
        respondent_id_col='idnumber',
        country_col='country',
        id_columns_to_combine=('country', 'idnumber'),
        id_separator='_',
        metadata_path='pulled_metadata/pulled_metadata_asianbarometer.json',
        file_patterns=('asian_barometer.csv', '*.csv', '*.dta', '*.sav'),
    ),
    
    'latinobarometer': SurveyConfig(
        name='Latinobarómetro',
        survey_id='latinobarometer',
        folder_name='Latinobarometro',
        respondent_id_col='respondent_id',  # Constructed from IDENPA + NUMENTRE
        country_col='IDENPA',
        metadata_path='pulled_metadata/pulled_metadata_latinobarometer.json',
        file_patterns=('*.sav', '*.dta', '*.csv'),
        id_columns_to_combine=('IDENPA', 'NUMENTRE'),
        id_separator='_',
    ),
    
        'ess_wave_10': SurveyConfig(
        name='European Social Survey Wave 10',
        survey_id='ess_wave_10',
        folder_name='ESS/wave_10',
        respondent_id_col='respondent_id',  # Changed: now constructed
        country_col='cntry',
        metadata_path='pulled_metadata/pulled_metadata_ess10.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
        id_columns_to_combine=('cntry', 'idno'),  # Added: country + id
        id_separator='_',
    ),

    'ess_wave_11': SurveyConfig(
        name='European Social Survey Wave 11',
        survey_id='ess_wave_11',
        folder_name='ESS/wave_11',
        respondent_id_col='respondent_id',  # Changed: now constructed
        country_col='cntry',
        metadata_path='pulled_metadata/pulled_metadata_ess11.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
        id_columns_to_combine=('cntry', 'idno'),  # Added: country + id
        id_separator='_',
    ),
}


def get_survey_config(survey_id: str) -> SurveyConfig:
    """
    Get configuration for a specific survey.
    
    Args:
        survey_id: The survey identifier (e.g., 'wvs', 'afrobarometer')
        
    Returns:
        SurveyConfig for the requested survey
        
    Raises:
        KeyError: If survey_id is not in the registry
    """
    if survey_id not in SURVEY_REGISTRY:
        available = ', '.join(sorted(SURVEY_REGISTRY.keys()))
        raise KeyError(
            f"Unknown survey: '{survey_id}'. Available surveys: {available}"
        )
    return SURVEY_REGISTRY[survey_id]


def list_surveys() -> List[str]:
    """Return list of all available survey IDs."""
    return list(SURVEY_REGISTRY.keys())


def list_surveys_detailed() -> str:
    """Return formatted string with all surveys and their details."""
    lines = ["Available Surveys:", "=" * 50]
    for survey_id, config in sorted(SURVEY_REGISTRY.items()):
        lines.append(f"\n{survey_id}:")
        lines.append(f"  Name: {config.name}")
        lines.append(f"  Folder: {config.folder_name}")
        lines.append(f"  ID Column: {config.respondent_id_col}")
        lines.append(f"  Country Column: {config.country_col}")
    return '\n'.join(lines)


# =============================================================================
# Survey Groups (for convenience in experiments)
# =============================================================================

BAROMETER_SURVEYS = ['afrobarometer', 'arabbarometer', 'asianbarometer', 'latinobarometer']
ESS_SURVEYS = ['ess_wave_10', 'ess_wave_11']
ALL_SURVEYS = list(SURVEY_REGISTRY.keys())