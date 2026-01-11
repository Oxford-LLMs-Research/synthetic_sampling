"""
Survey configuration registry.

This module defines the configuration for each survey source, including
column mappings, file patterns, and any survey-specific preprocessing.

The SURVEY_REGISTRY is the single source of truth for survey configurations.
To add a new survey, add an entry here with the appropriate settings.
"""

from dataclasses import dataclass
<<<<<<< HEAD
from typing import Optional, List, Dict, Tuple


@dataclass(frozen=True)
class CountrySpecificConfig:
    """
    Configuration for handling country-specific variables in a survey.
    
    Some surveys (notably ESS) have variables that exist only for specific
    countries, such as education levels, religious denominations, and party
    affiliations.
    
    Attributes:
        enabled: Whether country-specific handling is active
        concept_prefixes: Mapping of variable prefix -> concept name
            e.g., {'edlv': 'education_level', 'prtvt': 'party_voted'}
            Note: for party_voted, the handler will also check prtvg* and prtvc*
        country_var: Variable containing country codes
        min_countries: Minimum countries for a prefix to be considered a group
    """
    enabled: bool = False
    concept_prefixes: Tuple[Tuple[str, str], ...] = ()
    country_var: str = 'cntry'
    min_countries: int = 3
    
    def get_prefixes_dict(self) -> Dict[str, str]:
        """Return concept_prefixes as a dict."""
        return dict(self.concept_prefixes)


# ESS configuration - handles education, religion, and party vote
# Note: party_voted uses multiple prefixes (prtvt, prtvg, prtvc) which
# the handler will expand automatically
ESS_COUNTRY_SPECIFIC_CONFIG = CountrySpecificConfig(
    enabled=True,
    concept_prefixes=(
        ('edlv', 'education_level'),
        ('rlgdn', 'religion_denomination'),
        ('rlgde', 'religion_raised'),
        ('prtvt', 'party_voted'),  # Handler expands to include prtvg*, prtvc*
        ('prtcl', 'party_close'),
    ),
    country_var='cntry',
    min_countries=3,
)
=======
from typing import Optional, List, Dict
>>>>>>> origin/main


@dataclass(frozen=True)
class SurveyConfig:
    """
    Immutable configuration for a single survey source.
<<<<<<< HEAD
=======
    
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
>>>>>>> origin/main
    """
    name: str
    survey_id: str
    folder_name: str
    respondent_id_col: str
    country_col: str
<<<<<<< HEAD
    metadata_path: str
    
    file_patterns: tuple = ('*.csv',)
    multi_file: bool = False
    encoding: str = 'utf-8'
    
    id_columns_to_combine: Optional[tuple] = None
    id_separator: str = '_'
    
    country_specific: Optional[CountrySpecificConfig] = None
    
    def get_file_patterns(self) -> List[str]:
        return list(self.file_patterns)
    
    def has_country_specific_vars(self) -> bool:
        return self.country_specific is not None and self.country_specific.enabled
=======
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
>>>>>>> origin/main


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
<<<<<<< HEAD
        respondent_id_col='respondent_id',
=======
        respondent_id_col='respondent_id',  # Constructed from IDENPA + NUMENTRE
>>>>>>> origin/main
        country_col='IDENPA',
        metadata_path='pulled_metadata/pulled_metadata_latinobarometer.json',
        file_patterns=('*.sav', '*.dta', '*.csv'),
        id_columns_to_combine=('IDENPA', 'NUMENTRE'),
        id_separator='_',
    ),
    
<<<<<<< HEAD
    'ess_wave_10': SurveyConfig(
        name='European Social Survey Wave 10',
        survey_id='ess_wave_10',
        folder_name='ESS/wave_10',
        respondent_id_col='respondent_id',
        country_col='cntry',
        metadata_path='pulled_metadata/pulled_metadata_ess10.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
        id_columns_to_combine=('cntry', 'idno'),
        id_separator='_',
        country_specific=ESS_COUNTRY_SPECIFIC_CONFIG,
=======
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
>>>>>>> origin/main
    ),

    'ess_wave_11': SurveyConfig(
        name='European Social Survey Wave 11',
        survey_id='ess_wave_11',
        folder_name='ESS/wave_11',
<<<<<<< HEAD
        respondent_id_col='respondent_id',
        country_col='cntry',
        metadata_path='pulled_metadata/pulled_metadata_ess11.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
        id_columns_to_combine=('cntry', 'idno'),
        id_separator='_',
        country_specific=ESS_COUNTRY_SPECIFIC_CONFIG,
=======
        respondent_id_col='respondent_id',  # Changed: now constructed
        country_col='cntry',
        metadata_path='pulled_metadata/pulled_metadata_ess11.json',
        file_patterns=('*.csv', '*.dta', '*.sav'),
        id_columns_to_combine=('cntry', 'idno'),  # Added: country + id
        id_separator='_',
>>>>>>> origin/main
    ),
}


def get_survey_config(survey_id: str) -> SurveyConfig:
<<<<<<< HEAD
    if survey_id not in SURVEY_REGISTRY:
        available = ', '.join(sorted(SURVEY_REGISTRY.keys()))
        raise KeyError(f"Unknown survey: '{survey_id}'. Available: {available}")
=======
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
>>>>>>> origin/main
    return SURVEY_REGISTRY[survey_id]


def list_surveys() -> List[str]:
<<<<<<< HEAD
    return list(SURVEY_REGISTRY.keys())

=======
    """Return list of all available survey IDs."""
    return list(SURVEY_REGISTRY.keys())


>>>>>>> origin/main
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


<<<<<<< HEAD
=======
# =============================================================================
# Survey Groups (for convenience in experiments)
# =============================================================================

>>>>>>> origin/main
BAROMETER_SURVEYS = ['afrobarometer', 'arabbarometer', 'asianbarometer', 'latinobarometer']
ESS_SURVEYS = ['ess_wave_10', 'ess_wave_11']
ALL_SURVEYS = list(SURVEY_REGISTRY.keys())