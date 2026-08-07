"""Survey configuration registry: the six surveys used in the study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CountrySpecificConfig:
    """ESS-style country-specific variable handling."""

    enabled: bool = False
    concept_prefixes: Tuple[Tuple[str, str], ...] = ()
    country_var: str = "cntry"
    min_countries: int = 3

    def get_prefixes_dict(self) -> Dict[str, str]:
        return dict(self.concept_prefixes)


ESS_COUNTRY_SPECIFIC_CONFIG = CountrySpecificConfig(
    enabled=True,
    concept_prefixes=(
        ("edlv", "education_level"),
        ("rlgdn", "religion_denomination"),
        ("rlgde", "religion_raised"),
        ("prtvt", "party_voted"),
        ("prtcl", "party_close"),
    ),
    country_var="cntry",
    min_countries=3,
)


@dataclass(frozen=True)
class SurveyConfig:
    """Immutable configuration for a single survey source."""

    name: str
    survey_id: str
    folder_name: str
    respondent_id_col: str
    country_col: str
    metadata_path: str
    file_patterns: tuple = ("*.csv",)
    multi_file: bool = False
    encoding: str = "utf-8"
    id_columns_to_combine: Optional[tuple] = None
    id_separator: str = "_"
    country_specific: Optional[CountrySpecificConfig] = None
    # Phase 0 fills these; stubs so temporal work does not invent ad-hoc joins.
    interview_date_col: Optional[str] = None
    interview_date_format: Optional[str] = None

    def get_file_patterns(self) -> List[str]:
        return list(self.file_patterns)

    def has_country_specific_vars(self) -> bool:
        return self.country_specific is not None and self.country_specific.enabled


SURVEY_REGISTRY: Dict[str, SurveyConfig] = {
    "wvs": SurveyConfig(
        name="World Values Survey",
        survey_id="wvs",
        folder_name="WVS",
        respondent_id_col="D_INTERVIEW",
        country_col="B_COUNTRY",
        metadata_path="pulled_metadata_wvs.json",
        file_patterns=("*.csv", "*.dta", "*.sav"),
    ),
    "afrobarometer": SurveyConfig(
        name="Afrobarometer",
        survey_id="afrobarometer",
        folder_name="Afrobarometer",
        respondent_id_col="RESPNO",
        country_col="COUNTRY",
        metadata_path="pulled_metadata_afrobarometer.json",
        file_patterns=("*.csv", "*.dta", "*.sav"),
    ),
    "arabbarometer": SurveyConfig(
        name="Arab Barometer",
        survey_id="arabbarometer",
        folder_name="Arabbarometer",
        respondent_id_col="ID",
        country_col="COUNTRY",
        metadata_path="pulled_metadata_arabbarometer.json",
        file_patterns=("*.csv", "*.dta", "*.sav"),
    ),
    "asianbarometer": SurveyConfig(
        name="Asian Barometer",
        survey_id="asianbarometer",
        folder_name="Asianbarometer",
        respondent_id_col="idnumber",
        country_col="country",
        id_columns_to_combine=("country", "idnumber"),
        id_separator="_",
        metadata_path="pulled_metadata_asianbarometer.json",
        file_patterns=("asian_barometer.csv", "*.csv", "*.dta", "*.sav"),
    ),
    "latinobarometer": SurveyConfig(
        name="Latinobarómetro",
        survey_id="latinobarometer",
        folder_name="Latinobarometro",
        respondent_id_col="respondent_id",
        country_col="IDENPA",
        metadata_path="pulled_metadata_latinobarometer.json",
        file_patterns=("*.sav", "*.dta", "*.csv"),
        id_columns_to_combine=("IDENPA", "NUMENTRE"),
        id_separator="_",
    ),
    "ess_wave_10": SurveyConfig(
        name="European Social Survey Wave 10",
        survey_id="ess_wave_10",
        folder_name="ESS/wave_10",
        respondent_id_col="respondent_id",
        country_col="cntry",
        metadata_path="pulled_metadata_ess10.json",
        file_patterns=("*.csv", "*.dta", "*.sav"),
        id_columns_to_combine=("cntry", "idno"),
        id_separator="_",
        country_specific=ESS_COUNTRY_SPECIFIC_CONFIG,
        interview_date_col="inwds",
        interview_date_format="%Y-%m-%d %H:%M:%S",
    ),
    "ess_wave_11": SurveyConfig(
        name="European Social Survey Wave 11",
        survey_id="ess_wave_11",
        folder_name="ESS/wave_11",
        respondent_id_col="respondent_id",
        country_col="cntry",
        metadata_path="pulled_metadata_ess11.json",
        file_patterns=("*.csv", "*.dta", "*.sav"),
        id_columns_to_combine=("cntry", "idno"),
        id_separator="_",
        country_specific=ESS_COUNTRY_SPECIFIC_CONFIG,
        interview_date_col="inwds",
        interview_date_format="%Y-%m-%d %H:%M:%S",
    ),
}


def get_survey_config(survey_id: str) -> SurveyConfig:
    if survey_id not in SURVEY_REGISTRY:
        available = ", ".join(sorted(SURVEY_REGISTRY.keys()))
        raise KeyError(f"Unknown survey: '{survey_id}'. Available: {available}")
    return SURVEY_REGISTRY[survey_id]


def list_surveys() -> List[str]:
    return list(SURVEY_REGISTRY.keys())


def list_surveys_detailed() -> str:
    lines = ["Available Surveys:", "=" * 50]
    for survey_id, config in sorted(SURVEY_REGISTRY.items()):
        lines.append(f"\n{survey_id}:")
        lines.append(f"  Name: {config.name}")
        lines.append(f"  Folder: {config.folder_name}")
        lines.append(f"  ID Column: {config.respondent_id_col}")
        lines.append(f"  Country Column: {config.country_col}")
        if config.interview_date_col:
            lines.append(f"  Interview date col: {config.interview_date_col} "
                         f"(Phase 0: format {config.interview_date_format})")
    return "\n".join(lines)


BAROMETER_SURVEYS = [
    "afrobarometer", "arabbarometer", "asianbarometer", "latinobarometer",
]
ESS_SURVEYS = ["ess_wave_10", "ess_wave_11"]
ALL_SURVEYS = list(SURVEY_REGISTRY.keys())
