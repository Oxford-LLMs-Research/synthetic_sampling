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
    # Interview timing, filled by the Phase 0 census (11 Aug 2026). Values
    # verified against the microdata itself (see PAPER_STATE 11 Aug); the
    # source documentation is stored in WORK/data/questioneers/.
    interview_date_col: Optional[str] = None
    interview_date_format: Optional[str] = None
    # Fallbacks for sources without a single full-date column: a per-respondent
    # year column, and/or the component columns that exist (in the order named).
    interview_year_col: Optional[str] = None
    interview_date_parts: Optional[Tuple[str, ...]] = None
    # Wave identity and observed fieldwork period (min-max of interview dates
    # in the file we hold), so instances can carry survey-time context.
    wave_label: Optional[str] = None
    field_period: Optional[str] = None

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
        # J_INTDATE is an integer YYYYMMDD; -4/-5 are missing codes (5,484
        # rows carry them and have no interview date).
        interview_date_col="J_INTDATE",
        interview_date_format="%Y%m%d",
        interview_year_col="A_YEAR",
        wave_label="World Values Survey wave 7",
        field_period="2017-2023",
    ),
    "afrobarometer": SurveyConfig(
        name="Afrobarometer",
        survey_id="afrobarometer",
        folder_name="Afrobarometer",
        respondent_id_col="RESPNO",
        country_col="COUNTRY",
        metadata_path="pulled_metadata_afrobarometer.json",
        file_patterns=("*.csv", "*.dta", "*.sav"),
        interview_date_col="DATEINTR",
        interview_date_format="%Y-%m-%d",
        wave_label="Afrobarometer Round 9",
        field_period="2021-2023",
    ),
    "arabbarometer": SurveyConfig(
        name="Arab Barometer",
        survey_id="arabbarometer",
        folder_name="Arabbarometer",
        respondent_id_col="ID",
        country_col="COUNTRY",
        metadata_path="pulled_metadata_arabbarometer.json",
        file_patterns=("*.csv", "*.dta", "*.sav"),
        # DATE is the interview date; Q1001YEAR is a birth year, not timing.
        interview_date_col="DATE",
        interview_date_format="%Y-%m-%d",
        wave_label="Arab Barometer Wave VIII",
        field_period="2023-2024",
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
        # No full-date column; 'year' is an integer, 'month' a month NAME
        # string ('February'). No day-of-month exists in the combined file.
        interview_year_col="year",
        interview_date_parts=("year", "month"),
        wave_label="Asian Barometer wave 6",
        field_period="2021-2023",
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
        # No year column: NUMINVES is constant 23 (the 2023 study). DIAREAL /
        # MESREAL are numeric day and month; observed months are 2-4.
        interview_date_parts=("DIAREAL", "MESREAL"),
        wave_label="Latinobarometro 2023",
        field_period="Feb-Apr 2023",
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
        wave_label="European Social Survey round 10",
        field_period="2020-2022",
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
        wave_label="European Social Survey round 11",
        field_period="2023-2024",
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
