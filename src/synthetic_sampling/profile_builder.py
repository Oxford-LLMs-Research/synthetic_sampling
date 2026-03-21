"""
High-level interface for building respondent profiles from survey metadata.

``ProfileBuilder`` is the recommended entry point for external users. It
wraps the internal ``RespondentProfileGenerator`` machinery and provides:

* **Metadata exploration** -- browse surveys, sections, and variables
  without needing raw data files.
* **Data loading** -- point at a CSV / DTA / SAV file (or pass a DataFrame)
  and the builder handles the rest.
* **Profile generation** -- generate stratified respondent profiles and
  prediction instances with sensible defaults.
* **Prompt formatting** -- convert profiles to LLM-ready text using 10
  built-in presets or a custom template.

Quick start::

    from synthetic_sampling import ProfileBuilder

    builder = ProfileBuilder('wvs')
    builder.list_sections()
    builder.list_variables('demographics')

    builder.load_data('~/data/WVS/wvs.csv')
    instance = builder.generate_instance(
        respondent_id=12345, target_code='Q35A',
        n_sections=3, m_features_per_section=2, seed=42,
    )
    print(instance.to_prompt(profile_format='qa'))
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from .config.surveys import SURVEY_REGISTRY, SurveyConfig, get_survey_config
from .loaders.file_io import load_file
from .profiles.dataclasses import (
    PredictionInstance,
    ProfileConfig,
    RespondentProfile,
)
from .profiles.formats import PROFILE_FORMATS, list_profile_formats
from .profiles.generator import RespondentProfileGenerator
from .profiles.utils import get_bundled_metadata_dir, load_metadata


class ProfileBuilder:
    """
    High-level interface for exploring survey metadata and generating
    respondent profiles.

    Parameters
    ----------
    survey : str or dict
        Either a registered survey id (``'wvs'``, ``'ess_wave_10'``, ...)
        or a raw metadata dict with the standard schema
        ``{section: {var_code: {question, description, values}}}``.
    respondent_id_col : str, optional
        Column name that uniquely identifies respondents.  When *survey*
        is a registered id this is filled from the registry; override if
        your data uses a different column.
    country_col : str, optional
        Column name for the country variable.  Same defaulting logic as
        *respondent_id_col*.
    missing_value_labels : list[str], optional
        Exact labels treated as missing / artifact values.
    missing_value_patterns : list[str], optional
        Case-insensitive substring patterns for missing values.
    use_semantic_filtering : bool
        Enable sentence-transformer similarity filtering (requires the
        ``sentence-transformers`` package).  Default ``False``.
    similarity_model : str
        Model name for semantic filtering.
    similarity_threshold : float
        Cosine-similarity threshold for excluding features.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        survey: Union[str, dict],
        *,
        respondent_id_col: Optional[str] = None,
        country_col: Optional[str] = None,
        missing_value_labels: Optional[List[str]] = None,
        missing_value_patterns: Optional[List[str]] = None,
        use_semantic_filtering: bool = False,
        similarity_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
    ):
        # Resolve metadata ------------------------------------------------
        if isinstance(survey, str):
            self._survey_config: Optional[SurveyConfig] = get_survey_config(survey)
            self._survey_id: Optional[str] = survey
            metadata_path = (
                get_bundled_metadata_dir() / self._survey_config.metadata_path
            )
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Bundled metadata not found: {metadata_path}"
                )
            self._metadata: dict = load_metadata(str(metadata_path))
        elif isinstance(survey, dict):
            self._survey_config = None
            self._survey_id = None
            self._metadata = survey
        else:
            raise TypeError(
                "'survey' must be a survey id string or a metadata dict"
            )

        # Column overrides -------------------------------------------------
        if respondent_id_col is not None:
            self._respondent_id_col = respondent_id_col
        elif self._survey_config is not None:
            self._respondent_id_col = self._survey_config.respondent_id_col
        else:
            self._respondent_id_col = None

        if country_col is not None:
            self._country_col = country_col
        elif self._survey_config is not None:
            self._country_col = self._survey_config.country_col
        else:
            self._country_col = None

        # Generator settings -----------------------------------------------
        _default_labels = [
            "Missing", "No answer", "Refused",
            "Not applicable", "Not asked",
        ]
        _default_patterns = [
            "not asked", "missing", "refused", "nan", "na",
            "not available", "no response", "not applicable",
        ]
        self._missing_value_labels = (
            missing_value_labels if missing_value_labels is not None
            else _default_labels
        )
        self._missing_value_patterns = (
            missing_value_patterns if missing_value_patterns is not None
            else _default_patterns
        )
        self._use_semantic_filtering = use_semantic_filtering
        self._similarity_model = similarity_model if use_semantic_filtering else None
        self._similarity_threshold = similarity_threshold

        # Data / generator (populated by load_data) -------------------------
        self._data: Optional[pd.DataFrame] = None
        self._generator: Optional[RespondentProfileGenerator] = None

    # ------------------------------------------------------------------
    # Metadata exploration  (no data required)
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict:
        """Raw metadata dict (section -> var_code -> info)."""
        return self._metadata

    @property
    def survey_id(self) -> Optional[str]:
        """Registered survey id, or ``None`` for custom metadata."""
        return self._survey_id

    def list_sections(self, include_excluded: bool = False) -> List[str]:
        """
        Return the section names in the metadata.

        Parameters
        ----------
        include_excluded : bool
            If ``True``, include the ``EXCLUDED`` section (if present).
        """
        sections = list(self._metadata.keys())
        if not include_excluded:
            sections = [s for s in sections if s != "EXCLUDED"]
        return sections

    def list_variables(
        self, section: Optional[str] = None
    ) -> Dict[str, dict]:
        """
        Return variables and their metadata.

        Parameters
        ----------
        section : str, optional
            If given, only variables from that section.  Otherwise all
            non-excluded variables are returned.

        Returns
        -------
        dict
            ``{var_code: {question, description, values, ...}}``
        """
        if section is not None:
            if section not in self._metadata:
                available = ", ".join(self.list_sections(include_excluded=True))
                raise KeyError(
                    f"Section '{section}' not found. Available: {available}"
                )
            return dict(self._metadata[section])

        merged: Dict[str, dict] = {}
        for sec, variables in self._metadata.items():
            if sec == "EXCLUDED":
                continue
            merged.update(variables)
        return merged

    def describe_variable(self, var_code: str) -> str:
        """
        Return a human-readable description of a variable.

        Includes the variable code, section, question text, and all
        value labels.
        """
        for section, variables in self._metadata.items():
            if var_code in variables:
                info = variables[var_code]
                lines = [
                    f"Variable : {var_code}",
                    f"Section  : {section}",
                    f"Question : {info.get('question', 'N/A')}",
                    f"Desc     : {info.get('description', 'N/A')}",
                ]
                values = info.get("values", {})
                if isinstance(values, dict) and values:
                    lines.append("Values   :")
                    for code, label in values.items():
                        lines.append(f"  {code} = {label}")
                return "\n".join(lines)

        available_codes = list(self.list_variables().keys())[:10]
        raise KeyError(
            f"Variable '{var_code}' not found in metadata. "
            f"Example codes: {available_codes}"
        )

    def variable_options(self, var_code: str) -> Dict[str, str]:
        """
        Return the value mapping for a variable (code -> label).
        """
        for section, variables in self._metadata.items():
            if var_code in variables:
                values = variables[var_code].get("values", {})
                if isinstance(values, dict):
                    return dict(values)
                return {}
        raise KeyError(f"Variable '{var_code}' not found in metadata")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @property
    def data_loaded(self) -> bool:
        """Whether survey data has been loaded."""
        return self._data is not None

    def load_data(
        self,
        data: Union[str, Path, pd.DataFrame],
        *,
        respondent_id_col: Optional[str] = None,
        country_col: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> "ProfileBuilder":
        """
        Load survey response data and initialise the profile generator.

        Parameters
        ----------
        data : str, Path, or DataFrame
            Path to a survey data file (CSV / DTA / SAV) or a
            pre-loaded DataFrame.
        respondent_id_col : str, optional
            Override the respondent id column for this data load.
        country_col : str, optional
            Override the country column for this data load.
        encoding : str
            Encoding for CSV files (ignored for other formats).

        Returns
        -------
        ProfileBuilder
            ``self``, for method chaining.
        """
        if isinstance(data, pd.DataFrame):
            self._data = data
        else:
            path = Path(data).expanduser()
            self._data = load_file(path, encoding=encoding)

        rid_col = respondent_id_col or self._respondent_id_col
        cntry_col = country_col or self._country_col

        # Handle composite-id surveys (e.g. Latinobarometer, Asian Barometer)
        if (
            self._survey_config is not None
            and self._survey_config.id_columns_to_combine is not None
        ):
            cols = list(self._survey_config.id_columns_to_combine)
            sep = self._survey_config.id_separator
            if all(c in self._data.columns for c in cols):
                self._data[rid_col] = (
                    self._data[cols]
                    .astype(str)
                    .agg(sep.join, axis=1)
                )

        self._generator = RespondentProfileGenerator(
            survey_data=self._data,
            metadata=self._metadata,
            respondent_id_col=rid_col,
            country_col=cntry_col,
            survey=self._survey_id,
            missing_value_labels=self._missing_value_labels,
            missing_value_patterns=self._missing_value_patterns,
            similarity_model=self._similarity_model,
            similarity_threshold=self._similarity_threshold,
        )
        return self

    # ------------------------------------------------------------------
    # Profile generation  (data required)
    # ------------------------------------------------------------------

    def _require_data(self) -> RespondentProfileGenerator:
        if self._generator is None:
            raise RuntimeError(
                "No survey data loaded. Call .load_data() first."
            )
        return self._generator

    def set_target_questions(
        self,
        target_codes: List[str],
        **kwargs: Any,
    ) -> "ProfileBuilder":
        """
        Register target questions and exclude them from the feature pool.

        Delegates to
        :pymethod:`RespondentProfileGenerator.set_target_questions`.

        Returns
        -------
        ProfileBuilder
            ``self``, for method chaining.
        """
        gen = self._require_data()
        gen.set_target_questions(target_codes, **kwargs)
        return self

    def set_always_include(self, feature_codes: List[str]) -> "ProfileBuilder":
        """
        Mark features that must appear in every profile.

        Returns
        -------
        ProfileBuilder
            ``self``, for method chaining.
        """
        gen = self._require_data()
        gen.set_always_include(feature_codes)
        return self

    def generate_profile(
        self,
        respondent_id: Union[int, str],
        n_sections: int = 3,
        m_features_per_section: int = 2,
        seed: int = 42,
        shuffle_features: bool = False,
        target_code: Optional[str] = None,
    ) -> RespondentProfile:
        """
        Generate a respondent profile with stratified random sampling.

        Parameters
        ----------
        respondent_id : int or str
            Respondent identifier in the loaded data.
        n_sections : int
            Number of thematic sections to sample.
        m_features_per_section : int
            Features to draw from each section.
        seed : int
            Random seed for reproducibility.
        shuffle_features : bool
            Shuffle feature order within the profile.
        target_code : str, optional
            If provided, applies per-target semantic similarity exclusions.

        Returns
        -------
        RespondentProfile
        """
        gen = self._require_data()
        return gen.generate_profile(
            respondent_id=respondent_id,
            n_sections=n_sections,
            m_features_per_section=m_features_per_section,
            seed=seed,
            shuffle_features=shuffle_features,
            target_code=target_code,
        )

    def generate_instance(
        self,
        respondent_id: Union[int, str],
        target_code: str,
        n_sections: int = 3,
        m_features_per_section: int = 2,
        seed: int = 42,
        shuffle_features: bool = False,
        skip_missing_targets: bool = True,
    ) -> Optional[PredictionInstance]:
        """
        Generate a single prediction instance (profile + target + ground truth).

        Call :pymeth:`set_target_questions` before using this method.

        Returns
        -------
        PredictionInstance or None
            ``None`` when the respondent's answer is missing and
            *skip_missing_targets* is ``True``.
        """
        gen = self._require_data()
        return gen.generate_prediction_instance(
            respondent_id=respondent_id,
            target_code=target_code,
            n_sections=n_sections,
            m_features_per_section=m_features_per_section,
            seed=seed,
            shuffle_features=shuffle_features,
            skip_missing_targets=skip_missing_targets,
        )

    def generate_dataset(
        self,
        respondent_ids: List[Union[int, str]],
        n_sections: int = 3,
        m_features_per_section: int = 2,
        seed: int = 42,
        target_codes: Optional[List[str]] = None,
        shuffle_features: bool = False,
        skip_missing_targets: bool = True,
        as_dicts: bool = True,
    ) -> list:
        """
        Generate prediction instances for multiple respondents.

        Parameters
        ----------
        respondent_ids : list
            Respondents to include.
        as_dicts : bool
            If ``True`` return list of dicts; otherwise
            :class:`PredictionInstance` objects.

        Returns
        -------
        list
        """
        gen = self._require_data()
        return gen.generate_dataset_as_list(
            respondent_ids=respondent_ids,
            n_sections=n_sections,
            m_features_per_section=m_features_per_section,
            seed=seed,
            target_codes=target_codes,
            shuffle_features=shuffle_features,
            skip_missing_targets=skip_missing_targets,
            as_dicts=as_dicts,
        )

    # ------------------------------------------------------------------
    # Convenience: available respondents
    # ------------------------------------------------------------------

    def respondent_ids(self, n: Optional[int] = None) -> list:
        """
        Return respondent ids from the loaded data.

        Parameters
        ----------
        n : int, optional
            If given, return the first *n* ids.
        """
        gen = self._require_data()
        if self._respondent_id_col and self._respondent_id_col in self._data.columns:
            ids = self._data[self._respondent_id_col].unique().tolist()
        else:
            ids = self._data.index.unique().tolist()
        if n is not None:
            ids = ids[:n]
        return ids

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def available_formats() -> Dict[str, str]:
        """
        Return all built-in profile format presets with examples.

        Returns
        -------
        dict
            ``{preset_name: example_output}``
        """
        return list_profile_formats()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        survey_label = self._survey_id or "custom"
        sections = self.list_sections()
        n_vars = sum(
            len(v)
            for s, v in self._metadata.items()
            if s != "EXCLUDED"
        )
        data_info = (
            f"{len(self._data)} rows"
            if self._data is not None
            else "no data loaded"
        )
        return (
            f"ProfileBuilder(survey={survey_label!r}, "
            f"sections={len(sections)}, variables={n_vars}, "
            f"data={data_info})"
        )
