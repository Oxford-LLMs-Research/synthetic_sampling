"""
Utility Functions for Respondent Profile Generation

This module provides utility functions for loading metadata,
resolving bundled metadata paths, and verifying profile structures.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataclasses import RespondentProfile


def get_bundled_metadata_dir() -> Path:
    """
    Return the absolute path to the bundled survey metadata directory.

    This directory contains pre-built metadata JSON files for all
    supported surveys (WVS, ESS, Afrobarometer, etc.). The files
    ship with the package so users never need to locate them manually.

    Returns
    -------
    Path
        Absolute path to ``src/synthetic_sampling/profiles/metadata``
    """
    return Path(__file__).resolve().parent / 'metadata'


def load_survey_metadata(survey_id: str) -> dict:
    """
    Load the bundled metadata JSON for a given survey.

    Parameters
    ----------
    survey_id : str
        Survey identifier (e.g. ``'wvs'``, ``'ess_wave_10'``).
        Must match an entry in ``SURVEY_REGISTRY``.

    Returns
    -------
    dict
        Nested dict: section -> variable_code -> variable info

    Raises
    ------
    KeyError
        If *survey_id* is not in the registry.
    FileNotFoundError
        If the metadata file is missing from the package.
    """
    from ..config.surveys import get_survey_config

    config = get_survey_config(survey_id)
    metadata_path = get_bundled_metadata_dir() / config.metadata_path

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            f"Expected at: {metadata_path}"
        )

    return load_metadata(str(metadata_path))


def load_metadata(filepath: str) -> dict:
    """Load metadata JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify_profile_nesting(profiles: list['RespondentProfile']) -> bool:
    """
    Verify that a list of profiles (sorted by size) are properly nested.
    
    Returns True if each profile's features are a subset of the next larger one.
    """
    if len(profiles) < 2:
        return True
    
    for i in range(len(profiles) - 1):
        smaller = set(profiles[i].feature_codes)
        larger = set(profiles[i + 1].feature_codes)
        if not smaller.issubset(larger):
            return False
    return True

