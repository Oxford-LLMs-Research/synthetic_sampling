"""
Utility Functions for Respondent Profile Generation

This module provides utility functions for loading metadata and
verifying profile structures.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataclasses import RespondentProfile


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

