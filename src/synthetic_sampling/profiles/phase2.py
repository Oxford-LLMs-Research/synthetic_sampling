"""Phase 2 (second main run) profile construction.

One profile depth, 36 features, built to stay close to run-1: the 24-feature
core follows run-1's own path (3x2 -> 4x3 -> 6x4 by expansion, which fixes
which features land in the profile), then a round-robin top-up over the same
six sections adds the rest. So the run-1 rich profile is a strict subset of
the Phase 2 profile for the same (respondent, target, seed).

Settings carried from run-1's generate_main_dataset.py: semantic filter
all-MiniLM-L6-v2 at 0.85 against the whole target list, and a SHA-256 seed per
respondent x target. The missing-value list is run-1's with two phrases
narrowed ("decline" -> "decline to answer", "can't" -> "can't choose") now
that patterns match whole words; "Don't know" stays a valid answer.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from .dataclasses import RespondentProfile

BASE_SEED = 42
N_FEATURES = 36
SIMILARITY_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85

# Run-1's richness levels, in expansion order; the last is the core.
CORE_LEVELS = ((3, 2), (4, 3), (6, 4))
CORE_TAG = "s6m4"

MISSING_VALUE_LABELS = [
    "Missing", "Refused", "No answer", "Not asked",
    "Not applicable", "Decline to answer", "Can't choose",
    "Do not understand", "Not available", "No response",
]
MISSING_VALUE_PATTERNS = [
    "missing", "refused", "no answer", "not asked",
    "not applicable", "decline to answer", "can't choose",
    "do not understand", "not available", "no response",
    "nan", "na", "n/a",
]


def profile_tag(k: int = N_FEATURES) -> str:
    return f"{CORE_TAG}x{k}"


def respondent_target_seed(base_seed: int, respondent_id: Any,
                           target_code: str) -> int:
    """Run-1's seed for one respondent x target pair (same hash, same
    truncation), so a pair draws the same core here as it did there."""
    combined = f"{base_seed}_{respondent_id}_{target_code}"
    return int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)


def build_core(gen, respondent_id: Any, target_code: str,
               seed: int) -> RespondentProfile:
    """The run-1 rich profile, reached by run-1's expansion path."""
    profile: Optional[RespondentProfile] = None
    for n_sections, m_features in CORE_LEVELS:
        if profile is None:
            profile = gen.generate_profile(
                respondent_id=respondent_id, n_sections=n_sections,
                m_features_per_section=m_features, seed=seed,
                shuffle_features=False, target_code=target_code)
        else:
            profile = gen.expand_profile(
                profile=profile,
                add_sections=n_sections - profile.config.n_sections,
                add_features_per_section=(
                    m_features - profile.config.m_features_per_section),
                target_code=target_code)
    return profile


def build_profile(gen, respondent_id: Any, target_code: str,
                  k: int = N_FEATURES,
                  base_seed: int = BASE_SEED) -> RespondentProfile:
    """Core plus top-up to k features. May return fewer than k when the
    respondent's sections are exhausted; raises ValueError / KeyError when
    the core itself cannot be built (as run-1 did)."""
    seed = respondent_target_seed(base_seed, respondent_id, target_code)
    core = build_core(gen, respondent_id, target_code, seed)
    return gen.top_up_profile(core, k=k, seed=seed + k,
                              target_code=target_code)
