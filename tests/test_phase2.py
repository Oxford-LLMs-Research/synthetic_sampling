"""Phase 2 profile construction: whole-word missing patterns, the round-robin
top-up past the smallest section, and the run-1 seed."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from synthetic_sampling.profiles import phase2
from synthetic_sampling.profiles.generator import RespondentProfileGenerator

# Six sections; "small" holds 4 features, so an equal quota tops out at 6x4.
SECTION_SIZES = {"a": 12, "b": 10, "c": 9, "d": 8, "e": 7, "small": 4}


def _survey(n_respondents=6, missing=()):
    metadata, cols = {}, {}
    for section, size in SECTION_SIZES.items():
        metadata[section] = {}
        for i in range(size):
            code = f"{section}{i}"
            metadata[section][code] = {
                "question": f"Question {code}?",
                "values": {"1": "Yes", "2": "No", "9": "Missing"},
            }
            cols[code] = [1 + (r + i) % 2 for r in range(n_respondents)]
    metadata["a"]["T"] = {"question": "Target?",
                          "values": {"1": "Agree", "2": "Disagree"}}
    cols["T"] = [1] * n_respondents
    df = pd.DataFrame(cols)
    for code in missing:
        df.loc[0, code] = 9
    df["rid"] = range(n_respondents)
    return df, metadata


def _generator(**kw):
    df, metadata = _survey(**kw)
    gen = RespondentProfileGenerator(
        df, metadata, respondent_id_col="rid",
        missing_value_labels=phase2.MISSING_VALUE_LABELS,
        missing_value_patterns=phase2.MISSING_VALUE_PATTERNS)
    gen.set_target_questions(["T"])
    return gen


def test_missing_patterns_match_whole_words_only():
    gen = _generator()
    # Real answers that the old substring test on "na" removed.
    for label in ("National government", "China", "Ghana", "Ordinary citizens",
                  "Not at all emotionally attached", "Inactive member",
                  "The country is in decline",
                  "I can't remember whether I voted", "Don't know"):
        assert not gen._is_missing_value_label(label), label
    for label in ("NA", "N/A", "nan", "Not applicable", "Missing", "Refused",
                  "Refused to answer", "No answer", "Do not know / No answer",
                  "Not asked in this country", "Decline to answer",
                  "Can't choose", "Do not understand"):
        assert gen._is_missing_value_label(label), label


def test_seed_is_the_run1_hash():
    want = int(hashlib.sha256(b"42_BEN1123_Q27B").hexdigest()[:8], 16)
    assert phase2.respondent_target_seed(42, "BEN1123", "Q27B") == want
    assert (phase2.respondent_target_seed(42, "BEN1123", "Q9C")
            != phase2.respondent_target_seed(42, "BEN1123", "Q27B"))


def test_profile_reaches_36_past_the_smallest_section():
    gen = _generator()
    seed = phase2.respondent_target_seed(phase2.BASE_SEED, 1, "T")
    core = phase2.build_core(gen, 1, "T", seed)
    assert core.n_features == 24

    profile = phase2.build_profile(gen, 1, "T")
    assert profile.n_features == 36
    assert "T" not in profile.feature_codes
    # Run-1's rich profile is a strict subset, in the same leading order.
    assert profile.feature_codes[:24] == core.feature_codes
    # "small" is exhausted by the core; the other five take the 12 extra
    # features as evenly as possible.
    per_section = pd.Series(
        [f["section"] for f in profile.features.values()]).value_counts()
    assert per_section["small"] == 4
    others = per_section.drop("small")
    assert others.sum() == 32 and others.max() - others.min() <= 1


def test_profile_is_deterministic_and_varies_by_target_seed():
    gen = _generator()
    a = phase2.build_profile(gen, 2, "T")
    b = phase2.build_profile(gen, 2, "T")
    assert a.feature_codes == b.feature_codes
    c = phase2.build_profile(gen, 2, "T", base_seed=7)
    assert a.feature_codes != c.feature_codes


def test_top_up_skips_missing_values_and_comes_back_short_when_exhausted():
    missing = [f"b{i}" for i in range(4)]
    gen = _generator(missing=missing)
    profile = phase2.build_profile(gen, 0, "T")
    assert not set(missing) & set(profile.feature_codes)

    # 50 features exist in the pool; asking for more than the respondent can
    # supply returns what there is, without raising.
    full = phase2.build_profile(gen, 0, "T", k=200)
    assert full.n_features == sum(SECTION_SIZES.values()) - len(missing)
    assert not set(missing) & set(full.feature_codes)
