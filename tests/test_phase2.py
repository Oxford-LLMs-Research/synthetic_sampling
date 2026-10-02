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
                  "I can't remember whether I voted", "Don't know",
                  "Not sure"):
        assert not gen._is_missing_value_label(label), label
    for label in ("NA", "N/A", "nan", "Not applicable", "Missing", "Refused",
                  "Refused to answer", "No answer", "Do not know / No answer",
                  "Not asked in this country", "Decline to answer",
                  "Can't choose", "Do not understand",
                  "Don't understand the question", "Prefer not to say"):
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


def test_unique_question_text_keeps_one_of_each_wording():
    df, metadata = _survey()
    # Two codings of one item in section "a", as ESS edulvlb / eisced.
    metadata["a"]["a0"]["question"] = metadata["a"]["a1"]["question"] = "Same?"
    gen = RespondentProfileGenerator(
        df, metadata, respondent_id_col="rid",
        missing_value_labels=phase2.MISSING_VALUE_LABELS,
        missing_value_patterns=phase2.MISSING_VALUE_PATTERNS)
    gen.set_target_questions(["T"])
    gen.unique_question_text = True
    for rid in range(6):
        profile = phase2.build_profile(gen, rid, "T", k=49)
        texts = [f["question"] for f in profile.features.values()]
        assert len(texts) == len(set(texts)) == 49   # 50 in the pool, one twin
        assert len({"a0", "a1"} & set(profile.feature_codes)) == 1


def test_wording_repairs_in_the_harmonised_view():
    from synthetic_sampling.profiles.utils import load_survey_metadata

    def questions(sid):
        return {v: m["question"] for b in load_survey_metadata(sid).values()
                if isinstance(b, dict) for v, m in b.items()
                if isinstance(m, dict) and "question" in m}

    wvs = questions("wvs")
    assert "authority" in wvs["Q45"] and "technology" not in wvs["Q45"]
    assert "technology" in wvs["Q44"]
    afro = questions("afrobarometer")
    assert len({afro["Q45PT1"], afro["Q45PT2"], afro["Q45PT3"]}) == 3
    assert afro["Q45PT2"].endswith("(second response)")
    ess = questions("ess_wave_11")
    assert ess["lnghom1"] != ess["lnghom2"]
    assert ess["anctrya1"] != ess["anctrya2"]


def test_bare_code_guard():
    labels = {"Yes", "No", "4"}
    assert phase2.is_bare_code("188009.0", labels)
    assert phase2.is_bare_code("94.0", labels)
    assert phase2.is_bare_code("1", set())
    assert not phase2.is_bare_code("4", labels)        # the label is the number
    assert not phase2.is_bare_code("Yes", labels)
    assert not phase2.is_bare_code("AR: Capital Federal", set())
    assert phase2.label_set({"values": {"1": "Yes", 2: 4}}) == {"Yes", "4"}
    assert phase2.label_set({"values": None}) == set()


def test_embedded_labels_fill_only_missing_values_maps(tmp_path, monkeypatch):
    """Latinobarometer REG / CIUDAD: pulled with no values map, labelled in
    the .sav. A pulled map is never overwritten."""
    import types
    import pyreadstat
    from synthetic_sampling.surveys import DataPaths
    from synthetic_sampling.surveys.loaders import SurveyLoader
    from synthetic_sampling.surveys.registry import get_survey_config

    cfg = get_survey_config("latinobarometer")
    folder = tmp_path / cfg.folder_name
    folder.mkdir()
    (folder / "x.sav").write_bytes(b"")
    fake = types.SimpleNamespace(variable_value_labels={
        "REG": {32001.0: "AR: Capital Federal ", 32002.0: "AR: Metropolitana"},
        "S7": {1.0: "from the file"},
    })
    monkeypatch.setattr(pyreadstat, "read_sav",
                        lambda *a, **k: (None, fake))
    metadata = {"demographics": {
        "REG": {"question": "Which region do you live in?"},
        "EDAD": {"question": "Age?", "values": None},
        "S7": {"question": "Ethnicity?", "values": {"1": "Asian"}},
    }}
    loader = SurveyLoader(DataPaths.default_bundled(tmp_path, tmp_path),
                          verbose=False)
    out = loader._fill_embedded_labels(cfg, metadata)["demographics"]
    assert out["REG"]["values"] == {"32001": "Capital Federal, Argentina",
                                    "32002": "Metropolitana, Argentina"}
    assert out["EDAD"].get("values") is None      # the file has no labels
    assert out["S7"]["values"] == {"1": "Asian"}  # pulled map untouched
    assert "values" not in metadata["demographics"]["REG"]  # input intact


def test_latino_region_labels_name_the_country_and_drop_the_ordinal():
    from synthetic_sampling.surveys.harmonise import clean_embedded_label as c
    assert c("latinobarometer", "CL: XIV Region: Los Rios") == "Los Rios, Chile"
    assert c("latinobarometer", "CL: Region Metropolitana") == "Region Metropolitana, Chile"
    assert c("latinobarometer", "MX: Circunscripcion II") == "Circunscripcion II, Mexico"
    assert c("latinobarometer", "VE: Trujillo-Pampanito II") == "Trujillo-Pampanito II, Venezuela"
    assert c("latinobarometer", "DO: Resto del pais") == "Resto del pais, Rep. Dominicana"
    assert c("wvs", "CL: XIV Region: Los Rios") == "CL: XIV Region: Los Rios"


def test_arab_q547_codes_follow_the_response_grid():
    from synthetic_sampling.profiles.utils import load_survey_metadata
    meta = load_survey_metadata("arabbarometer")
    for var in ("Q547_1", "Q547_4", "Q547_TUN2"):
        values = next(b[var]["values"] for b in meta.values()
                      if isinstance(b, dict) and var in b)
        assert values["1"] == "Strongly favor"
        assert values["4"] == "Strongly oppose"
        assert "5" not in values
