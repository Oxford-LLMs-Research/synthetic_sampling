"""Tests for scoring prompts, arms contract, resume, and coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthetic_sampling.scoring.prompts import build_prompt, render_profile
from synthetic_sampling.scoring.arms import DEFAULT_ARMS
from synthetic_sampling.scoring.runner import _load_done, _select_instances
from synthetic_sampling.checks.coverage import coverage_report
from synthetic_sampling.checks.smoke import check_smoke
from synthetic_sampling.checks.number_verify import verify_numbers
from synthetic_sampling.analysis.metrics import normalized_accuracy, auc
from synthetic_sampling.surveys.registry import SURVEY_REGISTRY, list_surveys
from synthetic_sampling.surveys.harmonise import scan_option_sets, filter_missingness_codes
from synthetic_sampling.profiles.utils import get_bundled_metadata_dir, load_survey_metadata
from synthetic_sampling.profiles.leakage import target_exclusions


INST = {
    "example_id": "wvs_1_Q1_s3m3",
    "questions": {"How old are you?": "18-24", "Gender?": "Female"},
    "target_question": "Do you trust most people?",
}


def test_default_arms_include_label_num_and_echo():
    assert DEFAULT_ARMS[0] == "label_num"
    assert "echo_plain" in DEFAULT_ARMS
    assert "echo_qonly" in DEFAULT_ARMS
    assert "echo_ctxfree" in DEFAULT_ARMS
    assert "generate" not in DEFAULT_ARMS
    assert "label_alpha" not in DEFAULT_ARMS


def test_label_num_prompt_has_trailing_space():
    opts = ["Yes", "No", "Don't know"]
    prompt = build_prompt(INST, opts, "label_num")
    assert prompt.endswith("Answer: ")
    assert "1. Yes" in prompt
    assert "Reply with only the option number." in prompt


def test_echo_plain_prompt_no_trailing_space_after_answer():
    prompt = build_prompt(INST, ["Yes", "No"], "echo_plain")
    assert prompt.endswith("Answer:")
    assert "Options:" not in prompt


def test_echo_ctxfree_is_answer_only():
    assert build_prompt(INST, ["Yes"], "echo_ctxfree") == "Answer:"


def test_render_profile_qa():
    text = render_profile({"Q?": "A"})
    assert text == "Q: Q?\nA: A"


def test_resume_skips_done(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"example_id": "a"}) + "\n"
        + json.dumps({"example_id": "b"}) + "\n",
        encoding="utf-8",
    )
    assert _load_done(out) == {"a", "b"}

    inp = tmp_path / "in.jsonl"
    rows = [
        {
            "example_id": "a",
            "option_sets": {"original": ["Yes", "No"]},
            "questions": {},
            "target_question": "Q?",
        },
        {
            "example_id": "c",
            "option_sets": {"original": ["Yes", "No"]},
            "questions": {},
            "target_question": "Q?",
        },
    ]
    inp.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    selected = _select_instances(inp, {"a"}, replicate_frac=0.0)
    assert [r["example_id"] for r in selected] == ["c"]


def test_coverage_report(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inst = {
        "example_id": "e1",
        "results": {
            "original|label_num": {
                "scores": {"Yes": -0.1, "No": -0.2},
                "predicted": "Yes",
            },
            "original|echo_plain": {
                "scores": {"Yes": -1.0, "No": -2.0},
                "predicted": "Yes",
            },
        },
    }
    inp.write_text("{}\n", encoding="utf-8")
    out.write_text(json.dumps(inst) + "\n", encoding="utf-8")
    text, ok = coverage_report(out, inp)
    assert ok
    assert "VERDICT" in text
    assert "label_num" in text


def test_smoke_passes_on_enough_good_rows():
    opts = {"Yes": -0.1, "No": -0.5}
    rows = [
        {
            "example_id": f"e{i}",
            "results": {
                "original|label_num": {"scores": opts, "predicted": "Yes"},
            },
        }
        for i in range(45)
    ]
    fatal, notes = check_smoke(rows)
    assert fatal == []
    assert any("label_num" in n for n in notes)


def test_normalized_accuracy_and_auc():
    assert normalized_accuracy(0.5, 2) == pytest.approx(0.0)
    assert normalized_accuracy(1.0, 2) == pytest.approx(1.0)
    scores = __import__("numpy").array([0.1, 0.9, 0.2, 0.8])
    labels = __import__("numpy").array([0, 1, 0, 1])
    assert auc(scores, labels) == pytest.approx(1.0)


def test_number_verify(tmp_path: Path):
    csv = tmp_path / "src.csv"
    csv.write_text("metric,value\nnorm_acc,0.25\n", encoding="utf-8")
    assert verify_numbers({"norm_acc": 0.25}, csv) == []
    bad = verify_numbers({"norm_acc": 0.26}, csv)
    assert bad and "mismatch" in bad[0]


def test_registry_and_bundled_metadata():
    ids = list_surveys()
    assert "wvs" in ids
    assert "ess_wave_10" in ids
    assert SURVEY_REGISTRY["ess_wave_10"].has_country_specific_vars()
    assert SURVEY_REGISTRY["ess_wave_10"].interview_date_col == "inwds"
    # Phase 0 (11 Aug 2026): every survey carries wave identity and at least
    # one interview-timing source, verified against the microdata by
    # scripts/phase0/verify_interview_dates.py.
    for sid, cfg in SURVEY_REGISTRY.items():
        assert cfg.wave_label and cfg.field_period, sid
        assert (cfg.interview_date_col or cfg.interview_year_col
                or cfg.interview_date_parts), sid
    assert SURVEY_REGISTRY["wvs"].interview_date_col == "J_INTDATE"
    assert SURVEY_REGISTRY["latinobarometer"].interview_date_parts == (
        "DIAREAL", "MESREAL")
    meta_dir = get_bundled_metadata_dir()
    assert (meta_dir / "pulled_metadata_wvs.json").exists()
    meta = load_survey_metadata("wvs")
    assert isinstance(meta, dict) and meta


def test_hygiene_scan_and_filter():
    metadata = {
        "sec": {
            "Q1": {"values": {"1": "None at all", "2": "None et all", "3": "94.0"}},
        }
    }
    report = scan_option_sets(metadata, survey_id="wvs")
    kinds = {f.kind for f in report.findings}
    assert "bare_numeric" in kinds or "typo_duplicate" in kinds
    assert filter_missingness_codes(["Yes", "94.0"], ["94.0"]) == ["Yes"]


def test_leakage_exclusions():
    excl = target_exclusions(["Q1", "Q2"])
    assert excl == {"Q1", "Q2"}


def test_extra_field_renders_between_profile_and_question():
    inst = {**INST, "extra": "The survey was conducted in 2024.\n\n"}
    for arm in ("label_num", "echo_plain"):
        prompt = build_prompt(inst, ["Yes", "No"], arm)
        assert "The survey was conducted in 2024." in prompt
        assert (prompt.index("Profile:")
                < prompt.index("The survey was conducted")
                < prompt.index("Question:"))
    # PMI premises exclude the context line.
    assert "2024" not in build_prompt(inst, ["Yes", "No"], "echo_qonly")
    assert "2024" not in build_prompt(inst, ["Yes", "No"], "echo_ctxfree")
    # Absent field falls back to the paper template unchanged.
    assert build_prompt(dict(INST), ["Yes", "No"], "echo_plain") == \
        build_prompt({**INST, "extra": None}, ["Yes", "No"], "echo_plain")


def test_injection_converter_conditions():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "convert_injection_instances",
        Path(__file__).resolve().parents[1] / "scripts" / "injection"
        / "convert_injection_instances.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    src = {
        "example_id": "wvs_1_Q1_s6m4", "base_id": "wvs_1_Q1",
        "survey": "wvs", "target_code": "Q1", "id": "1", "country": "20.0",
        "questions": {"How old are you?": "18-24"},
        "target_question": "Do you trust most people?",
        "options": ["Yes", "No"], "answer": "No",
        "country_question": "In which country do you live?",
        "country_name": "Kenya", "country_placebo_name": "India",
        "survey_year": 2024, "survey_year_placebo": 2020,
        "interview_date": "2024-07-09",
    }
    base = conv.convert_one("country", src, "baseline")
    assert base["example_id"] == "wvs_1_Q1_s6m4_baseline"
    assert base["base_id"] == "wvs_1_Q1_s6m4"
    assert base["ground_truth_index"] == 1
    assert base["option_sets"] == {"original": ["Yes", "No"]}
    assert "extra" not in base and "country" not in str(base["questions"])

    real = conv.convert_one("country", src, "with_country")
    assert list(real["questions"])[-1] == "In which country do you live?"
    assert real["questions"]["In which country do you live?"] == "Kenya"
    placebo = conv.convert_one("country", src, "with_country_placebo")
    assert placebo["questions"]["In which country do you live?"] == "India"

    assert conv.convert_one("temporal", src, "with_year")["extra"] == \
        "The survey was conducted in 2024.\n\n"
    assert conv.convert_one("temporal", src, "with_year_placebo")["extra"] == \
        "The survey was conducted in 2020.\n\n"
    assert conv.convert_one("temporal", src, "with_date")["extra"] == \
        "The interview took place on 2024-07-09.\n\n"
    assert conv.conditions("temporal", {**src, "interview_date": None}) == \
        ["baseline", "with_year", "with_year_placebo",
         "with_country", "with_country_and_year"]


def test_injection_converter_combined_cell():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "convert_injection_instances2",
        Path(__file__).resolve().parents[1] / "scripts" / "injection"
        / "convert_injection_instances.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    src = {
        "example_id": "wvs_1_Q1_s6m4", "survey": "wvs", "target_code": "Q1",
        "id": "1", "country": "404", "questions": {"How old are you?": "18-24"},
        "target_question": "Do you trust most people?",
        "options": ["Yes", "No"], "answer": "No",
        "survey_year": 2024, "survey_year_placebo": 2020,
        "interview_date": "2024-07-09",
        "country_question": conv.COUNTRY_QUESTION, "country_name": "Kenya",
    }
    # Temporal substrate with a resolved country name gains the 2x2 cells.
    assert conv.conditions("temporal", src) == [
        "baseline", "with_year", "with_year_placebo", "with_date",
        "with_country", "with_country_and_year"]
    # Without a name the original condition list is unchanged.
    bare = {k: v for k, v in src.items()
            if k not in ("country_question", "country_name")}
    assert conv.conditions("temporal", bare) == [
        "baseline", "with_year", "with_year_placebo", "with_date"]

    combo = conv.convert_one("temporal", src, "with_country_and_year")
    assert list(combo["questions"])[-1] == conv.COUNTRY_QUESTION
    assert combo["questions"][conv.COUNTRY_QUESTION] == "Kenya"
    assert combo["extra"] == "The survey was conducted in 2024.\n\n"
    solo = conv.convert_one("temporal", src, "with_country")
    assert solo["questions"][conv.COUNTRY_QUESTION] == "Kenya"
    assert "extra" not in solo


def test_profile_text_replaces_qa_rendering():
    inst = {**INST, "profile_text": "A young woman who distrusts strangers."}
    for arm in ("label_num", "echo_plain"):
        prompt = build_prompt(inst, ["Yes", "No"], arm)
        assert "A young woman who distrusts strangers." in prompt
        assert "Q: How old are you?" not in prompt
    # PMI premises exclude the prose profile entirely.
    assert "young woman" not in build_prompt(inst, ["Yes", "No"], "echo_qonly")


def test_reasoning_inserted_between_question_and_options():
    inst = {**INST, "reasoning": "They trust family but said neighbours lie."}
    prompt = build_prompt(inst, ["Yes", "No"], "label_num")
    q = prompt.index("Question:")
    r = prompt.index("Reasoning: They trust family")
    o = prompt.index("Options:")
    i = prompt.index("Instructions:")
    assert q < r < o < i
    assert prompt.endswith("Answer: ")
    # echo path keeps reasoning before Instructions too
    p2 = build_prompt(inst, ["Yes", "No"], "echo_plain")
    assert p2.index("Question:") < p2.index("Reasoning:") < p2.index("Instructions:")
    assert "neighbours" not in build_prompt(inst, ["Yes", "No"], "echo_qonly")
