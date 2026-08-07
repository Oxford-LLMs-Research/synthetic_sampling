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
