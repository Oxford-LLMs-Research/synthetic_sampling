"""Tests for the B3 narrative and C1 reasoning harness scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# scripts/ is organized one folder per experiment, mirroring outputs/
FOLDER = {
    "check_narratives": "narrative",
    "score_narrative_roundtrip": "narrative",
    "build_narrative_tasks": "narrative",
    "make_narrative_set": "narrative",
    "generate_reasoning": "reasoning",
    "make_reasoned_set": "reasoning",
    "convert_injection_instances": "injection",
}


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS / FOLDER[name] / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD_NARRATIVE = (
    "This respondent lives in a rural area and describes their household "
    "situation with some care. " * 12
)


def test_narrative_gates_pass_and_fail():
    cn = load("check_narratives")
    assert cn.gate_draft(GOOD_NARRATIVE) == []
    assert "length_3_words" in cn.gate_draft("Far too short.")
    assert "qa_scaffolding" in cn.gate_draft(
        GOOD_NARRATIVE + "\nQ: How old are you?")
    assert "list_formatting" in cn.gate_draft(
        GOOD_NARRATIVE + "\n- votes often\n- trusts courts")
    assert "not_third_person" in cn.gate_draft(
        "I am a farmer who distrusts the courts. " * 20)
    assert "heading" in cn.gate_draft(
        "## Portrait\n" + GOOD_NARRATIVE)


def test_narrative_near_copy_detection():
    cn = load("check_narratives")
    assert cn.trigram_jaccard(GOOD_NARRATIVE, GOOD_NARRATIVE) == 1.0
    other = ("A different person entirely, fond of markets and wary of "
             "strangers, appears here in fresh words. " * 10)
    assert cn.trigram_jaccard(GOOD_NARRATIVE, other) < 0.1


def test_roundtrip_scoring_exact_notstated_wrong():
    rt = load("score_narrative_roundtrip")
    truths = ["Somewhat satisfied", "Never", "Yes"]
    sc = rt.score_extraction(truths, {
        "1": "somewhat  satisfied",   # normalization: case and spacing
        "2": "NOT_STATED",
        "3": "No",
    })
    assert sc["n_exact"] == 1
    assert sc["n_not_stated"] == 1
    assert sc["n_wrong"] == 1
    assert sc["wrong_questions"] == [2, 3]
    assert not sc["pass"]
    perfect = rt.score_extraction(truths, {
        "1": "Somewhat satisfied", "2": "Never", "3": "Yes"})
    assert perfect["pass"] and perfect["n_exact"] == 3


def test_reasoning_parse_protocol():
    mr = load("make_reasoned_set")
    opts = ["Trust completely", "Trust somewhat", "No trust"]
    # digit primary, last marker wins
    p = mr.parse_stated_answer(
        "They lean cautious. Final answer: 1\nWait. Final answer: 3", opts)
    assert p == {"parse": "digit", "stated_index": 2}
    # out-of-range digit is a failure, not clamped
    assert mr.parse_stated_answer("Final answer: 7", opts)["stated_index"] is None
    # option-text secondary
    p = mr.parse_stated_answer("Final answer:\nTrust somewhat", opts)
    assert p == {"parse": "option_text", "stated_index": 1}
    # no marker is a datum
    assert mr.parse_stated_answer("They would trust somewhat.", opts) == \
        {"parse": "no_marker", "stated_index": None}


def test_reasoning_truncation_precommitment():
    mr = load("make_reasoned_set")
    raw = ("The respondent votes often and trusts courts.\n"
           "Final answer: 2\ntrailing junk")
    reasoning, has = mr.truncate_at_marker(raw)
    assert has
    assert "Final answer" not in reasoning
    assert reasoning.endswith("trusts courts.")
    keep, has = mr.truncate_at_marker("No marker anywhere here.")
    assert not has and keep == "No marker anywhere here."


def test_reasoning_prompt_shape():
    gr = load("generate_reasoning")
    inst = {
        "questions": {"How old are you?": "18-24"},
        "target_question": "Do you trust most people?",
        "option_sets": {"original": ["Yes", "No"]},
    }
    prompt = gr.build_reasoning_prompt(inst)
    assert prompt.index("Profile:") < prompt.index("Question:") \
        < prompt.index("Options:") < prompt.index("Final answer") \
        < prompt.index("Reasoning:")
    assert prompt.endswith("Reasoning:")
    assert "1. Yes\n2. No" in prompt
