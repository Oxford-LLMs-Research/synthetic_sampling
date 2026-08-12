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
    "make_c2_set": "thinking",
    "make_c3_set": "thinking",
    "make_c3_direct_set": "thinking",
    "generate_thinking": "thinking",
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


def test_thinking_generation_prompt_invites_reasoning():
    """Generation must not reuse the scoring 'No reasoning' format lock."""
    from synthetic_sampling.scoring.prompts import build_chat_messages

    gt = load("generate_thinking")
    inst = {
        "questions": {"How old are you?": "18-24"},
        "target_question": "Do you trust most people?",
        "option_sets": {"original": ["Yes", "No"]},
    }
    content = gt.build_thinking_messages(inst)[0]["content"]
    assert "No reasoning" not in content
    assert "Think carefully" in content
    assert content.index("Profile:") < content.index("Question:") \
        < content.index("Options:") < content.index("Instructions:")
    assert "1. Yes\n2. No" in content
    assert "reply with only the option number" in content.lower()
    # Scoring template still carries the format lock (unchanged).
    scored = build_chat_messages(
        inst, inst["option_sets"]["original"], "chat_label_num")[0]["content"]
    assert "No reasoning" in scored


def test_make_c2_set_assembles_toggle_pairs(tmp_path, capsys):
    import csv
    import json

    mod = load("make_c2_set")

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps({"example_id": e}) + "\n"
                             for e in ("e1", "e2", "e3")),
                     encoding="utf-8")
    src_row = {
        "survey": "wvs", "target_code": "Q1", "id": 7, "country": "X",
        "target_question": "Trust?", "ground_truth": "Yes",
        "ground_truth_index": 0, "questions": {"Age?": "30"},
        "option_sets": {"original": ["Yes", "No"]},
    }
    ladder = tmp_path / "ladder.jsonl"
    ladder.write_text(
        "".join(json.dumps({"example_id": e, **src_row}) + "\n"
                for e in ("e1", "e2", "e3")),
        encoding="utf-8")
    # e1 loops (two restatements, neither using a "Final answer" marker;
    # the "cannot answer 3" phrase must NOT count), e2 errored, e3 has no
    # transcript at all (capped canary shape).
    trans = tmp_path / "trans.jsonl"
    trans.write_text(
        json.dumps({"example_id": "e1",
                    "thinking_raw": "<think>the answer is 1. I cannot "
                                    "answer 3 of these. Wait - "
                                    "the answer is 1</think>1",
                    "finish_reason": "stop"}) + "\n"
        + json.dumps({"example_id": "e2", "error": "boom"}) + "\n",
        encoding="utf-8")

    out = tmp_path / "c2_set.jsonl"
    mod.main(["--transcripts", str(trans), "--out", str(out),
              "--tasks", str(tasks), "--ladder-set", str(ladder)])

    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert [r["example_id"] for r in rows] == ["e1_toff", "e1_ton"]
    assert rows[0]["arm_label"] == "direct" and "reasoning" not in rows[0]
    assert rows[1]["arm_label"] == "thinking"
    # the injected reasoning is the THINK CONTENT only: no tags, no answer
    assert rows[1]["reasoning"].startswith("the answer is 1")
    sidecar = list(csv.DictReader(
        open(tmp_path / "c2_set_parse.csv", encoding="utf-8")))
    assert {r["example_id"]: r["parse"] for r in sidecar} == {
        "e1": "bare_digit", "e2": "generation_error"}
    assert sidecar[0]["stated_index"] == "0"
    # loop census counts marker-less restatements too
    assert sidecar[0]["loop_markers"] == "2"
    # missing transcripts are absent from the set but counted in the log
    assert "1 substrate pairs without a transcript" in capsys.readouterr().out


def test_make_c3_set_thinking_only_and_close_only_block(tmp_path, capsys):
    import csv
    import json

    mod = load("make_c3_set")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"example_id": "e1"}) + "\n", encoding="utf-8")
    src = {
        "survey": "wvs", "target_code": "Q1", "id": 7, "country": "X",
        "target_question": "Trust?", "ground_truth": "Yes",
        "ground_truth_index": 0, "questions": {"Age?": "30"},
        "option_sets": {"original": ["Yes", "No"]},
    }
    ladder = tmp_path / "ladder.jsonl"
    ladder.write_text(json.dumps({"example_id": "e1", **src}) + "\n",
                      encoding="utf-8")
    # Thinking-2507 close-only shape (template pre-opened <think>).
    trans = tmp_path / "trans.jsonl"
    trans.write_text(json.dumps({
        "example_id": "e1",
        "thinking_raw": "they trust family\n</think>\n\n1",
        "finish_reason": "stop",
    }) + "\n", encoding="utf-8")
    out = tmp_path / "c3.jsonl"
    mod.main(["--transcripts", str(trans), "--out", str(out),
              "--tasks", str(tasks), "--ladder-set", str(ladder)])
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert len(rows) == 1
    assert rows[0]["example_id"] == "e1_thinking"
    assert rows[0]["arm_label"] == "thinking"
    assert rows[0]["reasoning"] == "they trust family"
    assert "toff" not in rows[0]["example_id"]
    sc = list(csv.DictReader(open(tmp_path / "c3_parse.csv", encoding="utf-8")))
    assert sc[0]["has_block"] == "True" and sc[0]["closed"] == "True"
    assert sc[0]["parse"] == "bare_digit" and sc[0]["stated_index"] == "0"


def test_make_c3_direct_set_caps_for_canary(tmp_path):
    import json

    mod = load("make_c3_direct_set")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps({"example_id": e}) + "\n"
                             for e in ("e1", "e2", "e3")), encoding="utf-8")
    src = {
        "survey": "wvs", "target_code": "Q1", "id": 7, "country": "X",
        "target_question": "Trust?", "ground_truth": "Yes",
        "ground_truth_index": 0, "questions": {"Age?": "30"},
        "option_sets": {"original": ["Yes", "No"]},
    }
    ladder = tmp_path / "ladder.jsonl"
    ladder.write_text("".join(json.dumps({"example_id": e, **src}) + "\n"
                              for e in ("e1", "e2", "e3")), encoding="utf-8")
    out = tmp_path / "direct.jsonl"
    mod.main(["--out", str(out), "--tasks", str(tasks),
              "--ladder-set", str(ladder), "--limit", "2"])
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert len(rows) == 2
    assert all(r["arm_label"] == "direct" for r in rows)
    assert all("reasoning" not in r for r in rows)
    assert rows[0]["example_id"].endswith("_direct")


def test_make_reasoned_set_assembles_2x2(tmp_path):
    """End-to-end assembly: qa+reasoned for every pair, narrative cells only
    where the pair's narrative validated (B3 exclusion inherited)."""
    import json

    mr = load("make_reasoned_set")

    def jl(path, rows):
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    src = {
        "survey": "wvs", "target_code": "Q1", "id": "R1", "country": "1",
        "target_question": "Trust?",
        "option_sets": {"original": ["Yes", "No"]},
        "ground_truth": "Yes", "ground_truth_index": 0,
        "questions": {"Age?": "18-24"},
    }
    tasks = jl(tmp_path / "tasks.jsonl",
               [{"example_id": "eA"}, {"example_id": "eB"}])
    ladder = jl(tmp_path / "ladder.jsonl",
                [{"example_id": "eA", **src}, {"example_id": "eB", **src}])
    qa_tr = jl(tmp_path / "qa_tr.jsonl", [
        {"example_id": "eA", "reasoning_raw": "Thinks.\nFinal answer: 1",
         "finish_reason": "stop"},
        {"example_id": "eB", "reasoning_raw": "Hmm.\nFinal answer: 2",
         "finish_reason": "stop"},
    ])
    # Only eA has a validated narrative (eB = inherited B3 exclusion).
    nset = jl(tmp_path / "narr_set.jsonl", [
        {"example_id": "eA_narr1", "arm_label": "narrative1",
         "base_id": "eA", "profile_text": "A young respondent.", **src},
    ])
    ntr = jl(tmp_path / "narr_tr.jsonl", [
        {"example_id": "eA_narr1",
         "reasoning_raw": "Prose thoughts.\nFinal answer: 1",
         "finish_reason": "stop"},
    ])
    out = tmp_path / "c1_set.jsonl"
    rc = mr.main([
        "--transcripts", str(qa_tr), "--out", str(out),
        "--tasks", str(tasks), "--ladder-set", str(ladder),
        "--narrative-transcripts", str(ntr), "--narrative-set", str(nset),
    ])
    assert rc == 0

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm_label"], []).append(r)
    assert sorted(by_arm) == [
        "narrative_direct", "narrative_reasoned", "qa", "reasoned"]
    assert len(by_arm["qa"]) == len(by_arm["reasoned"]) == 2
    assert len(by_arm["narrative_direct"]) == 1
    assert len(by_arm["narrative_reasoned"]) == 1
    # All four cells of eA share base_id (paired within one serving).
    assert {r["base_id"] for r in rows if r["base_id"] == "eA"} == {"eA"}
    assert sum(r["base_id"] == "eA" for r in rows) == 4
    # The reasoned cells carry TRUNCATED transcripts; direct cells none.
    nr = by_arm["narrative_reasoned"][0]
    assert "Final answer" not in nr["reasoning"]
    assert nr["profile_text"] == "A young respondent."
    assert "reasoning" not in by_arm["narrative_direct"][0]
    # Sidecar has a row per transcript with its cell.
    sidecar = (tmp_path / "c1_set_parse.csv").read_text(encoding="utf-8")
    assert "qa" in sidecar and "narrative" in sidecar
