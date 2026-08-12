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


def test_smoke_fails_when_one_label_arm_is_dead():
    # A healthy raw label arm must NOT mask a dead chat arm (or vice versa):
    # each *label_num arm clears the miss threshold independently.
    good = {"Yes": -0.1, "No": -0.5}
    dead = {"Yes": float("-inf"), "No": float("-inf")}
    rows = [
        {
            "example_id": f"e{i}",
            "results": {
                "original|label_num": {"scores": good, "predicted": "Yes"},
                "original|chat_label_num": {"scores": dead, "predicted": "Yes"},
            },
        }
        for i in range(45)
    ]
    fatal, _ = check_smoke(rows)
    assert any("chat_label_num" in f for f in fatal)
    assert not any(f.startswith("label_num:") for f in fatal)


def test_split_think_handles_all_block_shapes():
    from synthetic_sampling.scoring.thinking import split_think

    r = split_think("<think>weighing the profile</think>Final answer: 3")
    assert r == {"think": "weighing the profile",
                 "answer": "Final answer: 3",
                 "has_block": True, "closed": True}
    # truncated inside the block (max_tokens): reasoning kept, no answer
    r = split_think("<think>they said neighbours lie and")
    assert r["has_block"] and not r["closed"]
    assert r["think"].startswith("they said") and r["answer"] == ""
    # empty block (thinking suppressed by template)
    r = split_think("<think></think>4. Somewhat like")
    assert r == {"think": "", "answer": "4. Somewhat like",
                 "has_block": True, "closed": True}
    # no block at all
    r = split_think("Final answer: 2")
    assert r == {"think": "", "answer": "Final answer: 2",
                 "has_block": False, "closed": False}


def test_parse_stated_chat_protocol():
    from synthetic_sampling.scoring.thinking import parse_stated_chat

    opts = ["Yes", "No", "Don't know"]
    assert parse_stated_chat("Final answer: 2", opts) == {
        "parse": "digit", "stated_index": 1}
    assert parse_stated_chat("2.", opts) == {
        "parse": "bare_digit", "stated_index": 1}
    assert parse_stated_chat("Option 3", opts) == {
        "parse": "bare_digit", "stated_index": 2}
    assert parse_stated_chat("Don't know.", opts) == {
        "parse": "option_text", "stated_index": 2}
    assert parse_stated_chat("Final answer: 9", opts)["parse"] == "digit_out_of_range"
    assert parse_stated_chat("", opts)["parse"] == "empty_answer"
    assert parse_stated_chat("It depends on many things.", opts)["parse"] == "unparseable"


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


def test_phase0_harmonisation_decisions():
    """The 11 Aug Phase 0 decisions, one assertion per decision."""
    from synthetic_sampling.surveys.harmonise import (
        apply_harmonisation, code_enters_profile, dedupe_option_labels)

    # D1: option sets present unique labels, order preserved.
    assert dedupe_option_labels(
        ["Never", "Never", "Rarely", "Sometimes", "Rarely"]) == [
        "Never", "Rarely", "Sometimes"]

    # D2: netustm wording states minutes on both waves.
    for sid, fn in (("ess_wave_10", "pulled_metadata_ess10.json"),
                    ("ess_wave_11", "pulled_metadata_ess11.json")):
        meta = load_survey_metadata(sid)
        var = next(b["netustm"] for b in meta.values()
                   if isinstance(b, dict) and "netustm" in b)
        assert "minutes" in var["question"]
        assert "hours" not in var["question"]

    # D3: hand labels present; unlabeled and no-answer codes never enter
    # a profile; substantive sentinel does.
    wvs = load_survey_metadata("wvs")
    q234a = next(b["Q234A"] for b in wvs.values()
                 if isinstance(b, dict) and "Q234A" in b)
    assert q234a["values"]["-4"] == "Not asked"
    assert not code_enters_profile("wvs", "Q234A", -4, q234a["values"])
    ess11 = load_survey_metadata("ess_wave_11")
    anc = next(b["anctrya2"] for b in ess11.values()
               if isinstance(b, dict) and "anctrya2" in b)
    assert anc["values"]["555555"] == "No second ancestry"
    assert code_enters_profile("ess_wave_11", "anctrya2", 555555, anc["values"])
    assert not code_enters_profile("wvs", "QX", "77", {"1": "Yes"})  # unlabeled
    assert not code_enters_profile("arabbarometer", "Q1015", 99999, None)
    assert code_enters_profile("arabbarometer", "Q1015", 1500, None)

    # D4: the Q48 typo is rewritten (and the census's KNOWN_DEFECTS agree).
    q48 = next(b["Q48"] for b in wvs.values()
               if isinstance(b, dict) and "Q48" in b)
    assert "None et all" not in q48["values"].values()
    assert list(q48["values"].values()).count("None at all") >= 2

    # D5: Afro case renames applied; Arab/Latino drops removed; Latino S17
    # renamed to the file column.
    afro = load_survey_metadata("afrobarometer")
    afro_vars = {v for b in afro.values() if isinstance(b, dict) for v in b}
    assert "Q45PT1" in afro_vars and "Q45pt1" not in afro_vars
    arab = load_survey_metadata("arabbarometer")
    arab_vars = {v for b in arab.values() if isinstance(b, dict) for v in b}
    assert "QGAZA1" not in arab_vars and "Q1034" not in arab_vars
    lat = load_survey_metadata("latinobarometer")
    lat_vars = {v for b in lat.values() if isinstance(b, dict) for v in b}
    assert "S17" in lat_vars and "S17.C" not in lat_vars
    assert "REEDUC.1" not in lat_vars and "P38CSN.1" not in lat_vars

    # D6: KNOWN_DEFECTS carries Q48, not the stale Q149.
    from synthetic_sampling.surveys.harmonise import KNOWN_DEFECTS
    assert any(d.var_code == "Q48" for d in KNOWN_DEFECTS)
    assert not any(d.var_code == "Q149" for d in KNOWN_DEFECTS)

    # Idempotence: harmonising the harmonised view changes nothing.
    assert apply_harmonisation(wvs, survey_id="wvs") == wvs


def test_generator_enforces_phase0_profile_rules():
    """The generator, not just the metadata, keeps non-answers out of
    profiles and duplicate labels out of option sets."""
    import pandas as pd
    from synthetic_sampling.profiles.generator import (
        RespondentProfileGenerator as ProfileGenerator)

    metadata = {
        "sec": {
            "Q234A": {
                "question": "Test q?",
                "values": {"1": "Yes", "2": "No", "-4": "Not asked"},
            },
            "Q180": {
                "question": "Justifiable?",
                "values": {"1": "Never", "2": "Never", "3": "Sometimes",
                           "4": "Always"},
            },
        }
    }
    df = pd.DataFrame(
        {"Q234A": [1, -4, 77], "Q180": [1, 2, 3], "rid": [1, 2, 3]}
    ).set_index("rid", drop=False)
    gen = ProfileGenerator(df, metadata, respondent_id_col="rid",
                           survey="wvs")

    # Routed no-answer code (-4 "Not asked") never counts as a valid value...
    assert gen._respondent_has_valid_value("Q234A", df.loc[1])
    assert not gen._respondent_has_valid_value("Q234A", df.loc[2])
    # ...and neither does an unlabeled code (77 is not in the values map).
    assert not gen._respondent_has_valid_value("Q234A", df.loc[3])

    # Option lists: the drop label is not an option, duplicates collapse.
    opts = gen._filter_valid_options(
        metadata["sec"]["Q234A"]["values"], feature_code="Q234A")
    assert opts == ["Yes", "No"]
    assert gen._filter_valid_options(
        metadata["sec"]["Q180"]["values"], feature_code="Q180") == [
        "Never", "Sometimes", "Always"]


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


def test_chat_messages_are_the_label_prompt_minus_answer_scaffold():
    from synthetic_sampling.scoring.prompts import build_chat_messages

    opts = ["Yes", "No", "Don't know"]
    msgs = build_chat_messages(INST, opts, "chat_label_num")
    assert [m["role"] for m in msgs] == ["user"]
    raw = build_prompt(INST, opts, "label_num")
    assert raw == msgs[0]["content"] + "\n\nAnswer: "
    # optional fields ride along identically (content parity by construction)
    inst = {**INST, "profile_text": "A cautious retiree.",
            "reasoning": "They said neighbours lie."}
    content = build_chat_messages(inst, opts, "chat_label_num")[0]["content"]
    assert "A cautious retiree." in content
    assert "Reasoning: They said neighbours lie." in content
    assert not content.endswith("Answer: ")


def test_chat_label_arm_scores_first_token_and_forwards_kwargs():
    from synthetic_sampling.scoring.arms import score_arm

    calls = []

    class _Resp:
        status_code = 200

        def __init__(self, top):
            self._top = top

        def json(self):
            return {"choices": [{"logprobs": {"content": [{
                "token": "1",
                "top_logprobs": [
                    {"token": k, "logprob": v} for k, v in self._top.items()
                ],
            }]}}]}

    class _Session:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            # rotation 0 favours slot 1 strongly, rotation 1 weakly, so the
            # option shown first in rotation 0 wins the average.
            top = ({"1": -0.1, "2": -3.0} if len(calls) == 1
                   else {"1": -0.5, "2": -3.0})
            return _Resp(top)

    urls = {"completions": "http://x/v1/completions",
            "chat": "http://x/v1/chat/completions"}
    rec = score_arm(_Session(), urls, {}, "m", INST, ["Yes", "No"],
                    "chat_label_num",
                    chat_template_kwargs={"enable_thinking": False})
    assert rec["predicted"] == "Yes"
    assert rec["predicted_index"] == 0
    assert all(u.endswith("/chat/completions") for u, _ in calls)
    for _, payload in calls:
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["temperature"] == 0
        assert payload["logprobs"] is True and payload["top_logprobs"] == 20
        assert payload["messages"][0]["role"] == "user"
