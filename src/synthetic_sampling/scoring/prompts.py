"""Prompt template and profile rendering for /completions scoring."""

from __future__ import annotations

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Predict how the respondent would answer "
    "the target question using their prior answers.\n\n"
    "Profile: {profile}\n\n"
    "{extra}"
    "Question: {question}\n\n"
    "Instructions: Reply with a short concise answer. No reasoning. "
    "No explanation. No extra text.\n\n"
    "Answer:"
)


def render_profile(questions: dict) -> str:
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in questions.items())


def build_prompt(inst: dict, options: list[str], arm: str) -> str:
    """Paper template, varied only in option block and emission instruction.

    Label arms append a trailing space after ``Answer:``. Without it, some
    tokenisers emit a standalone space first and the digit readout returns
    empty on every slot.

    Optional instance fields, all excluded from the PMI premises
    (``echo_qonly``, ``echo_ctxfree``), which are option-fluency premises,
    not context conditions:

    - ``extra``: a context block between profile and question (the injection
      riders: "The survey was conducted in <year>.").
    - ``profile_text``: prose replacing the rendered q:a profile (the
      narrative presentation arm, RUN_CATALOGUE B3).
    - ``reasoning``: a reasoning transcript inserted after the question (the
      reason-then-answer arm, RUN_CATALOGUE C1; the readout then happens at
      the post-reasoning position).
    """
    profile = (inst.get("profile_text")
               or render_profile(dict(inst["questions"])))
    base = PROMPT_TEMPLATE.format(
        profile=profile, extra=inst.get("extra") or "",
        question=inst["target_question"])
    if inst.get("reasoning"):
        head, sep, tail = base.partition("\n\nInstructions:")
        base = (f"{head}\n\nReasoning: {inst['reasoning'].strip()}"
                f"{sep}{tail}")

    if arm == "echo_plain":
        return base
    if arm == "echo_ctxfree":
        # Minimal domain premise; empty prompt loses first-token logprobs.
        return "Answer:"
    if arm == "echo_qonly":
        return PROMPT_TEMPLATE.format(
            profile="", extra="", question=inst["target_question"])
    if arm in ("label_num", "label_num_natural"):
        block = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        instr = "Reply with only the option number."
    else:
        raise ValueError(f"unsupported arm for build_prompt: {arm}")

    head, _, _ = base.partition("\nInstructions:")
    tail = (f"{head.rstrip()}\n\nOptions:\n{block}\n\nInstructions: {instr} "
            "No reasoning. No explanation. No extra text.\n\nAnswer:")
    return tail + " " if arm.startswith("label_") else tail
