"""Think-block handling for the native-thinking arms (RUN_CATALOGUE C2/C3).

A thinking-mode chat generation returns ``<think>...</think>`` followed by
the visible answer. The C2 protocol (locked 12 Aug) scores the REASONING
with thinking off, so the block must be separated from the answer text
deterministically, and a malformed block is a DATUM, never repaired.

Cases handled, all covered by tests:
- normal:      <think>...</think> answer   -> (reasoning, answer, closed)
- truncated:   <think>... [max_tokens]     -> (reasoning, "", not closed)
- empty block: <think></think> answer      -> ("", answer, closed)
- close-only:  ...</think> answer          -> (reasoning, answer, closed)
  (Qwen3-Thinking-2507: the chat template pre-opens ``<think>``, so the
  model output often has only the close tag — C3)
- no block:    answer                      -> ("", answer, no block)
"""

from __future__ import annotations

import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_think(text: str) -> dict:
    """Split a thinking-mode generation into block content and answer text.

    Only the FIRST think block is treated as the block (the template emits
    exactly one); anything after its close tag is answer text. Returns
    ``think`` (block content), ``answer`` (visible text), ``has_block``,
    and ``closed`` (False when generation was cut inside the block).

    Thinking-2507 pre-opens the block in the template: generation may
    contain ``</think>`` with no opening tag. That shape is a closed block.
    """
    if THINK_OPEN not in text:
        if THINK_CLOSE in text:
            think, _, answer = text.partition(THINK_CLOSE)
            return {"think": think.strip(), "answer": answer.strip(),
                    "has_block": True, "closed": True}
        return {"think": "", "answer": text.strip(),
                "has_block": False, "closed": False}
    head, _, rest = text.partition(THINK_OPEN)
    think, sep, answer = rest.partition(THINK_CLOSE)
    if not sep:
        return {"think": think.strip(), "answer": "",
                "has_block": True, "closed": False}
    return {"think": think.strip(),
            "answer": (head.strip() + "\n" + answer.strip()).strip(),
            "has_block": True, "closed": True}


_DIGIT = re.compile(r"(?i)final answer\s*[:\-]?\s*(?:option\s*)?(\d+)")


def parse_stated_chat(answer_text: str, options: list[str]) -> dict:
    """Stated-answer parse for the post-think answer text.

    Same pre-registered protocol as C1 (digit primary, exact option text
    secondary, failures are data), applied to the visible answer segment.
    A bare leading digit counts: chat answers often start "4." with no
    marker because the instruction sits in the user turn.
    """
    matches = list(_DIGIT.finditer(answer_text))
    if matches:
        idx = int(matches[-1].group(1))
        if 1 <= idx <= len(options):
            return {"parse": "digit", "stated_index": idx - 1}
        return {"parse": "digit_out_of_range", "stated_index": None}
    first = answer_text.strip().splitlines()[0].strip() if answer_text.strip() else ""
    m = re.match(r"^(?:option\s*)?(\d+)\s*[.):]?", first, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(options):
            return {"parse": "bare_digit", "stated_index": idx - 1}
        return {"parse": "digit_out_of_range", "stated_index": None}
    cleaned = first.strip().strip(".").strip()
    for i, o in enumerate(options):
        if cleaned.lower() == o.lower():
            return {"parse": "option_text", "stated_index": i}
    if not answer_text.strip():
        return {"parse": "empty_answer", "stated_index": None}
    return {"parse": "unparseable", "stated_index": None}
