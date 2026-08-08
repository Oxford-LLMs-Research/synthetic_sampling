"""C1 stage 2: assemble the reason-then-answer scoring set.

Consumes the reasoning transcripts and emits, per pair, two runner instances
sharing base_id so the contrast is paired within one serving:

  <eid>_rqa        the plain k=24 profile (same-serving baseline)
  <eid>_reasoned   the profile plus the transcript TRUNCATED before its
                   "Final answer" line, carried in the ``reasoning`` field;
                   label_num then reads the distribution at the
                   post-reasoning, pre-commitment position

The pre-registered parse protocol for the STATED answer (the second
readout): take the LAST "Final answer:" marker; primary parse is a digit
1..M; secondary is an exact option-text match after the marker; anything
else is a parse failure. A missing marker, a parse failure, or a truncated
generation is a DATUM, recorded in the sidecar CSV, never repaired; the
instance still enters the label-readout arm (with the full transcript when
no marker exists to truncate at).

Usage:
  python scripts/reasoning/make_reasoned_set.py \
      --transcripts outputs/reasoning/generated/reasoning_qwen_qwen3-32b.jsonl \
      --out outputs/reasoning/inputs/reasoning_label_set_qwen_qwen3-32b.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"

MARKER = re.compile(r"(?im)^[ \t]*final answer\s*[:\-]", re.MULTILINE)
DIGIT = re.compile(r"(?i)final answer\s*[:\-]?\s*(?:option\s*)?(\d+)")


def truncate_at_marker(text: str) -> tuple[str, bool]:
    """Transcript up to (excluding) the LAST final-answer line."""
    matches = list(MARKER.finditer(text))
    if not matches:
        return text.strip(), False
    return text[: matches[-1].start()].strip(), True


def parse_stated_answer(text: str, options: list[str]) -> dict:
    """Pre-registered protocol: digit primary, exact option text secondary."""
    matches = list(DIGIT.finditer(text))
    if matches:
        idx = int(matches[-1].group(1))
        if 1 <= idx <= len(options):
            return {"parse": "digit", "stated_index": idx - 1}
        return {"parse": "digit_out_of_range", "stated_index": None}
    m = list(MARKER.finditer(text))
    if m:
        tail = text[m[-1].end():].strip().splitlines()
        first = tail[0].strip().strip(".").strip() if tail else ""
        for i, o in enumerate(options):
            if first.lower() == o.lower():
                return {"parse": "option_text", "stated_index": i}
        return {"parse": "unparseable_after_marker", "stated_index": None}
    return {"parse": "no_marker", "stated_index": None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--ladder-set", type=Path, default=LADDER_SET)
    args = ap.parse_args(argv)

    eids = set()
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            eids.add(json.loads(line)["example_id"])
    source = {}
    with open(args.ladder_set, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in eids:
                source[r["example_id"]] = r

    transcripts = {}
    with open(args.transcripts, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            transcripts[r["example_id"]] = r

    sidecar_path = args.out.with_name(args.out.stem + "_parse.csv")
    n_pairs = n_err = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh, \
            open(sidecar_path, "w", encoding="utf-8", newline="") as sc:
        w = csv.DictWriter(sc, fieldnames=[
            "example_id", "parse", "stated_index", "has_marker",
            "finish_reason", "reasoning_words"])
        w.writeheader()
        for eid in sorted(eids):
            t = transcripts.get(eid)
            if t is None or "error" in t:
                n_err += 1
                w.writerow({"example_id": eid, "parse": "generation_error",
                            "stated_index": "", "has_marker": "",
                            "finish_reason": (t or {}).get("error", "missing"),
                            "reasoning_words": ""})
                continue
            src = source[eid]
            options = src["option_sets"]["original"]
            raw = t["reasoning_raw"]
            reasoning, has_marker = truncate_at_marker(raw)
            parsed = parse_stated_answer(raw, options)
            w.writerow({"example_id": eid, **{
                "parse": parsed["parse"],
                "stated_index": ("" if parsed["stated_index"] is None
                                 else parsed["stated_index"]),
                "has_marker": has_marker,
                "finish_reason": t.get("finish_reason"),
                "reasoning_words": len(reasoning.split())}})
            common = {
                "base_id": eid,
                "survey": src["survey"], "target_code": src["target_code"],
                "id": src.get("id"), "country": src.get("country"),
                "target_question": src["target_question"],
                "option_sets": src["option_sets"],
                "ground_truth": src["ground_truth"],
                "ground_truth_index": src["ground_truth_index"],
                "questions": src["questions"],
            }
            fh.write(json.dumps({
                "example_id": f"{eid}_rqa", "arm_label": "qa", **common},
                ensure_ascii=False) + "\n")
            fh.write(json.dumps({
                "example_id": f"{eid}_reasoned", "arm_label": "reasoned",
                "reasoning": reasoning, **common},
                ensure_ascii=False) + "\n")
            n_pairs += 1

    print(f"{n_pairs} pairs assembled (2 instances each), "
          f"{n_err} generation errors -> {args.out}")
    print(f"parse sidecar -> {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
