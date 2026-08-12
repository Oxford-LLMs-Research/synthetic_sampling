"""C2 stage 2: assemble the thinking-toggle scoring set.

Per pair, two runner instances sharing base_id so the toggle contrast is
paired within one serving:

  <eid>_toff   the plain instance (direct cell; thinking off at scoring)
  <eid>_ton    the plain instance + the THINK-BLOCK CONTENT in the
               ``reasoning`` field (thinking cell) — the stated answer
               after the block is EXCLUDED, so the label readout sits at
               the post-reasoning, pre-commitment position, and scoring
               runs with thinking off (the locked 12 Aug protocol; never
               a live-thinking shallow logprob read)

Sidecar CSV records, per transcript: has_block / closed (a generation cut
inside its think block is a datum), think_words, the stated-answer parse
(C1 protocol + bare_digit for markerless chat answers), finish_reason, and
loop_markers (how many times the think content restates a final answer —
the C1 pathology, predicted <5% here). Failures are DATA, never repaired;
an errored generation drops the pair from the set and is counted.

Usage:
  python scripts/thinking/make_c2_set.py \
      --transcripts outputs/thinking/generated/thinking_qwen_qwen3-32b.jsonl \
      --out outputs/thinking/inputs/c2_label_set_qwen_qwen3-32b.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from synthetic_sampling.scoring.thinking import parse_stated_chat, split_think

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"

LOOP_MARKER = re.compile(r"(?i)final answer\s*[:\-]?\s*(?:option\s*)?\d+")


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
            "example_id", "parse", "stated_index", "has_block", "closed",
            "think_words", "loop_markers", "finish_reason"])
        w.writeheader()
        for eid in sorted(eids):
            t = transcripts.get(eid)
            if t is None or "error" in t:
                n_err += 1
                w.writerow({"example_id": eid, "parse": "generation_error",
                            "stated_index": "", "has_block": "",
                            "closed": "", "think_words": "",
                            "loop_markers": "",
                            "finish_reason": (t or {}).get("error", "missing")})
                continue
            src = source[eid]
            options = src["option_sets"]["original"]
            split = split_think(t["thinking_raw"])
            parsed = parse_stated_chat(split["answer"], options)
            w.writerow({
                "example_id": eid, "parse": parsed["parse"],
                "stated_index": ("" if parsed["stated_index"] is None
                                 else parsed["stated_index"]),
                "has_block": split["has_block"], "closed": split["closed"],
                "think_words": len(split["think"].split()),
                "loop_markers": len(LOOP_MARKER.findall(split["think"])),
                "finish_reason": t.get("finish_reason"),
            })
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
                "example_id": f"{eid}_toff", "arm_label": "direct",
                **common}, ensure_ascii=False) + "\n")
            fh.write(json.dumps({
                "example_id": f"{eid}_ton", "arm_label": "thinking",
                "reasoning": split["think"], **common},
                ensure_ascii=False) + "\n")
            n_pairs += 1

    print(f"{n_pairs} toggle pairs assembled (direct + thinking), "
          f"{n_err} generation errors -> {args.out}")
    print(f"parse sidecar -> {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
