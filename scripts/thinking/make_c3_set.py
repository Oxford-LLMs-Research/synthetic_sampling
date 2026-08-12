"""C3 stage 2: assemble the Thinking-checkpoint scoring set.

Training-axis cell only (RUN_CATALOGUE C3): one runner instance per pair
with the THINK-BLOCK CONTENT in ``reasoning`` — the stated answer after
the block is EXCLUDED, so the label readout sits at the post-reasoning,
pre-commitment position. There is no ``_toff`` twin on these weights:
Thinking-2507 cannot disable thinking, so the matched Instruct sibling
(direct chat readout, separate serving) is the comparison cell.

Sidecar schema matches C2 (has_block / closed / parse / loop_markers);
``check_c2_gates.py`` reuses it. Only eids present in the transcripts
file are assembled (canary ``--limit`` safe).

Usage:
  python scripts/thinking/make_c3_set.py \\
      --transcripts outputs/thinking/generated/thinking_qwen_...thinking-2507.jsonl \\
      --out outputs/thinking/inputs/c3_thinking_set_....jsonl
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from synthetic_sampling.scoring.thinking import parse_stated_chat, split_think

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"


def _loop_marker():
    spec = importlib.util.spec_from_file_location(
        "make_c2_set", Path(__file__).with_name("make_c2_set.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LOOP_MARKER


def main(argv: list[str] | None = None) -> int:
    loop_marker = _loop_marker()
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
    n_ok = n_err = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh, \
            open(sidecar_path, "w", encoding="utf-8", newline="") as sc:
        w = csv.DictWriter(sc, fieldnames=[
            "example_id", "parse", "stated_index", "has_block", "closed",
            "think_words", "loop_markers", "finish_reason"])
        w.writeheader()
        for eid in sorted(transcripts):
            if eid not in source:
                continue
            t = transcripts[eid]
            if "error" in t:
                n_err += 1
                w.writerow({"example_id": eid, "parse": "generation_error",
                            "stated_index": "", "has_block": "",
                            "closed": "", "think_words": "",
                            "loop_markers": "",
                            "finish_reason": t.get("error", "error")})
                continue
            raw = t.get("thinking_raw")
            if raw is None:
                n_err += 1
                w.writerow({"example_id": eid, "parse": "generation_error",
                            "stated_index": "", "has_block": "",
                            "closed": "", "think_words": "",
                            "loop_markers": "",
                            "finish_reason": "null_content"})
                continue
            src = source[eid]
            options = src["option_sets"]["original"]
            split = split_think(raw)
            parsed = parse_stated_chat(split["answer"], options)
            w.writerow({
                "example_id": eid, "parse": parsed["parse"],
                "stated_index": ("" if parsed["stated_index"] is None
                                 else parsed["stated_index"]),
                "has_block": split["has_block"], "closed": split["closed"],
                "think_words": len(split["think"].split()),
                "loop_markers": len(loop_marker.findall(split["think"])),
                "finish_reason": t.get("finish_reason"),
            })
            fh.write(json.dumps({
                "example_id": f"{eid}_thinking",
                "arm_label": "thinking",
                "base_id": eid,
                "survey": src["survey"], "target_code": src["target_code"],
                "id": src.get("id"), "country": src.get("country"),
                "target_question": src["target_question"],
                "option_sets": src["option_sets"],
                "ground_truth": src["ground_truth"],
                "ground_truth_index": src["ground_truth_index"],
                "questions": src["questions"],
                "reasoning": split["think"],
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    n_missing = len(eids & set(source)) - n_ok - n_err
    print(f"{n_ok} thinking instances assembled, "
          f"{n_err} generation errors, "
          f"{n_missing} substrate pairs without a transcript -> {args.out}")
    print(f"parse sidecar -> {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
