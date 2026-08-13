"""A3: assemble the default-option (DK) scoring set (RUN_CATALOGUE A3).

The question: do "Don't know" / refusal options enter the numbered list?
Under old echo the model picked DK on ~47% of instances where humans picked
it 1.5% of the time (instrument-driven: echo's fluency term favours short
frequent strings). label_num kills that artifact but not the design
question, and Phase 2's distribution-first estimand makes the predicted DK
share a direct function of this convention.

TWO cells, not the catalogue's three (amended 13 Aug): label_num scores
under a full latin square, every option visits every digit slot, so
absolute option position is cancelled by construction and a forced-last
cell would measure only relative-order effects — and DK options already
sit at the tail of all 21 original lists anyway. What survives rotation:

  <eid>_dkpresent   dk_present   options exactly as the survey had them
  <eid>_dkabsent    dk_absent    non-substantive options removed,
                                 ground_truth_index remapped (null when the
                                 respondent's true answer WAS one of them)

Non-substantive options are the frozen set below — the only such strings
in the substrate (verified: no other option matches a broad DK regex).

Substrate: every k=24 informative ladder pair whose target carries at
least one non-substantive option, duplicate-option hygiene targets
excluded. No sampling: per-target DK shares need all the pairs there are.

Sidecar a3_dk_meta.json records, per target, the removed options and the
HUMAN DK-truth share on these same rows — the matched comparison the
predicted DK share is read against.

Usage:
  python scripts/default_options/make_a3_set.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
DEFAULT_IN = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"
OUT_DIR = REPO / "outputs" / "default_options" / "inputs"

# The complete set of non-substantive option strings in the ladder
# substrate. Frozen on purpose: verify_a3_set.py sweeps a broad DK regex
# over every option and fails if anything outside this set matches, so a
# future substrate cannot silently widen or miss the category.
DK_OPTIONS = frozenset({"Don't know", "Do not know", "Refusal"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ladder-set", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "a3_dk_set.jsonl")
    args = ap.parse_args(argv)

    rows: list[dict] = []
    with open(args.ladder_set, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["example_id"].endswith("_ladder_informative_k024"):
                continue
            opts = r["option_sets"]["original"]
            if len(set(opts)) != len(opts):
                continue  # duplicate-option hygiene targets
            if not any(o in DK_OPTIONS for o in opts):
                continue  # no DK option: the cells would be identical
            rows.append(r)

    per_target: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "n_dk_truth": 0, "dk_options": None})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for src in sorted(rows, key=lambda r: r["example_id"]):
            eid = src["example_id"]
            opts = src["option_sets"]["original"]
            dk_here = [o for o in opts if o in DK_OPTIONS]
            substantive = [o for o in opts if o not in DK_OPTIONS]
            truth = src["ground_truth"]
            truth_is_dk = truth in DK_OPTIONS

            t = per_target[(src["survey"], src["target_code"])]
            t["n"] += 1
            t["n_dk_truth"] += int(truth_is_dk)
            t["dk_options"] = dk_here

            common = {
                "base_id": eid,
                "survey": src["survey"], "target_code": src["target_code"],
                "id": src.get("id"), "country": src.get("country"),
                "target_question": src["target_question"],
                "questions": src["questions"],
                "ground_truth": truth,
                "dk_options": dk_here,
                "truth_is_dk": truth_is_dk,
            }
            # Cells adjacent: LIMIT keeps whole pairs, smoke spans both.
            fh.write(json.dumps({
                "example_id": f"{eid}_dkpresent", "arm_label": "dk_present",
                "option_sets": {"original": opts},
                "ground_truth_index": src["ground_truth_index"],
                **common}, ensure_ascii=False) + "\n")
            fh.write(json.dumps({
                "example_id": f"{eid}_dkabsent", "arm_label": "dk_absent",
                "option_sets": {"original": substantive},
                "ground_truth_index": (None if truth_is_dk
                                       else substantive.index(truth)),
                **common}, ensure_ascii=False) + "\n")
            n += 2

    meta = {
        "dk_option_strings": sorted(DK_OPTIONS),
        "substrate": "ladder_readout k024 informative, all pairs, "
                     "DK-carrying clean targets only",
        "n_pairs": n // 2, "n_instances": n,
        "n_targets": len(per_target),
        "targets": {
            f"{s}|{t}": {
                "n": v["n"], "n_dk_truth": v["n_dk_truth"],
                "human_dk_share": round(v["n_dk_truth"] / v["n"], 6),
                "dk_options": v["dk_options"],
            }
            for (s, t), v in sorted(per_target.items())
        },
    }
    (args.out.parent / "a3_dk_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    overall = sum(v["n_dk_truth"] for v in per_target.values()) / (n // 2)
    print(f"{n // 2} pairs x 2 cells = {n} instances, "
          f"{len(per_target)} targets, human DK-truth share "
          f"{overall:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
