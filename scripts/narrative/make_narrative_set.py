"""B3 stage 4: assemble the scoring set from validated narratives.

For every pair whose TWO drafts passed both the automatic gates and the
round-trip coverage gate, emit three runner instances sharing base_id (the
source ladder example_id), so all contrasts are paired within one serving:

  <eid>_narrqa   the plain k=24 q:a profile (same-serving baseline)
  <eid>_narr1    profile_text = draft 1 narrative
  <eid>_narr2    profile_text = draft 2 narrative

narr1-vs-narr2 scoring agreement is the wording-variance ceiling;
narrative-vs-qa is the format effect, read against that ceiling. Pairs with
any failed draft are excluded and counted (exclusions are data).

Two input modes:
  --roundtrip CSV  per-set use: filter a raw narratives file by its own
                   round-trip result (one tranche at a time).
  (omitted)        substrate use: --narratives is already the consolidated
                   validated file, so every draft in it has passed. Pass
                   --register to make the assembly self-checking — the pair
                   count and the excluded set must match the register, or
                   the script fails rather than quietly shipping a short set.

Usage:
  python scripts/narrative/make_narrative_set.py \
      --narratives outputs/narrative/inputs/validated_narratives.jsonl \
      --register   outputs/narrative/inputs/validation_register.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
IN_DIR = REPO / "outputs" / "narrative" / "inputs"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--narratives", type=Path, required=True)
    ap.add_argument("--roundtrip", type=Path, default=None,
                    help="per-set mode; omit when --narratives is the "
                         "consolidated validated file")
    ap.add_argument("--register", type=Path, default=None,
                    help="validation_register.json; cross-checks the "
                         "assembled pair and exclusion sets")
    ap.add_argument("--tasks", type=Path,
                    default=IN_DIR / "narrative_tasks.jsonl")
    ap.add_argument("--ladder-set", type=Path, default=LADDER_SET)
    ap.add_argument("--out", type=Path,
                    default=IN_DIR / "narrative_label_set.jsonl")
    args = ap.parse_args(argv)

    drafts: dict[str, str] = {}
    with open(args.narratives, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            drafts[r["task_id"]] = r["narrative"].strip()

    if args.roundtrip is not None:
        passed = set()
        with open(args.roundtrip, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["pass"] in ("True", "true", "1"):
                    passed.add(row["task_id"])
    else:
        # Consolidated mode: the file IS the pass list.
        passed = set(drafts)

    tasks_by_pair: dict[str, dict] = {}
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks_by_pair.setdefault(t["example_id"], {})[t["draft"]] = t

    source = {}
    with open(args.ladder_set, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in tasks_by_pair:
                source[r["example_id"]] = r

    n_pairs = 0
    excluded: list[str] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for eid in sorted(tasks_by_pair):
            ids = {d: f"{eid}#d{d}" for d in (1, 2)}
            if not all(ids[d] in passed and ids[d] in drafts
                       for d in (1, 2)):
                excluded.append(eid)
                continue
            src = source[eid]
            common = {
                "base_id": eid,
                "survey": src["survey"], "target_code": src["target_code"],
                "id": src.get("id"), "country": src.get("country"),
                "target_question": src["target_question"],
                "option_sets": src["option_sets"],
                "ground_truth": src["ground_truth"],
                "ground_truth_index": src["ground_truth_index"],
            }
            fh.write(json.dumps({
                "example_id": f"{eid}_narrqa", "arm_label": "qa",
                "questions": src["questions"], **common},
                ensure_ascii=False) + "\n")
            for d in (1, 2):
                fh.write(json.dumps({
                    "example_id": f"{eid}_narr{d}",
                    "arm_label": f"narrative{d}", "questions": {},
                    "profile_text": drafts[ids[d]], **common},
                    ensure_ascii=False) + "\n")
            n_pairs += 1

    print(f"{n_pairs} pairs assembled (3 instances each, "
          f"{3 * n_pairs} total), {len(excluded)} pairs excluded "
          f"(failed drafts) -> {args.out}")

    if args.register is not None:
        reg = json.loads(args.register.read_text(encoding="utf-8"))
        want_pairs = reg["total_validated_pairs"]
        want_excl = set(reg["exclusions"])
        problems = []
        if n_pairs != want_pairs:
            problems.append(f"assembled {n_pairs} pairs, register says "
                            f"{want_pairs}")
        if set(excluded) != want_excl:
            missing = sorted(want_excl - set(excluded))
            extra = sorted(set(excluded) - want_excl)
            problems.append(f"exclusion set differs: {len(missing)} in "
                            f"register but assembled ({missing[:3]}), "
                            f"{len(extra)} assembled-out but not in register "
                            f"({extra[:3]})")
        if problems:
            for p in problems:
                print(f"REGISTER MISMATCH: {p}")
            return 1
        print(f"register check OK: {n_pairs} validated, "
              f"{len(excluded)} excluded, exclusion ids match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
