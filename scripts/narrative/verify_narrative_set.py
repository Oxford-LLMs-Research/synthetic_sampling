"""B3: pin every property of the assembled scoring set against its sources.

Standing rule (CLAUDE.md): reported numbers are verified by script, never by
eye. The set is the thing three scoring arms are read off, so the checks
cover the contrast's assumptions, not just row counts:

  structure   3 instances per base_id, arms {qa, narrative1, narrative2},
              unique example_ids, no excluded pair present
  pairing     the paired contrast needs the common fields (target question,
              options, ground truth) IDENTICAL across a base_id's 3 rows,
              otherwise narrative-vs-qa is not a within-pair comparison
  content     qa carries the 24 source q:a pairs and no prose; narrative arms
              carry prose and no q:a; the prose is byte-identical to the
              validated file it came from
  independence  the two drafts differ, and trigram Jaccard stays under the
              near-copy gate (this is the wording-variance ceiling's premise)
  renderable  build_prompt puts the narrative in the profile slot for the
              narrative arms and the q:a block for qa; ground truth resolves
              to a real option index in every row

Usage:
  python scripts/narrative/verify_narrative_set.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
IN_DIR = REPO / "outputs" / "narrative" / "inputs"

ARMS = ("qa", "narrative1", "narrative2")
COMMON = ("survey", "target_code", "target_question", "option_sets",
          "ground_truth", "ground_truth_index", "id", "country")
NEAR_COPY_MAX = 0.5
K = 24


def trigram_jaccard(a: str, b: str) -> float:
    def grams(s: str) -> set[str]:
        w = s.lower().split()
        return {" ".join(w[i:i + 3]) for i in range(max(0, len(w) - 2))}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", type=Path,
                    default=IN_DIR / "narrative_label_set.jsonl")
    ap.add_argument("--narratives", type=Path,
                    default=IN_DIR / "validated_narratives.jsonl")
    ap.add_argument("--register", type=Path,
                    default=IN_DIR / "validation_register.json")
    ap.add_argument("--tasks", type=Path,
                    default=IN_DIR / "narrative_tasks.jsonl")
    ap.add_argument("--ladder-set", type=Path,
                    default=OUTER / "outputs_recovered"
                    / "ladder_readout_set.jsonl")
    args = ap.parse_args(argv)

    checks: list[tuple[bool, str]] = []

    def ck(ok: bool, label: str) -> None:
        checks.append((bool(ok), label))

    rows = [json.loads(l) for l in
            open(args.set, encoding="utf-8") if l.strip()]
    reg = json.loads(args.register.read_text(encoding="utf-8"))
    drafts = {}
    with open(args.narratives, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            drafts[r["task_id"]] = r["narrative"].strip()
    tasks = {}
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks.setdefault(t["example_id"], {})[t["draft"]] = t

    by_base: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_base.setdefault(r["base_id"], {})[r["arm_label"]] = r

    n_pairs = reg["total_validated_pairs"]
    ck(len(rows) == 3 * n_pairs,
       f"row count {len(rows)} == 3 x {n_pairs} validated pairs")
    ck(len(by_base) == n_pairs, f"distinct base_ids {len(by_base)} == {n_pairs}")
    ck(len({r["example_id"] for r in rows}) == len(rows),
       "example_ids unique")
    ck(all(set(a) == set(ARMS) for a in by_base.values()),
       "every base_id carries exactly the 3 arms")
    ck(not (set(by_base) & set(reg["exclusions"])),
       f"none of the {len(reg['exclusions'])} excluded pairs present")
    ck(set(by_base) <= set(tasks),
       "every base_id is a pair of the 734-pair substrate")

    # Pairing: the within-pair contrast requires identical common fields.
    bad_common = [b for b, a in by_base.items()
                  if any(a[arm].get(f) != a["qa"].get(f)
                         for arm in ARMS for f in COMMON)]
    ck(not bad_common,
       f"common fields identical across arms within every base_id "
       f"({len(bad_common)} mismatched)")

    # Content shape per arm.
    qa = [a["qa"] for a in by_base.values()]
    ck(all(len(r["questions"]) == K for r in qa),
       f"qa arm carries {K} q:a pairs in every row")
    ck(all(not r.get("profile_text") for r in qa),
       "qa arm carries no prose profile")
    narr = [(b, d, by_base[b][f"narrative{d}"])
            for b in by_base for d in (1, 2)]
    ck(all(r.get("profile_text", "").strip() for _, _, r in narr),
       "narrative arms all carry non-empty prose")
    ck(all(not r["questions"] for _, _, r in narr),
       "narrative arms carry no q:a block (prose must be the only profile)")
    ck(all(r["profile_text"] == drafts[f"{b}#d{d}"]
           for b, d, r in narr),
       "prose byte-identical to validated_narratives.jsonl")
    ck(all(r["questions"] == tasks[b][1]["questions"] for b, r in
           ((b, a["qa"]) for b, a in by_base.items())),
       "qa questions identical to the source task profile")

    # Independence: the wording-variance ceiling depends on it.
    jac = [(b, trigram_jaccard(by_base[b]["narrative1"]["profile_text"],
                               by_base[b]["narrative2"]["profile_text"]))
           for b in by_base]
    over = [(b, j) for b, j in jac if j > NEAR_COPY_MAX]
    ck(all(by_base[b]["narrative1"]["profile_text"]
           != by_base[b]["narrative2"]["profile_text"] for b in by_base),
       "the two drafts of a pair are never identical")
    ck(not over,
       f"near-copy gate: all pairs at trigram Jaccard <= {NEAR_COPY_MAX} "
       f"({len(over)} over)")

    # Ground truth resolves, and the readout has options to number.
    def opts(r: dict) -> list[str]:
        return list(r["option_sets"]["original"])

    ck(all(0 <= r["ground_truth_index"] < len(opts(r)) for r in rows),
       "ground_truth_index in range of the option set in every row")
    ck(all(opts(r)[r["ground_truth_index"]] == r["ground_truth"]
           for r in rows),
       "ground_truth == option_sets['original'][ground_truth_index]")
    ck(all(len(opts(r)) >= 2 for r in rows),
       "every row has >= 2 options for label_num")

    # Renderable: the prompt builder must actually use the prose.
    sys.path.insert(0, str(REPO / "src"))
    from synthetic_sampling.scoring.prompts import build_prompt

    sample = sorted(by_base)[0]
    p_qa = build_prompt(by_base[sample]["qa"], opts(by_base[sample]["qa"]),
                        "label_num")
    p_n1 = build_prompt(by_base[sample]["narrative1"],
                        opts(by_base[sample]["narrative1"]), "label_num")
    ck(by_base[sample]["narrative1"]["profile_text"] in p_n1
       and "\nQ: " not in p_n1.split("Question:")[0],
       "narrative arm renders prose in the profile slot, no q:a leakage")
    ck("\nQ: " in p_qa and "A: " in p_qa,
       "qa arm renders the q:a block")
    ck(p_qa.endswith("Answer: ") and p_n1.endswith("Answer: "),
       "label_num prompts end with the trailing-space readout position")

    lens = [len(build_prompt(r, opts(r), "label_num")) for r in rows]
    ck(max(lens) < 40000,
       f"longest label_num prompt {max(lens)} chars (max_model_len 16384 "
       f"tokens ~ 65k chars)")

    n_qa = sum(len(build_prompt(a["qa"], opts(a["qa"]), "label_num"))
               for a in by_base.values()) / len(by_base)
    n_pr = sum(len(r["profile_text"]) for _, _, r in narr) / len(narr)
    print(f"\nset: {len(rows)} instances, {len(by_base)} pairs x 3 arms")
    print(f"  mean qa prompt {n_qa:.0f} chars, mean prose profile "
          f"{n_pr:.0f} chars, longest prompt {max(lens)} chars")
    print(f"  draft-pair trigram Jaccard: mean "
          f"{sum(j for _, j in jac) / len(jac):.3f}, "
          f"max {max(j for _, j in jac):.3f}")
    surveys: dict[str, int] = {}
    for b in by_base:
        surveys[by_base[b]["qa"]["survey"]] = surveys.get(
            by_base[b]["qa"]["survey"], 0) + 1
    print("  pairs by survey: " + ", ".join(
        f"{k} {v}" for k, v in sorted(surveys.items())))
    targets = {by_base[b]["qa"]["target_code"] for b in by_base}
    print(f"  distinct target questions: {len(targets)}")

    print()
    fails = 0
    for ok, label in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        fails += not ok
    print(f"\n{len(checks) - fails}/{len(checks)} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
