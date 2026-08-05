#!/usr/bin/env python
r"""Sample the paper's OWN instances for the readout experiment.

The question is whether reading a label token recovers person-level signal that
echo scoring destroys. Answering it needs many respondents per question, because
discrimination AUC and prior-corrected accuracy are both computed within a
question across respondents. The surface-form set has about nine, so it cannot;
the main grid has hundreds.

Sampling from the paper's own s6m4 instances rather than building new ones has
three advantages. The profiles are exactly the ones the paper scored, so nothing
about profile construction differs. The stored SGLang-era results cover the same
example_ids, so the freshly scored echo arm can be checked against them, which
both validates the harness and measures the serving shift on this exact sample.
And it costs no generation step.

Scores are NOT reused. Comparing stored echo against freshly run labels would
confound the readout with the serving stack, and that confound is large: the
same model on the same instances under a different stack agrees on only 64.4% of
predictions (checks/control_check.py). Every arm is re-scored in one run.

Questions are stratified by survey and by the number of answer options, since
option count drives both the chance baseline and the size of any label bias.

    python .../sample_readout_set.py --questions 48 --respondents 100
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
# The instances that were ACTUALLY scored. outputs/main_data_run_19_jan_26.zip
# is a different, earlier sample: it joins to only 133 of the 267 questions, and
# using it would have produced a silently non-representative subset. This
# directory joins at 100%.
INSTANCES = REPO / "outputs" / "main_data_smaller_20_jan_26" / "main_data"
RESULTS = REPO.parent / "results"
OUT = REPO / "outputs" / "scaling_experiment" / "readout_set.jsonl"


def scored_ids(model: str) -> set[str]:
    """example_ids the stored run actually scored, so the control is available."""
    ids = set()
    d = RESULTS / model
    for f in sorted(d.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                ids.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=48)
    ap.add_argument("--respondents", type=int, default=100)
    ap.add_argument("--level", default="s6m4")
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    have = scored_ids(args.model)
    print(f"{len(have):,} example_ids scored in the stored {args.model} run")

    byq: dict[tuple, list] = collections.defaultdict(list)
    seen_level = joined = 0
    for path in sorted(INSTANCES.glob("*_instances.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("profile_type") != args.level:
                    continue
                seen_level += 1
                if r["example_id"] not in have:
                    continue
                joined += 1
                # A labelled menu cannot show the same wording twice, and
                # echo scoring already collapses repeats because it keys
                # scores by option string. So de-duplicate rather than drop
                # the question: excluding every question with a repeated
                # label would bias the sample toward the ones where the two
                # readouts cannot differ on that account, and repeats are
                # common enough to cost half the grid.
                seen, opts = set(), []
                for o in r.get("options") or []:
                    if isinstance(o, str) and o.strip() and o not in seen:
                        seen.add(o)
                        opts.append(o)
                if len(opts) < 2 or r.get("answer") not in opts:
                    continue
                r["_opts"] = opts
                byq[(r["survey"], r["target_code"])].append(r)
    rate = joined / max(seen_level, 1)
    print(f"{seen_level:,} instances at {args.level}, {joined:,} join to the "
          f"stored run ({rate:.1%})")
    if rate < 0.98:
        raise SystemExit(
            f"only {rate:.1%} of instances join to the stored results. This is "
            "the wrong instance directory; the sample would be a silently "
            "non-representative subset of questions.")
    print(f"{len(byq)} questions at {args.level} with stored scores")

    eligible = {k: v for k, v in byq.items() if len(v) >= args.respondents}
    print(f"{len(eligible)} have at least {args.respondents} respondents")

    # Stratify by survey and option count so neither the chance baseline nor any
    # label bias is confounded with which questions were drawn.
    rng = random.Random(args.seed)
    strata: dict[tuple, list] = collections.defaultdict(list)
    for k, v in eligible.items():
        strata[(k[0], min(len(v[0]["_opts"]), 8))].append(k)
    picked: list[tuple] = []
    keys = sorted(strata)
    while len(picked) < args.questions and keys:
        for s in list(keys):
            if len(picked) >= args.questions:
                break
            pool = [q for q in strata[s] if q not in picked]
            if not pool:
                keys.remove(s)
                continue
            picked.append(rng.choice(pool))

    out, per_survey, per_M = [], collections.Counter(), collections.Counter()
    for q in picked:
        rows = eligible[q]
        take = rng.sample(rows, args.respondents)
        per_survey[q[0]] += 1
        per_M[len(rows[0]["_opts"])] += 1
        for r in take:
            out.append({
                "example_id": r["example_id"], "base_id": r.get("base_id"),
                "survey": r["survey"], "target_code": r["target_code"],
                "id": r.get("id"), "country": r.get("country"),
                "questions": r["questions"],
                "target_question": r["target_question"],
                "ground_truth": r["answer"],
                "ground_truth_index": r["_opts"].index(r["answer"]),
                # score_formats.py reads option_sets; the readout experiment has
                # one set, and the scorer adds its own replicate.
                "option_sets": {"original": list(r["_opts"])},
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_opt = sum(len(r["option_sets"]["original"]) for r in out) / max(len(out), 1)
    print(f"\n{len(out):,} instances over {len(picked)} questions "
          f"x {args.respondents} respondents")
    print(f"  surveys: {dict(per_survey)}")
    print(f"  option counts: {dict(sorted(per_M.items()))}")
    print(f"  mean options {n_opt:.1f}; two option sets (original + replicate)")
    print(f"  approx requests: {len(out) * 2 * (4 * n_opt + 2):,.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
