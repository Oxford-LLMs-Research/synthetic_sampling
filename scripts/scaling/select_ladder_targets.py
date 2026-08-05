#!/usr/bin/env python
r"""Pick a small, balanced set of targets for the by-feature ladder.

The ladder is a mechanical look at how the model's belief moves as features
accumulate, not a test of a hypothesis, so the target set should span the design
space rather than be chosen for effect size. Two dimensions matter:

  theme          the substantive kind of question, from target_topic_tag
  option count   how many distinct answers are offered, which interacts with
                 the normalisation and with the option-length preference the
                 scorer turns out to have (see checks/length_bias.py)

XGBoost headroom is deliberately NOT a selection criterion. Choosing targets
where tabular features are known to predict well would tilt the informative arm
toward finding something. It is reported for every candidate so its distribution
in the chosen set is visible, but it does not drive the pick.

Every theme is represented. Crossing theme with option count would either
restrict the design to the handful of themes that span all four buckets, which
silently drops most of the substantive range, or demand far more targets than a
mechanical look needs. So themes are exhaustive and option count is balanced
ACROSS the set instead of within each theme.

That is an assignment problem rather than a filter. Themes are processed most
constrained first, meaning those offering the fewest distinct buckets choose
before those that could fill any of them, and each theme takes the bucket that
is currently least represented. Ties go to the survey least represented so far,
then to the largest pool of respondents.

Candidates must survive one practical filter: presence in the k=96 scaling
instances, which supply the ladder's feature universe.

    python .../select_ladder_targets.py --per-theme 1
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO.parent / "analysis"
MAIN = REPO / "outputs" / "main_data_smaller_20_jan_26" / "main_data"
SCALE = REPO / "outputs" / "scaling_experiment"
BUCKETS = [(0, 2, "2"), (2, 4, "3-4"), (4, 6, "5-6"), (6, 999, "7+")]
MIN_LADDER_POOL = 60      # respondents with a k=96 profile, the ladder universe


def bucket(n: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo < n <= hi:
            return name
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-theme", type=int, default=1)
    ap.add_argument("--out", type=Path, default=SCALE / "ladder_targets.csv")
    args = ap.parse_args()

    meta = {}
    for f in sorted(MAIN.glob("*_instances.jsonl")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                k = (d["survey"], d["target_code"])
                if k not in meta:
                    meta[k] = {"survey": k[0], "target_code": k[1],
                               "topic": d.get("target_topic_tag")}
    t = pd.DataFrame(meta.values()).dropna(subset=["topic"])

    q = (pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc.csv")
         .groupby(["survey", "target_code"])["n_options"].first().reset_index())
    t = t.merge(q, on=["survey", "target_code"])
    t["opts"] = t["n_options"].map(bucket)

    x = pd.read_csv(ANALYSIS / "xgboost_baseline" / "results_merged.csv") \
          .query("profile_type == 's6m4'")
    x["headroom"] = x["xgb_norm_acc"] - x["majority_norm_acc"]
    t = t.merge(x[["survey", "target_code", "headroom", "n_respondents"]],
                on=["survey", "target_code"], how="left")

    # The ladder draws its feature universe from the k=96 profiles, so a target
    # with few of those cannot supply respondents however good it looks.
    pool = collections.Counter()
    for f in sorted(SCALE.glob("*_scaling_instances.jsonl")):
        survey = f.name.split("_scaling_instances")[0]
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                if d["profile_type"] == "s6m4x96":
                    pool[(survey, d["target_code"])] += 1
    t["ladder_pool"] = [pool.get((s, c), 0) for s, c in
                        zip(t["survey"], t["target_code"])]
    t = t[t["ladder_pool"] >= MIN_LADDER_POOL]
    print(f"{len(t)} targets with a topic, an option count and at least "
          f"{MIN_LADDER_POOL} k=96 profiles")

    names = [b[2] for b in BUCKETS]
    themes = sorted(t["topic"].unique())
    print(f"\n{len(themes)} themes represented, taking {args.per_theme} target(s) "
          f"from each")

    # Most constrained first: a theme offering one bucket must take it, so it
    # should choose before a theme that could fill any of them.
    order = sorted(themes, key=lambda th: (t[t["topic"] == th]["opts"].nunique(),
                                           len(t[t["topic"] == th])))
    n_bucket: collections.Counter = collections.Counter()
    n_survey: collections.Counter = collections.Counter()
    taken, rows = set(), []
    for th in order:
        sub = t[t["topic"] == th]
        for _ in range(args.per_theme):
            avail = sub[~sub.index.isin(taken)]
            if avail.empty:
                break
            # Least represented bucket this theme can supply, then least
            # represented survey, then the largest respondent pool.
            avail = avail.assign(
                _b=[n_bucket[b] for b in avail["opts"]],
                _s=[n_survey[s] for s in avail["survey"]])
            avail = avail.sort_values(["_b", "_s", "ladder_pool"],
                                      ascending=[True, True, False])
            pick = avail.iloc[0]
            taken.add(pick.name)
            n_bucket[pick["opts"]] += 1
            n_survey[pick["survey"]] += 1
            rows.append(pick)
    sel = pd.DataFrame(rows).reset_index(drop=True)

    print(f"\n{len(sel)} targets selected\n")
    cols = ["survey", "target_code", "topic", "opts", "n_options",
            "ladder_pool", "headroom"]
    print(sel[cols].sort_values(["opts", "topic"]).to_string(index=False))

    print("\noption-bucket balance across the set:")
    for name in names:
        print(f"  {name:<5}{int((sel['opts'] == name).sum()):>4}")
    print("\nsurveys:")
    for s, n in sel["survey"].value_counts().sort_index().items():
        print(f"  {s:<18}{n:>4}")
    print(f"\nheadroom in the chosen set: median {sel['headroom'].median():+.3f}, "
          f"{(sel['headroom'] > 0).sum()} of {len(sel)} positive")
    sel[["survey", "target_code"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
