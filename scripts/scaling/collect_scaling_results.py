#!/usr/bin/env python
"""Turn the cluster's JSONL shards into the CSV schema the analysis reads.

score_scaling.py writes one JSONL record per instance carrying only
example_id, the scores and profile_type. It does not carry survey,
respondent_id or target_code, because those are properties of the instance
rather than of the scoring. analyze_scaling.py needs all three: it merges
n_options on (survey, target_code) and builds the fixed pair set on
(survey, respondent_id, target_code).

So the join keys are recovered from the instance files that produced the
shards, keyed on example_id. They are NOT parsed out of example_id: 23 of the
36 Arab Barometer target codes contain an underscore, which is exactly the
corruption repair_ids.py exists to undo in the main results. The instance
files carry the fields separately and correctly, so we read them from there.

Validates before writing, because a silently truncated shard would show up as
a bend in the scaling curve rather than as an error:
  - every scored example_id is a known instance
  - ground_truth agrees with the instance's recorded answer
  - the predicted label is one of the offered options
  - coverage per level against the number of instances generated

Usage:
    python .../collect_scaling_results.py
    python .../collect_scaling_results.py --results <dir> --out <file.csv>
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "scaling_experiment"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load_instances() -> dict[str, dict]:
    """example_id -> the join keys and the recorded answer."""
    index: dict[str, dict] = {}
    for f in sorted(SRC.glob("*_scaling_instances.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                index[r["example_id"]] = {
                    "survey": r["survey"],
                    "respondent_id": r["id"],
                    "target_code": r["target_code"],
                    "country": r.get("country"),
                    "profile_type": r["profile_type"],
                    "n_features": r["n_features"],
                    "answer": r.get("answer"),
                    "options": r.get("options") or [],
                }
    if not index:
        raise SystemExit(f"no instance files under {SRC}")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=SRC / "results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.results / "scaling_results_all.csv")

    index = load_instances()
    generated = collections.Counter(v["n_features"] for v in index.values())
    print(f"{len(index):,} instances generated: "
          + ", ".join(f"k={k} {n:,}" for k, n in sorted(generated.items())))

    shards = sorted(args.results.glob("scaling_results_*.jsonl"))
    if not shards:
        raise SystemExit(
            f"no scaling_results_*.jsonl under {args.results}\n"
            "Pull them from the cluster first.")

    # Resume appends, so dedupe on example_id keeping the last write.
    records: dict[str, dict] = {}
    per_shard = {}
    for p in shards:
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                records[r["example_id"]] = r
                n += 1
        per_shard[p.name] = n
        print(f"  {p.name:<32} {n:>8,} lines")
    dupes = sum(per_shard.values()) - len(records)
    print(f"{len(records):,} unique scored instances"
          + (f"  ({dupes:,} duplicate lines collapsed)" if dupes else ""))

    problems = collections.Counter()
    rows = []
    for eid, r in records.items():
        meta = index.get(eid)
        if meta is None:
            problems["unknown example_id"] += 1
            continue
        if meta["answer"] != r.get("ground_truth"):
            problems["ground_truth disagrees with instance"] += 1
        if r.get("predicted") not in meta["options"]:
            problems["prediction not among options"] += 1
        rows.append({
            "example_id": eid,
            "survey": meta["survey"],
            "respondent_id": meta["respondent_id"],
            "target_code": meta["target_code"],
            "country": meta["country"],
            "profile_type": meta["profile_type"],
            "n_features": meta["n_features"],
            "ground_truth": r.get("ground_truth"),
            "predicted": r.get("predicted"),
            "correct": bool(r.get("correct")),
        })

    df = pd.DataFrame(rows)
    print()
    print("coverage against instances generated:")
    ok = True
    for k, n_gen in sorted(generated.items()):
        n_got = int((df["n_features"] == k).sum())
        pct = 100.0 * n_got / n_gen if n_gen else 0.0
        flag = "" if n_got == n_gen else "   <-- INCOMPLETE"
        print(f"  k={k:<3} {n_got:>8,} / {n_gen:>8,}  ({pct:5.1f}%){flag}")
        ok &= n_got == n_gen

    if problems:
        print()
        print("validation problems:")
        for what, n in problems.items():
            print(f"  {what}: {n:,}")
        ok = False

    print()
    print("raw accuracy by level (unnormalized, all scored instances):")
    for k, g in df.groupby("n_features"):
        print(f"  k={k:<3} {g['correct'].mean():.4f}  n={len(g):,}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print()
    print(f"wrote {out}  ({len(df):,} rows)")
    print("COLLECTION CLEAN" if ok else "COLLECTION INCOMPLETE OR INCONSISTENT")


if __name__ == "__main__":
    main()
