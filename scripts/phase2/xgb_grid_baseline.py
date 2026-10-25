"""XGBoost on the Phase 2 grid's own profiles, with out-of-fold predictions kept.

    python scripts/phase2/xgb_grid_baseline.py \
        outputs/phase2/inputs/instances_r0-50_short.jsonl \
        --out outputs/phase2/results/xgb_grid_r0-50 --regime groupkfold

Prompt parity: each instance's features are exactly the 36 profile items its
prompt carried, every other item missing (XGBoost's native NaN direction), as
in run 1's baseline. One model per (survey, target, option set). Hyperparameters
are run 1's, fixed in advance (supplement, XGBoost Baseline Details).

Regimes:
- groupkfold: GroupKFold on country, so a respondent is predicted by a model
  that never saw their country (audit F2). Single-country groups are skipped.
- stratified: StratifiedKFold(5, shuffle, seed 42), run 1's scheme.

Writes `<out>/xgb_<regime>/xgb_<regime>_shard0of1.jsonl`, one row per scored
instance in the grid's result format: `results["original|xgb"]` holds the
out-of-fold log-probabilities as `scores`, so `grid_claims.py` reads it like a
serving. Training uses the grid's own respondents only (floor 50 per country),
so this is a same-sample floor on the supervised ceiling, not the ceiling.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, StratifiedKFold

PARAMS = {
    "objective": "multi:softprob",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "mlogloss",
    "seed": 42,
    "tree_method": "hist",
}
ROUNDS = 200
ARM = "original|xgb"


def index_groups(path: str) -> dict:
    """Byte offsets of each (survey, target, option set) group's lines."""
    groups = collections.defaultdict(list)
    with open(path, "rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            d = json.loads(line)
            groups[(d["survey"], d["target_code"], tuple(d["options"]))].append(pos)
    return groups


def load_group(path: str, offsets: list) -> list:
    rows = []
    with open(path, "rb") as fh:
        for pos in offsets:
            fh.seek(pos)
            rows.append(json.loads(fh.readline()))
    return rows


def design(rows: list) -> pd.DataFrame:
    """Categorical feature frame over the union of the group's profile items."""
    cols = sorted({q for r in rows for q in r["questions"]})
    pos = {q: i for i, q in enumerate(cols)}
    data = [[None] * len(cols) for _ in rows]
    for i, r in enumerate(rows):
        for q, a in r["questions"].items():
            data[i][pos[q]] = a
    frame = pd.DataFrame(data, columns=[f"f{i}" for i in range(len(cols))])
    return frame.astype("category")


def folds(regime: str, y: np.ndarray, countries: np.ndarray):
    """Fold indices, or a string saying why the group is skipped."""
    if regime == "groupkfold":
        k = min(5, len(set(countries)))
        if k < 2:
            return "single country"
        return list(GroupKFold(n_splits=k).split(y, y, countries))
    # run 1's rule: cells under 20 respondents or with one class are skipped;
    # sparse classes make StratifiedKFold warn, and it cannot run when no class
    # reaches n_splits, so the split count drops to the largest class size
    counts = np.bincount(y)
    if len(y) < 20 or (counts > 0).sum() < 2:
        return "under 20 rows or one class"
    k = min(5, int(counts.max()))
    return list(StratifiedKFold(k, shuffle=True, random_state=42).split(y, y))


def fit_group(rows: list, regime: str, nthread: int):
    options = list(rows[0]["options"])
    k = len(options)
    y = np.array([r["ground_truth_index"] for r in rows], dtype=int)
    countries = np.array([r["country"] for r in rows])
    split = folds(regime, y, countries)
    if isinstance(split, str):
        return None, {"skipped": split}
    X = design(rows)
    probs = np.full((len(rows), k), np.nan)
    params = dict(PARAMS, num_class=k, nthread=nthread)
    for train, test in split:
        dtrain = xgb.DMatrix(X.iloc[train], label=y[train], enable_categorical=True)
        booster = xgb.train(params, dtrain, num_boost_round=ROUNDS)
        dtest = xgb.DMatrix(X.iloc[test], enable_categorical=True)
        probs[test] = booster.predict(dtest).reshape(len(test), k)
    meta = {"rows": len(rows), "countries": len(set(countries)),
            "features": X.shape[1], "folds": len(split)}
    return probs, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instances")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regime", choices=["groupkfold", "stratified"], required=True)
    ap.add_argument("--nthread", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N groups only (smoke)")
    args = ap.parse_args()

    name = f"xgb_{args.regime}"
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_shard0of1.jsonl")
    log_path = os.path.join(out_dir, f"{name}_groups.csv")

    t0 = time.time()
    groups = index_groups(args.instances)
    keys = sorted(groups)
    if args.limit:
        keys = keys[: args.limit]
    print(f"{len(groups)} groups indexed in {time.time() - t0:.0f}s; fitting {len(keys)}",
          flush=True)

    log = []
    with open(out_path, "w", encoding="utf-8") as out:
        for i, key in enumerate(keys):
            rows = load_group(args.instances, groups[key])
            probs, meta = fit_group(rows, args.regime, args.nthread)
            log.append({"survey": key[0], "target_code": key[1],
                        "n_options": len(key[2]), **meta})
            if probs is None:
                continue
            for r, p in zip(rows, probs):
                pred = int(np.argmax(p))
                scores = {o: (math.log(v) if v > 0 else float("-inf"))
                          for o, v in zip(r["options"], p)}
                out.write(json.dumps({
                    "example_id": r["example_id"],
                    "base_id": r["base_id"],
                    "survey": r["survey"],
                    "target_code": r["target_code"],
                    "ground_truth_index": r["ground_truth_index"],
                    "results": {ARM: {"scores": scores, "predicted_index": pred,
                                      "predicted": r["options"][pred]}},
                }) + "\n")
            if (i + 1) % 20 == 0 or i + 1 == len(keys):
                print(f"  {i + 1}/{len(keys)} groups, {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(log).to_csv(log_path, index=False)
    skipped = sum(1 for r in log if r.get("skipped"))
    print(f"wrote {out_path} ({len(log) - skipped} groups fitted, {skipped} skipped)")


if __name__ == "__main__":
    main()
