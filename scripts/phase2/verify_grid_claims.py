"""Independent recomputation of grid_claims.py headline numbers from raw JSONL.

    python scripts/phase2/verify_grid_claims.py ../outputs_recovered/phase2_grid_r0-50/qwen_qwen3-32b \
        --instances outputs/phase2/inputs/instances_r0-50_short.jsonl \
        --claims ../analysis/phase2/grid_claims/by_serving.csv

Fresh code, not grid_claims.py's: one per-instance frame, pandas groupby, the
definitions as written in grid_claims.py's docstring. Checks accuracy, question-
averaged normalized accuracy, the pooled dissenter split and its N0 penalty, the
hard entropy ratio, and the mean tally and leave-one-out TV over cells with
n >= 30. Exits 1 on any mismatch beyond 1e-9.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

TOL = 1e-9


def frame(folder: str, instances: str) -> pd.DataFrame:
    meta = {}
    with open(instances, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            meta[d["example_id"]] = (d["survey"], d["target_code"], "|".join(d["options"]),
                                     len(d["options"]), d["country"])
    recs = []
    for path in sorted(glob.glob(os.path.join(folder, "*_shard*of*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rec = r["results"].get("original|label_num") or r["results"].get("original|xgb")
                s, t, opts, m, c = meta[r["example_id"]]
                recs.append((s, t, opts, m, c, r["ground_truth_index"], rec["predicted_index"]))
    return pd.DataFrame(recs, columns=["survey", "target", "opts", "M", "country", "truth", "pred"])


def qmean(df: pd.DataFrame, col: str) -> float:
    """n-weighted within question over option-set groups, then across questions."""
    d = df.dropna(subset=[col])
    w = d["n"]
    per_q = (d[col] * w).groupby([d["survey"], d["target"]]).sum() / w.groupby(
        [d["survey"], d["target"]]).sum()
    return float(per_q.mean())


def entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--claims", required=True)
    args = ap.parse_args()
    name = os.path.basename(args.folder.rstrip("/\\"))
    df = frame(args.folder, args.instances)
    ref = pd.read_csv(args.claims).set_index("serving").loc[name]

    got = {"acc": float((df.truth == df.pred).mean())}
    rows = []
    for (s, t, opts), g in df.groupby(["survey", "target", "opts"]):
        M = int(g.M.iloc[0])
        n = len(g)
        tc = g.truth.value_counts().reindex(range(M), fill_value=0)
        pc = g.pred.value_counts().reindex(range(M), fill_value=0)
        nrm = lambda a: (a - 1 / M) / (1 - 1 / M)
        mode = int(np.argmax(tc.to_numpy()))
        row = {"survey": s, "target": t, "n": n, "norm": nrm((g.truth == g.pred).mean())}
        mod, dis = g[g.truth == mode], g[g.truth != mode]
        if len(mod) and len(dis):
            p_t, p_y = tc / n, pc / n
            row["pen"] = nrm((mod.truth == mod.pred).mean()) - nrm((dis.truth == dis.pred).mean())
            n0_mod = p_y[mode]
            n0_dis = sum(p_y[o] * p_t[o] for o in range(M) if o != mode) / (1 - p_t[mode])
            row["pen_n0"] = nrm(n0_mod) - nrm(n0_dis)
        if entropy(tc) > 0:
            row["er_hard"] = entropy(pc) / entropy(tc)
        rows.append(row)
    gdf = pd.DataFrame(rows)
    got["norm"] = qmean(gdf, "norm")
    sub = gdf.dropna(subset=["pen"])
    got["pen_obs"] = qmean(sub, "pen")
    got["pen_n0"] = qmean(sub, "pen_n0")
    got["er_hard"] = qmean(gdf, "er_hard")

    tvs, loos = [], []
    for (s, t, opts), g in df.groupby(["survey", "target", "opts"]):
        M = int(g.M.iloc[0])
        tot = g.truth.value_counts().reindex(range(M), fill_value=0)
        n_countries = g.country.nunique()
        for c, cell in g.groupby("country"):
            if len(cell) < 30 or n_countries < 2:
                continue
            q = cell.truth.value_counts().reindex(range(M), fill_value=0) / len(cell)
            y = cell.pred.value_counts().reindex(range(M), fill_value=0) / len(cell)
            rest = tot - cell.truth.value_counts().reindex(range(M), fill_value=0)
            if rest.sum() == 0:
                continue
            tvs.append(0.5 * (y - q).abs().sum())
            loos.append(0.5 * (rest / rest.sum() - q).abs().sum())
    got["tv_tally_mean"] = float(np.mean(tvs)) if tvs else float("nan")
    got["tv_loo_mean"] = float(np.mean(loos)) if loos else float("nan")

    bad = 0
    for k, v in got.items():
        r = float(ref[k])
        # a single shard has no cell with n >= 30, so both sides are NaN there
        ok = (np.isnan(v) and np.isnan(r)) or abs(v - r) <= TOL
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {k:16s} independent {v:.9f}  grid_claims {r:.9f}")
    print(f"{name}: {len(got) - bad}/{len(got)} match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
