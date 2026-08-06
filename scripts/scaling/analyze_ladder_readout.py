#!/usr/bin/env python
r"""Does the ceiling rise under informative features once the readout is fixed?

The ladder-x-elicitation results: 1,504 pairs, rungs k=0/1/8/24/48/96 + full,
informative and random orderings, arms echo_plain / echo_listed / label_num /
echo_ctxfree, one serving. This reads off:

  the curve        raw and normalized accuracy per (elicitation, ordering, k),
                   clustered on the 30 targets
  the gate         the informative-vs-random gap by rung under label_num: the
                   quantity that decides whether "individual fidelity fails"
                   was scoped to random features or is the model's ceiling
  the instrument   the slope under label_num against the slope under
                   echo_plain: how much of the published ladder slope was the
                   scoring rule
  pmi_free         echo_plain minus ctxfree per rung (the corrected
                   hidden-options rule, per rung)
  contrastive      each arm's score(k) minus its own score(k=0), the within-
                   design correction, cross-checking the 5 Aug re-read
  AUC              per (elicitation, k), option-aligned, modal set per target
  replicate        agreement per arm, the ceiling for everything above

    python analyze_ladder_readout.py --results <ladder_readout_results_*.jsonl>
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_readout import weighted_auc                          # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
OUT = REPO.parent / "analysis" / "ladder"

ELICITS = ("echo_plain", "echo_listed", "label_num", "pmi_free",
           "contrast_echo", "contrast_label")
SCORED = ("echo_plain", "echo_listed", "label_num", "echo_ctxfree")


def boot_targets(per_target: pd.Series, n: int = 2000, seed: int = 42):
    """CI for the mean of per-target values, resampling targets."""
    v = per_target.to_numpy()
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=SCALE / "ladder_readout_set.jsonl")
    args = ap.parse_args()
    tag = args.results.stem.replace("ladder_readout_results_", "")

    meta = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r

    # ---- assemble: one row per (pair, ordering, k), score vectors per arm --
    raw = {}
    for line in open(args.results, encoding="utf-8"):
        r = json.loads(line)
        m = meta.get(r["example_id"])
        if m is None:
            continue
        opts = m["option_sets"]["original"]
        vecs, rep_vecs = {}, {}
        for arm in SCORED:
            for setname, store in (("original", vecs),
                                   ("original_replicate", rep_vecs)):
                d = r["results"].get(f"{setname}|{arm}")
                if d and "error" not in d and d.get("scores"):
                    store[arm] = np.array(
                        [d["scores"].get(o, float("-inf")) for o in opts], float)
        raw[r["example_id"]] = (m, vecs, rep_vecs)

    # k=0 vectors per pair, for the contrastive rules.
    k0 = {}
    for eid, (m, vecs, _) in raw.items():
        if m["n_features"] == 0:
            stem = eid.rsplit("_shared_k", 1)[0]
            k0[stem] = vecs

    rows = []
    for eid, (m, vecs, rep_vecs) in raw.items():
        stem = eid.split("_ladder_")[0] + "_ladder"
        opts = m["option_sets"]["original"]
        base = k0.get(stem, {})
        derived = dict(vecs)
        if "echo_plain" in vecs and "echo_ctxfree" in vecs:
            derived["pmi_free"] = vecs["echo_plain"] - vecs["echo_ctxfree"]
        if "echo_plain" in vecs and "echo_plain" in base and m["n_features"] > 0:
            derived["contrast_echo"] = vecs["echo_plain"] - base["echo_plain"]
        if "label_num" in vecs and "label_num" in base and m["n_features"] > 0:
            derived["contrast_label"] = vecs["label_num"] - base["label_num"]
        row = {"eid": eid, "stem": stem,
               "target": f"{m['survey']}|{m['target_code']}",
               "ordering": m["arm"], "k": m["n_features"],
               "rung": ("full" if m["arm"] == "shared" and m["n_features"] > 0
                        else str(m["n_features"])),
               "truth": m["ground_truth"], "opts": tuple(opts)}
        for name, v in derived.items():
            if np.isfinite(v).any():
                row[f"pred_{name}"] = opts[int(np.nanargmax(
                    np.where(np.isfinite(v), v, -np.inf)))]
                row[f"scores_{name}"] = v
        for name, v in rep_vecs.items():
            row[f"rep_pred_{name}"] = opts[int(np.nanargmax(
                np.where(np.isfinite(v), v, -np.inf)))]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"{len(df):,} scored instances, {df.stem.nunique():,} pairs, "
          f"{df.target.nunique()} targets, model file {args.results.name}\n")

    # M = distinct recorded answers per target among these pairs (paper rule).
    gt = df[df.k == 0][["stem", "target", "truth"]].drop_duplicates("stem")
    M_by_t = gt.groupby("target").truth.nunique()
    df["M"] = df.target.map(M_by_t)

    # ---- 1. the curves ----------------------------------------------------
    print("=== normalized accuracy by rung (targets with M >= 2, "
          "target-clustered) ===")
    curve_rows = []
    acc = df[df.M >= 2]
    for elicit in ELICITS:
        col = f"pred_{elicit}"
        if col not in acc.columns:
            continue
        g0 = acc[acc[col].notna()]
        print(f"--- {elicit} ---")
        hdr = "  k".ljust(8) + "".join(f"{o:>14}" for o in
                                       ("informative", "random", "shared"))
        print(hdr)
        for rung in ("0", "1", "8", "24", "48", "96", "full"):
            line = f"  {rung:<6}"
            for ordering in ("informative", "random", "shared"):
                g = g0[(g0.rung == rung) & (g0.ordering == ordering)]
                if g.empty:
                    line += f"{'-':>14}"
                    continue
                na_t = g.assign(na=((g[col] == g.truth).astype(float) - 1 / g.M)
                                / (1 - 1 / g.M)).groupby("target").na.mean()
                lo, hi = boot_targets(na_t)
                line += f"{na_t.mean():>14.3f}"
                curve_rows.append({"elicit": elicit, "ordering": ordering,
                                   "rung": rung, "norm_acc": na_t.mean(),
                                   "lo": lo, "hi": hi, "n": len(g),
                                   "n_targets": len(na_t)})
            print(line)
        print()

    # ---- 2. the gate: informative - random under each elicitation ---------
    print("=== the gate: informative minus random, paired within pair, "
          "by rung ===")
    for elicit in ("echo_plain", "label_num", "pmi_free"):
        col = f"pred_{elicit}"
        if col not in df.columns:
            continue
        print(f"--- {elicit} ---")
        for rung in ("1", "8", "24", "48", "96"):
            a = df[(df.rung == rung) & (df.ordering == "informative")
                   & df[col].notna()].set_index("stem")
            b = df[(df.rung == rung) & (df.ordering == "random")
                   & df[col].notna()].set_index("stem")
            common = a.index.intersection(b.index)
            if len(common) < 50:
                continue
            d = ((a.loc[common, col] == a.loc[common, "truth"]).astype(float)
                 - (b.loc[common, col] == b.loc[common, "truth"]).astype(float))
            per_t = d.groupby(a.loc[common, "target"]).mean()
            lo, hi = boot_targets(per_t)
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"  k={rung:<4} raw gap {per_t.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] {star}  n={len(common):,}")
        print()

    # ---- 3. AUC by rung (echo_plain and label_num, modal set per target) --
    print("=== AUC by rung (informative ordering; modal option set) ===")
    modal = {t: collections.Counter(g.opts).most_common(1)[0][0]
             for t, g in df.groupby("target")}
    print("  k".ljust(8) + "".join(f"{e:>14}" for e in ("echo_plain", "label_num")))
    for rung in ("0", "1", "8", "24", "48", "96", "full"):
        line = f"  {rung:<6}"
        for elicit in ("echo_plain", "label_num"):
            scol = f"scores_{elicit}"
            aucs = []
            sel = df[(df.rung == rung)
                     & (df.ordering.isin(["informative", "shared"]))]
            for t, g in sel.groupby("target"):
                g = g[[tuple(o) == modal[t] for o in g.opts]]
                g = g[g[scol].notna()] if scol in g.columns else g.iloc[0:0]
                g = g[[np.isfinite(v).all() for v in g[scol]]] if len(g) else g
                if len(g) < 20 or g.truth.nunique() < 2:
                    continue
                arr = np.array(g[scol].tolist(), float)
                cent = arr - arr.mean(axis=1, keepdims=True)
                a = weighted_auc(cent, g.truth.to_numpy(), list(modal[t]))
                if not np.isnan(a):
                    aucs.append(a)
            line += (f"{np.mean(aucs):>14.3f}" if aucs else f"{'-':>14}")
        print(line)

    # ---- 4. replicate -----------------------------------------------------
    print("\n=== replicate agreement (the ceiling) ===")
    for arm in SCORED:
        pcol, rcol = f"pred_{arm}", f"rep_pred_{arm}"
        if rcol not in df.columns:
            continue
        g = df[df[rcol].notna() & df[pcol].notna()]
        if len(g):
            print(f"  {arm:<14} {(g[pcol] == g[rcol]).mean():>6.1%}"
                  f"   n={len(g):,}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(
        OUT / f"ladder_readout_curve_{tag}.csv", index=False)
    print(f"\nwrote ladder_readout_curve_{tag}.csv to {OUT}")


if __name__ == "__main__":
    main()
