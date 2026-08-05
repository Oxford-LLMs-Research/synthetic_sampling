#!/usr/bin/env python
r"""Do the paper's two prediction-dependent claims survive a better readout?

Two abstract claims rest not on an accuracy level but on WHICH option the model
names for WHICH respondent:

    heterogeneity flattening   the entropy ratio, H(predicted tally) over
                               H(empirical tally) per question, below 1 meaning
                               the model gives less varied answers than people
    the dissenter penalty      accuracy on respondents who hold their question's
                               modal answer against those who do not, and in
                               particular the below-chance figure where more than
                               80% of a population agrees

Both are therefore exposed to something the paper never varied: how the model is
asked. Changing the readout moves individual predictions on roughly half of
instances, so a claim about which respondents get predicted correctly could in
principle be an artefact of the elicitation rather than of the model.

This recomputes both quantities under each readout on identical instances and
respondents, so the comparison isolates the readout. Absolute values will not
match the paper: this is 48 of 267 questions, 100 respondents each, and one
model at a time. The question here is whether the DIRECTION and the ORDERING
survive, not whether the level reproduces.

    python .../readout_claims_check.py --results <readout_results_*.jsonl>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
OUT = REPO.parent / "analysis" / "readout"

ARMS = ["echo_plain", "echo_listed", "label_num", "label_num_natural"]


def norm_acc(correct: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (correct - 1.0 / M) / (1.0 - 1.0 / M)


def boot_ci(v: np.ndarray, q: np.ndarray, n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(q, return_inverse=True)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    m = np.array([v[inv == i].mean() for i in range(len(keys))])
    d = m[rng.integers(0, len(keys), size=(n, len(keys)))].mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=SCALE / "readout_results" /
                            "readout_results_qwen_qwen3-32b.jsonl")
    ap.add_argument("--input", type=Path, default=SCALE / "readout_set.jsonl")
    args = ap.parse_args()

    meta = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r

    rows = []
    for line in open(args.results, encoding="utf-8"):
        r = json.loads(line)
        m = meta.get(r["example_id"])
        if m is None:
            continue
        rec = {"eid": r["example_id"], "q": f"{m['survey']}|{m['target_code']}",
               "truth": m["ground_truth"]}
        ok = True
        for a in ARMS:
            d = r["results"].get(f"original|{a}")
            if not d or d.get("predicted") is None:
                ok = False
                break
            rec[a] = d["predicted"]
        if ok:
            rows.append(rec)
    d = pd.DataFrame(rows)
    print(f"{len(d):,} instances, {d.q.nunique()} questions, "
          f"model file {args.results.name}\n")

    # Per question: the modal answer, its share, and M as the paper defines it.
    g = d.groupby("q").truth
    mode = g.agg(lambda s: s.value_counts().idxmax())
    share = g.agg(lambda s: s.value_counts(normalize=True).max())
    M = g.nunique()
    d["mode"] = d.q.map(mode)
    d["share"] = d.q.map(share)
    d["M"] = d.q.map(M)
    d["is_modal"] = d.truth == d["mode"]
    d = d[d.M >= 2]

    print("=== heterogeneity flattening: entropy ratio by readout ===")
    print("H(predicted tally) / H(empirical tally), per question, then averaged")
    print("below 1 means the model gives less varied answers than people\n")
    print(f"{'readout':<20}{'entropy ratio':>15}{'95% CI':>20}{'questions <1':>15}")
    ent_rows = []
    for a in ARMS:
        vals, qs = [], []
        for q, gq in d.groupby("q"):
            emp = gq.truth.value_counts(normalize=True).to_numpy()
            pred = gq[a].value_counts(normalize=True).to_numpy()
            h_emp = entropy(emp[emp > 0])
            if h_emp <= 0:
                continue
            vals.append(entropy(pred[pred > 0]) / h_emp)
            qs.append(q)
        v = np.array(vals)
        lo, hi = boot_ci(v, np.array(qs))
        print(f"{a:<20}{v.mean():>15.3f}   [{lo:.3f}, {hi:.3f}]{(v < 1).sum():>13}/{len(v)}")
        ent_rows.append({"arm": a, "entropy_ratio": v.mean(), "lo": lo, "hi": hi,
                         "n_q": len(v), "n_below_1": int((v < 1).sum())})

    print("\n=== dissenter penalty by readout ===")
    print("normalized accuracy, respondents holding their question's modal answer")
    print("against those who do not\n")
    print(f"{'readout':<20}{'modal':>9}{'dissenter':>11}{'penalty':>10}"
          f"{'95% CI on penalty':>22}")
    pen_rows = []
    for a in ARMS:
        corr = (d[a] == d.truth).to_numpy().astype(float)
        na = norm_acc(corr, d.M.to_numpy())
        mod = na[d.is_modal.to_numpy()]
        dis = na[~d.is_modal.to_numpy()]
        # Cluster the interval on the question, and take the penalty per
        # question so a question with many dissenters cannot dominate.
        per_q = []
        for q, gq in d.groupby("q"):
            c = (gq[a] == gq.truth).to_numpy().astype(float)
            nn = norm_acc(c, gq.M.to_numpy())
            im = gq.is_modal.to_numpy()
            if im.all() or (~im).all():
                continue
            per_q.append(nn[im].mean() - nn[~im].mean())
        per_q = np.array(per_q)
        rng = np.random.default_rng(42)
        bs = per_q[rng.integers(0, len(per_q), size=(2000, len(per_q)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{a:<20}{mod.mean():>9.3f}{dis.mean():>11.3f}{per_q.mean():>10.3f}"
              f"      [{lo:+.3f}, {hi:+.3f}]")
        pen_rows.append({"arm": a, "norm_modal": mod.mean(),
                         "norm_dissenter": dis.mean(), "penalty": per_q.mean(),
                         "lo": lo, "hi": hi, "n_q": len(per_q)})

    print("\n=== the abstract's sharpest form: questions above 80% agreement ===")
    print("normalized accuracy of dissenters there; the paper reports -0.19\n")
    hi_share = d[d.share > 0.80]
    print(f"  {hi_share.q.nunique()} questions, {len(hi_share):,} respondents, "
          f"{(~hi_share.is_modal).sum():,} of them dissenters")
    if hi_share.q.nunique() >= 2:
        print(f"\n{'readout':<20}{'modal':>9}{'dissenter':>11}{'95% CI on dissenter':>24}")
        for a in ARMS:
            c = (hi_share[a] == hi_share.truth).to_numpy().astype(float)
            nn = norm_acc(c, hi_share.M.to_numpy())
            im = hi_share.is_modal.to_numpy()
            lo, hi = boot_ci(nn[~im], hi_share.q.to_numpy()[~im])
            print(f"{a:<20}{nn[im].mean():>9.3f}{nn[~im].mean():>11.3f}"
                  f"        [{lo:+.3f}, {hi:+.3f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.results.stem.replace("readout_results", "").strip("_") or "default"
    pd.DataFrame(ent_rows).to_csv(OUT / f"claims_entropy_{tag}.csv", index=False)
    pd.DataFrame(pen_rows).to_csv(OUT / f"claims_dissenter_{tag}.csv", index=False)
    print(f"\nwrote claims_entropy_{tag}.csv and claims_dissenter_{tag}.csv to {OUT}")


if __name__ == "__main__":
    main()
