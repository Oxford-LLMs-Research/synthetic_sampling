"""A3 default-option analysis, to the 13 Aug pre-registration as
amended 15 Aug (PAPER_STATE): Don't know and Refusal stay in Phase 2.

Written BEFORE the jobs land, because this is the experiment most likely
to be mis-read from a generic accuracy table: raw accuracy is NOT
comparable across cells (dk_absent has fewer options, and the 13 DK-truth
rows cannot be correct there). The locked estimands:

- DK census (descriptive), dk_present cell: per-target ARGMAX DK rate
  (the full-option argmax lands on a DK option) vs the matched human
  share from the set rows; DK softmax MASS reported descriptively only
  (raw readout mass is overconfident per T0.1; the criterion is argmax,
  which is temperature-invariant). Inflation flag per model: mean
  per-target inflation (predicted - human) <= 5pp AND at most 2 of 16
  targets over 10pp. A fail is a model finding (over-prediction of a
  real category). It does NOT drop Don't know / Refusal from Phase 2
  (the 13 Aug worst-case switcher was retracted 15 Aug).
- Substantive contrast (the stability read), substantive-truth rows only:
  dk_present scored by argmax over SUBSTANTIVE options (DK scores
  dropped; equivalent to renormalising) vs dk_absent argmax, paired on
  base_id, target-clustered bootstrap CI; within +-0.02 expected,
  falsifier +-0.04. Flip rate read against the replicate ceiling. The
  dk_absent cell is a diagnostic, not a candidate template.
- Refusal split: on targets carrying both "Don't know" and Refusal (the
  four clean ESS dual-carriers), the DK-argmax rate is reported per
  removed option, not just pooled.

    python scripts/default_options/analyze_a3.py --tag qwen_qwen3-32b

Every number quoted anywhere else must come out of the CSVs this writes;
a verify_a3_numbers.py pins them at landing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_sampling.analysis import (
    clustered_bootstrap_ci,
    normalized_accuracy,
    replicate_agreement,
)

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT.parent / "analysis" / "default_options"

ARMS = ("label_num", "echo_plain")
CELLS = ("dk_present", "dk_absent")

# Descriptive inflation band (PAPER_STATE 13 Aug; convention-switcher
# retracted 15 Aug). `passes` means "does not inflate", not "keep DK".
MEAN_INFLATION_MAX = 0.05
PER_TARGET_CAP = 0.10
PER_TARGET_CAP_MAX_N = 2


def _softmax(scores: dict) -> dict[str, float]:
    keys = list(scores)
    v = np.array([scores[k] for k in keys], dtype=float)
    if not len(v) or not np.all(np.isfinite(v)):
        return {}
    e = np.exp(v - v.max())
    p = e / e.sum()
    return dict(zip(keys, p.astype(float)))


def load(tag: str) -> tuple[pd.DataFrame, list[dict]]:
    inp = ROOT / "outputs" / "default_options" / "inputs" / "a3_dk_set.jsonl"
    res = (ROOT / "outputs" / "default_options" / "results"
           / f"a3_dk_results_{tag}.jsonl")

    meta: dict[str, dict] = {}
    for line in inp.open(encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r

    rows, raw = [], []
    for line in res.open(encoding="utf-8"):
        r = json.loads(line)
        raw.append(r)
        m = meta[r["example_id"]]
        opts = m["option_sets"]["original"]
        dk = set(m["dk_options"])
        row = {
            "example_id": r["example_id"],
            "base_id": m["base_id"],
            "cell": m["arm_label"],
            "target": f"{m['survey']}|{m['target_code']}",
            "survey": m["survey"],
            "ground_truth": m["ground_truth"],
            "truth_is_dk": bool(m["truth_is_dk"]),
            "n_options": len(opts),
        }
        for arm in ARMS:
            cell = (r.get("results") or {}).get(f"original|{arm}")
            pred, miss = None, True
            dk_hit, dk_mass = np.nan, np.nan
            subst_pred, subst_conf = None, np.nan
            if cell and "error" not in cell:
                scores = cell.get("scores") or {}
                finite = [v for v in scores.values() if np.isfinite(v)]
                miss = len(finite) != len(scores) or not scores
                pred = cell.get("predicted")
                probs = _softmax(scores)
                if m["arm_label"] == "dk_present":
                    dk_hit = float(pred in dk) if pred is not None else np.nan
                    dk_mass = (sum(p for o, p in probs.items() if o in dk)
                               if probs else np.nan)
                    # Locked estimand: argmax over substantive options only
                    # (renormalisation cannot change the subset argmax).
                    subst = {o: v for o, v in scores.items() if o not in dk}
                    finite_s = [v for v in subst.values() if np.isfinite(v)]
                    if subst and len(finite_s) == len(subst):
                        subst_pred = max(subst, key=subst.get)
                        sp = _softmax(subst)
                        subst_conf = max(sp.values()) if sp else np.nan
                else:
                    subst_pred = pred
                    subst_conf = max(probs.values()) if probs else np.nan
            row[f"{arm}_pred"] = pred
            row[f"{arm}_miss"] = miss
            row[f"{arm}_dk_hit"] = dk_hit
            row[f"{arm}_dk_mass"] = dk_mass
            row[f"{arm}_subst_pred"] = subst_pred
            row[f"{arm}_subst_conf"] = subst_conf
            row[f"{arm}_subst_correct"] = (
                np.nan if (row["truth_is_dk"] or subst_pred is None)
                else float(subst_pred == m["ground_truth"]))
        rows.append(row)
    return pd.DataFrame(rows), raw


def m_by_target(df: pd.DataFrame) -> dict[str, int]:
    base = df[(df["cell"] == "dk_present") & ~df["truth_is_dk"]]
    return {t: g["ground_truth"].nunique() for t, g in base.groupby("target")}


def norm_acc(sub: pd.DataFrame, arm: str, m_map: dict[str, int]) -> float:
    vals = []
    for t, g in sub.groupby("target"):
        m = m_map.get(t, 0)
        if m < 2:
            continue
        acc = g[f"{arm}_subst_correct"].mean()
        if np.isfinite(acc):
            vals.append(normalized_accuracy(float(acc), m))
    return float(np.mean(vals)) if vals else float("nan")


def tv_by_target(sub: pd.DataFrame, arm: str) -> float:
    """TV over substantive predictions vs substantive truths."""
    vals = []
    for _, g in sub.groupby("target"):
        preds = g[f"{arm}_subst_pred"].dropna()
        truths = g.loc[~g["truth_is_dk"], "ground_truth"]
        if preds.empty or truths.empty:
            continue
        labels = set(preds) | set(truths)
        p, q = Counter(preds), Counter(truths)
        np_, nq_ = len(preds), len(truths)
        vals.append(0.5 * sum(abs(p[k] / np_ - q[k] / nq_) for k in labels))
    return float(np.mean(vals)) if vals else float("nan")


def ece(sub: pd.DataFrame, arm: str, n_bins: int = 10) -> float:
    g = sub.dropna(subset=[f"{arm}_subst_conf", f"{arm}_subst_correct"])
    if g.empty:
        return float("nan")
    conf = g[f"{arm}_subst_conf"].to_numpy()
    corr = g[f"{arm}_subst_correct"].to_numpy()
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(out)


def paired_substantive(df: pd.DataFrame, arm: str) -> dict:
    """dk_present (substantive argmax) minus dk_absent, substantive-truth
    rows only — the ONLY accuracy comparison this design licenses."""
    sub = df[~df["truth_is_dk"]]
    a = sub[sub["cell"] == "dk_present"].set_index("base_id")
    b = sub[sub["cell"] == "dk_absent"].set_index("base_id")
    shared = a.index.intersection(b.index)
    da = a.loc[shared, f"{arm}_subst_correct"].astype(float)
    db = b.loc[shared, f"{arm}_subst_correct"].astype(float)
    keep = da.notna() & db.notna()
    d = (da - db)[keep]
    clusters = a.loc[shared, "target"].to_numpy()[keep.to_numpy()]
    agree = (a.loc[shared, f"{arm}_subst_pred"][keep]
             == b.loc[shared, f"{arm}_subst_pred"][keep]).mean()
    lo, hi = clustered_bootstrap_ci(d.to_numpy(), clusters)
    per_cluster = pd.Series(d.to_numpy()).groupby(clusters).mean()
    return {
        "n_pairs": int(keep.sum()),
        "delta_acc": float(per_cluster.mean()),
        "ci_lo": lo, "ci_hi": hi,
        "flip_rate": float(1.0 - agree),
        "agree_rate": float(agree),
    }


def dk_census(df: pd.DataFrame, meta_path: Path, arm: str,
              tag: str) -> tuple[list[dict], dict]:
    """Per-target predicted-vs-human DK, plus the locked verdict row."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pres = df[df["cell"] == "dk_present"]
    rows = []
    for t, g in pres.groupby("target"):
        s, code = t.split("|", 1)
        human = meta["targets"][f"{s}|{code}"]["human_dk_share"]
        pred_rate = float(g[f"{arm}_dk_hit"].mean())
        rows.append({
            "model": tag, "arm": arm, "target": t, "n": len(g),
            "human_dk_share": human,
            "pred_dk_rate": pred_rate,
            "pred_dk_mass": float(g[f"{arm}_dk_mass"].mean()),
            "inflation": pred_rate - human,
        })
    infl = np.array([r["inflation"] for r in rows], dtype=float)
    n_over = int((infl > PER_TARGET_CAP).sum())
    verdict = {
        "model": tag, "arm": arm,
        "n_targets": len(rows),
        "mean_inflation": float(infl.mean()),
        "n_targets_over_cap": n_over,
        "passes": bool(infl.mean() <= MEAN_INFLATION_MAX
                       and n_over <= PER_TARGET_CAP_MAX_N),
    }
    return rows, verdict


def refusal_split(df: pd.DataFrame, tag: str) -> list[dict]:
    """DK-argmax rate per removed option, on dual-carrier targets."""
    inp = ROOT / "outputs" / "default_options" / "inputs" / "a3_dk_set.jsonl"
    dk_by_target: dict[str, list[str]] = {}
    for line in inp.open(encoding="utf-8"):
        r = json.loads(line)
        dk_by_target[f"{r['survey']}|{r['target_code']}"] = r["dk_options"]
    pres = df[df["cell"] == "dk_present"]
    rows = []
    for t, g in pres.groupby("target"):
        dks = dk_by_target.get(t, [])
        if len(dks) < 2:
            continue
        for arm in ARMS:
            for dk in dks:
                rows.append({
                    "model": tag, "arm": arm, "target": t, "dk_option": dk,
                    "pred_rate": float((g[f"{arm}_pred"] == dk).mean()),
                })
    return rows


def run(tag: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df, raw = load(tag)
    m_map = m_by_target(df)
    meta_path = (ROOT / "outputs" / "default_options" / "inputs"
                 / "a3_dk_meta.json")
    print(f"{len(df)} scored rows ({df['base_id'].nunique()} base_ids), "
          f"{df['target'].nunique()} targets, "
          f"{int(df['truth_is_dk'].sum() / 2)} DK-truth pairs")

    levels, contrasts, census, verdicts = [], [], [], []
    for arm in ARMS:
        for cell in CELLS:
            g = df[df["cell"] == cell]
            levels.append({
                "model": tag, "arm": arm, "cell": cell, "n": len(g),
                "miss_rate": float(g[f"{arm}_miss"].mean()),
                "subst_acc_raw": float(g[f"{arm}_subst_correct"].mean()),
                "norm_acc": norm_acc(g, arm, m_map),
                "tv": tv_by_target(g, arm),
                "mean_conf": float(g[f"{arm}_subst_conf"].mean()),
                "ece": ece(g, arm),
            })
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "present-absent",
            **paired_substantive(df, arm),
        })
        rate, n = replicate_agreement(raw, arm=arm)
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "replicate",
            "n_pairs": n, "delta_acc": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "flip_rate": float(1.0 - rate) if np.isfinite(rate) else float("nan"),
            "agree_rate": rate,
        })
        c_rows, verdict = dk_census(df, meta_path, arm, tag)
        census.extend(c_rows)
        verdicts.append(verdict)

    split = refusal_split(df, tag)

    pd.DataFrame(levels).to_csv(OUTDIR / f"a3_levels_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(
        OUTDIR / f"a3_contrasts_{tag}.csv", index=False)
    pd.DataFrame(census).to_csv(
        OUTDIR / f"a3_dk_census_{tag}.csv", index=False)
    pd.DataFrame(verdicts).to_csv(
        OUTDIR / f"a3_dk_verdict_{tag}.csv", index=False)
    pd.DataFrame(split).to_csv(
        OUTDIR / f"a3_refusal_split_{tag}.csv", index=False)
    for stem in ("levels", "contrasts", "dk_census", "dk_verdict",
                 "refusal_split"):
        print(f"wrote {OUTDIR / f'a3_{stem}_{tag}.csv'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="qwen_qwen3-32b")
    a = ap.parse_args(argv)
    run(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
