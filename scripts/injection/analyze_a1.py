"""A1 injection retest analysis, to the 8 Aug pre-registration.

Reads the scored country / temporal files and writes the tables the
pre-registration names: per-condition accuracy (T0.1 M convention), paired
within-base_id deltas against baseline with target-clustered bootstrap CIs,
per-question TV against the empirical marginals, flip rates against the
same-serving replicate ceiling, the 2x2 interaction, and the
country-already-disclosed split.

    python scripts/injection/analyze_a1.py

Every number quoted anywhere else must come out of the CSVs this writes;
`verify_a1_numbers.py` re-checks them by script.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_sampling.analysis import clustered_bootstrap_ci, normalized_accuracy

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT.parent / "analysis" / "injection"

EXPERIMENTS = {
    "country_injection": "baseline",
    "temporal_context": "baseline",
}
ARMS = ("label_num", "echo_plain")


def own_country_names() -> dict[tuple[str, str], str]:
    """(survey, country code) -> country name, from the country instance file."""
    src = ROOT.parent / "outputs_recovered" / "country_injection_instances.jsonl"
    names: dict[tuple[str, str], str] = {}
    if not src.exists():
        return names
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        names[(r["survey"], str(r["country"]))] = r["country_name"]
    return names


def load(experiment: str, tag: str) -> tuple[pd.DataFrame, dict]:
    """Join the input file (conditions, option sets) to the scored file."""
    inp = ROOT / "outputs" / experiment / "inputs" / f"{experiment}_label_set.jsonl"
    res = ROOT / "outputs" / experiment / "results" / f"{experiment}_label_results_{tag}.jsonl"

    meta: dict[str, dict] = {}
    disclosed: dict[str, bool] = {}
    own = own_country_names()
    for line in inp.open(encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] == "baseline":
            # Pre-registered caveat: some temporal profiles already name the
            # respondent's country, so with_country there is explicit injection
            # ON TOP OF disclosure. The 8 Aug note put this at 232/2,100 but
            # recorded no rule; this is the stated rule used here — the
            # respondent's own country name appears verbatim as a profile
            # ANSWER. See the dated note in PAPER_STATE for the discrepancy.
            name = own.get((r["survey"], str(r.get("country"))))
            disclosed[r["base_id"]] = bool(name) and any(
                str(a).strip().lower() == name.strip().lower()
                for a in r["questions"].values()
            )
        meta[r["example_id"]] = {
            "condition": r["condition"],
            "base_id": r["base_id"],
            "survey": r["survey"],
            "target_code": r["target_code"],
            "ground_truth": r["ground_truth"],
            "options": r["option_sets"]["original"],
        }

    rows = []
    for line in res.open(encoding="utf-8"):
        r = json.loads(line)
        m = meta[r["example_id"]]
        row = {
            "example_id": r["example_id"],
            "base_id": m["base_id"],
            "condition": m["condition"],
            "target": f"{m['survey']}|{m['target_code']}",
            "survey": m["survey"],
            "ground_truth": m["ground_truth"],
            "n_options": len(m["options"]),
        }
        for arm in ARMS:
            cell = (r.get("results") or {}).get(f"original|{arm}")
            pred, miss = None, True
            if cell and "error" not in cell:
                scores = cell.get("scores") or {}
                finite = [v for v in scores.values() if np.isfinite(v)]
                miss = len(finite) != len(scores) or not scores
                pred = cell.get("predicted")
            row[f"{arm}_pred"] = pred
            row[f"{arm}_miss"] = miss
            row[f"{arm}_correct"] = float(pred == m["ground_truth"]) if pred is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df["country_disclosed"] = df["base_id"].map(disclosed)
    return df, meta


def m_by_target(df: pd.DataFrame) -> dict[str, int]:
    """M = distinct answers GIVEN per target, M >= 2 (T0.1 convention)."""
    base = df[df["condition"] == "baseline"]
    return {t: g["ground_truth"].nunique() for t, g in base.groupby("target")}


def norm_acc(sub: pd.DataFrame, arm: str, m_map: dict[str, int]) -> float:
    """Mean accuracy within target, normalized by that target's M, then across."""
    vals = []
    for t, g in sub.groupby("target"):
        m = m_map.get(t, 0)
        if m < 2:
            continue
        vals.append(normalized_accuracy(float(g[f"{arm}_correct"].mean()), m))
    return float(np.mean(vals)) if vals else float("nan")


def tv_by_target(sub: pd.DataFrame, arm: str) -> float:
    """Target-averaged total variation between predicted and true marginals."""
    vals = []
    for _, g in sub.groupby("target"):
        preds = g[f"{arm}_pred"].dropna()
        if preds.empty:
            continue
        labels = set(preds) | set(g["ground_truth"])
        p = Counter(preds)
        q = Counter(g["ground_truth"])
        np_, nq_ = len(preds), len(g)
        vals.append(0.5 * sum(abs(p[k] / np_ - q[k] / nq_) for k in labels))
    return float(np.mean(vals)) if vals else float("nan")


def paired(df: pd.DataFrame, arm: str, cond: str, baseline: str = "baseline") -> dict:
    """correct(cond) - correct(baseline) per base_id, target-clustered CI."""
    a = df[df["condition"] == cond].set_index("base_id")
    b = df[df["condition"] == baseline].set_index("base_id")
    shared = a.index.intersection(b.index)
    d = (a.loc[shared, f"{arm}_correct"] - b.loc[shared, f"{arm}_correct"]).astype(float)
    clusters = a.loc[shared, "target"].to_numpy()
    flip = (a.loc[shared, f"{arm}_pred"] != b.loc[shared, f"{arm}_pred"]).mean()
    v = d.to_numpy()
    lo, hi = clustered_bootstrap_ci(v, clusters)
    per_cluster = pd.Series(v).groupby(clusters).mean()
    return {
        "n_pairs": int(len(shared)),
        "delta_acc": float(per_cluster.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "flip_rate": float(flip),
    }


def run(tag: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    levels, deltas = [], []

    for experiment, baseline in EXPERIMENTS.items():
        path = ROOT / "outputs" / experiment / "results" / f"{experiment}_label_results_{tag}.jsonl"
        if not path.exists():
            print(f"skip {experiment}: no {path.name}")
            continue
        df, _ = load(experiment, tag)
        m_map = m_by_target(df)
        print(f"{experiment}: {len(df)} rows, {df['target'].nunique()} targets, "
              f"{sum(1 for m in m_map.values() if m < 2)} dropped for M<2")

        for arm in ARMS:
            for cond, g in df.groupby("condition"):
                levels.append({
                    "experiment": experiment, "model": tag, "arm": arm,
                    "condition": cond, "n": len(g),
                    "miss_rate": float(g[f"{arm}_miss"].mean()),
                    "acc_raw": float(g[f"{arm}_correct"].mean()),
                    "norm_acc": norm_acc(g, arm, m_map),
                    "tv": tv_by_target(g, arm),
                })
            for cond in sorted(set(df["condition"]) - {baseline}):
                deltas.append({
                    "experiment": experiment, "model": tag, "arm": arm,
                    "contrast": f"{cond}-{baseline}", **paired(df, arm, cond, baseline),
                })

            # Sharp tests named in the pre-registration.
            if experiment == "temporal_context":
                deltas.append({
                    "experiment": experiment, "model": tag, "arm": arm,
                    "contrast": "with_year-with_year_placebo",
                    **paired(df, arm, "with_year", "with_year_placebo"),
                })
                # 2x2 interaction: combined - (country + year) effects.
                dc = paired(df, arm, "with_country", baseline)["delta_acc"]
                dy = paired(df, arm, "with_year", baseline)["delta_acc"]
                dcy = paired(df, arm, "with_country_and_year", baseline)["delta_acc"]
                deltas.append({
                    "experiment": experiment, "model": tag, "arm": arm,
                    "contrast": "interaction_2x2", "n_pairs": 0,
                    "delta_acc": dcy - dc - dy,
                    "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "flip_rate": float("nan"),
                })
                # Country already disclosed by the profile vs not.
                for flag, label in ((True, "disclosed"), (False, "undisclosed")):
                    sub = df[df["country_disclosed"] == flag]
                    if sub.empty:
                        continue
                    deltas.append({
                        "experiment": experiment, "model": tag, "arm": arm,
                        "contrast": f"with_country-{baseline}|{label}",
                        **paired(sub, arm, "with_country", baseline),
                    })

    pd.DataFrame(levels).to_csv(OUTDIR / f"a1_levels_{tag}.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUTDIR / f"a1_deltas_{tag}.csv", index=False)
    print(f"wrote {OUTDIR / f'a1_levels_{tag}.csv'}")
    print(f"wrote {OUTDIR / f'a1_deltas_{tag}.csv'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="qwen_qwen3-32b")
    a = ap.parse_args(argv)
    run(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
