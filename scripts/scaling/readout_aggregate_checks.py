#!/usr/bin/env python
r"""Do the paper's aggregate-level claims survive a better readout?

Three claims in the paper rest on WHICH option the model names, aggregated over
respondents, and none of them was ever varied over the elicitation:

    marginal recovery    a synthetic sample's answer shares against the real
                         shares (the deployment pitch is aggregate, not
                         individual)
    default answers      the model emits a content-free default ("Don't know"
                         and its relatives) at 47% against a human rate of 1.5%
    Western anchoring    where a country's modal answer differs from the
                         Western bloc's, whose majority does the prediction
                         follow?

The readout experiment scored identical instances under several elicitations in
one serving, so recomputing each claim per arm isolates the readout. Scope: the
48-question readout sample, 100 respondents per question, one model per results
file. Absolute levels will not match the paper; the question is whether each
claim is a property of the model or of the echo readout.

Runs over every readout results file present, so it re-runs unchanged when the
model grid lands.

    python .../readout_aggregate_checks.py            # all result files
    python .../readout_aggregate_checks.py --results <one file>
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# analyze_readout re-wraps sys.stdout at import time; reconfigure rather than
# stacking a second TextIOWrapper, whose garbage collection closes the buffer.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))            # analyze_readout (boot_ci, qmean)
sys.path.insert(0, str(SCRIPTS.parent))     # analyze_default_answers (DEFAULTS)

from analyze_readout import boot_ci, qmean                    # noqa: E402
from analyze_default_answers import is_default                # noqa: E402
from analyze_marginal_recovery import js_divergence, tv_distance  # noqa: E402

REPO = SCRIPTS.parents[1]
SCALE = REPO / "outputs" / "scaling_experiment"
RESULTS_DIR = SCALE / "readout_results"
ANALYSIS = REPO.parent / "analysis"
OUT = ANALYSIS / "readout"

# Elicitations whose predictions are readings of the model. The generate arms
# are excluded (57.7% self-agreement at temperature 0), and the neutral PMI
# arms are excluded because their argmax is the fluency default by
# construction, not a prediction about anyone.
ARM_ORDER = ["echo_plain", "echo_listed", "label_num", "label_num_natural"]

MIN_COUNTRY_CELL = 10       # exploratory country table only
MIN_ANCHOR_N = 150          # below this the pull is directional, no CI stars


def load_meta(path: Path) -> dict:
    meta = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r
    return meta


def load_predictions(results_path: Path, meta: dict) -> pd.DataFrame:
    """One row per (instance, arm) with the arm's prediction, original set."""
    recs = []
    for line in open(results_path, encoding="utf-8"):
        r = json.loads(line)
        m = meta.get(r["example_id"])
        if m is None:
            continue
        for key, d in r["results"].items():
            setname, arm = key.split("|", 1)
            if setname != "original" or arm not in ARM_ORDER:
                continue
            if "error" in d or d.get("predicted") is None:
                continue
            recs.append({
                "eid": r["example_id"], "arm": arm,
                "survey": m["survey"], "target_code": m["target_code"],
                "q": f"{m['survey']}|{m['target_code']}",
                "respondent_id": str(m["id"]),
                "truth": m["ground_truth"], "pred": d["predicted"],
                "options": tuple(m["option_sets"]["original"]),
            })
    return pd.DataFrame(recs)


def tag_of(path: Path) -> str:
    stem = path.stem
    for prefix in ("readout_results_", "readout_grid_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


# --------------------------------------------------------------------------
def marginal_recovery(df: pd.DataFrame, arms: list[str], tag: str,
                      resp_country: pd.DataFrame) -> pd.DataFrame:
    print("=== marginal recovery by readout ===")
    print("TV(predicted shares, true shares) per question, then averaged;")
    print("lower is better, 0 means the synthetic tally matches the real one\n")

    rows = []
    for arm in arms:
        g = df[df.arm == arm]
        for q, gq in g.groupby("q"):
            true_c = collections.Counter(gq.truth)
            pred_c = collections.Counter(gq.pred)
            rows.append({"arm": arm, "q": q, "n": len(gq),
                         "tv": tv_distance(pred_c, true_c),
                         "jsd": js_divergence(pred_c, true_c)})
    per_q = pd.DataFrame(rows)

    print(f"{'readout':<20}{'mean TV':>9}{'mean JSD':>10}{'questions':>11}")
    for arm in arms:
        s = per_q[per_q.arm == arm]
        print(f"{arm:<20}{s.tv.mean():>9.3f}{s.jsd.mean():>10.3f}"
              f"{len(s):>11}")

    print("\npaired against echo_plain, per question (negative TV difference "
          "means the arm\ntracks the real shares better):")
    base = per_q[per_q.arm == "echo_plain"].set_index("q")
    for arm in arms:
        if arm == "echo_plain":
            continue
        g = per_q[per_q.arm == arm].set_index("q")
        common = base.index.intersection(g.index)
        if len(common) < 2:
            continue
        d = (g.loc[common, "tv"] - base.loc[common, "tv"]).to_numpy()
        lo, hi = boot_ci(d, common.to_numpy())
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {arm:<20} dTV {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] {star}"
              f"   n={len(common)} questions")

    # Exploratory only: the paper's own unit is the (question, country) cell at
    # n >= 30. 100 respondents per question scatter across countries, so cells
    # here are small and few; nothing at this granularity is a re-test.
    dfc = df.merge(resp_country, on=["survey", "respondent_id"], how="left")
    cell_rows = []
    for arm in arms:
        g = dfc[dfc.arm == arm]
        for q, gq in g.groupby("q"):
            global_true = collections.Counter(gq.truth)
            for country, cg in gq.groupby("country"):
                if len(cg) < MIN_COUNTRY_CELL or not isinstance(country, str):
                    continue
                true_c = collections.Counter(cg.truth)
                cell_rows.append({
                    "arm": arm, "q": q, "country": country, "n": len(cg),
                    "tv_model": tv_distance(collections.Counter(cg.pred), true_c),
                    "tv_global": tv_distance(global_true, true_c)})
    cells = pd.DataFrame(cell_rows)
    if not cells.empty:
        print(f"\nexploratory country cells (n >= {MIN_COUNTRY_CELL}; the "
              f"paper's threshold is 30, so this\nis indicative only):")
        print(f"{'readout':<20}{'cells':>7}{'TV model':>10}{'TV global':>11}"
              f"{'model beats global':>20}")
        for arm in arms:
            s = cells[cells.arm == arm]
            if s.empty:
                continue
            beats = (s.tv_model < s.tv_global).mean()
            print(f"{arm:<20}{len(s):>7}{s.tv_model.mean():>10.3f}"
                  f"{s.tv_global.mean():>11.3f}{beats:>19.1%}")
        cells.to_csv(OUT / f"marginal_recovery_cells_{tag}.csv", index=False)

    per_q.to_csv(OUT / f"marginal_recovery_readout_{tag}.csv", index=False)
    return per_q


# --------------------------------------------------------------------------
def default_rates(df: pd.DataFrame, arms: list[str], tag: str) -> pd.DataFrame:
    print("\n=== default answers by readout ===")
    print("share of predictions that are a content-free default, on the "
          "questions whose\noption set offers one; the human rate on the same "
          "instances is the yardstick\n")

    has_default = df.options.map(lambda opts: any(is_default(o) for o in opts))
    sub = df[has_default]
    n_q = sub.q.nunique()
    print(f"{n_q} of {df.q.nunique()} questions offer a default option "
          f"({sub.eid.nunique():,} instances); the rest are\nexcluded, "
          f"because no readout can emit a default that is not on offer")
    if sub.empty:
        return pd.DataFrame()

    rows = []
    print(f"\n{'readout':<20}{'model default':>15}{'human default':>15}")
    for arm in arms:
        g = sub[sub.arm == arm]
        if g.empty:
            continue
        pred_rate = g.pred.map(is_default).mean()
        human_rate = g.truth.map(is_default).mean()
        print(f"{arm:<20}{pred_rate:>15.1%}{human_rate:>15.1%}")
        rows.append({"arm": arm, "n": len(g), "n_q": g.q.nunique(),
                     "model_default_rate": pred_rate,
                     "human_default_rate": human_rate})

    print("\npaired against echo_plain, per question:")
    base = sub[sub.arm == "echo_plain"]
    base_q = base.assign(d=base.pred.map(is_default).astype(float)) \
                 .groupby("q").d.mean()
    for arm in arms:
        if arm == "echo_plain":
            continue
        g = sub[sub.arm == arm]
        if g.empty:
            continue
        gq = g.assign(d=g.pred.map(is_default).astype(float)).groupby("q").d.mean()
        common = base_q.index.intersection(gq.index)
        if len(common) < 2:
            continue
        d = (gq.loc[common] - base_q.loc[common]).to_numpy()
        lo, hi = boot_ci(d, common.to_numpy())
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {arm:<20} d(default rate) {d.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] {star}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"default_rates_readout_{tag}.csv", index=False)
    return out


# --------------------------------------------------------------------------
def western_anchoring(df: pd.DataFrame, arms: list[str], tag: str,
                      cells: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Western anchoring by readout ===")
    print("on WVS instances whose country's modal answer differs from the "
          "Western bloc's:\npull = P(pred = Western mode) - P(pred = own mode)\n")

    joined = df.merge(cells[["example_id", "country", "local_mode",
                             "bloc_mode", "pooled_mode"]],
                      left_on="eid", right_on="example_id", how="inner")
    n = joined[joined.arm == arms[0]].eid.nunique() if len(joined) else 0
    print(f"{n} readout instances fall in disagreement cells "
          f"({joined.country.nunique() if len(joined) else 0} countries, "
          f"{joined.target_code.nunique() if len(joined) else 0} questions)")
    if n == 0:
        print("nothing to test on this sample")
        return pd.DataFrame()
    if n < MIN_ANCHOR_N:
        print(f"below {MIN_ANCHOR_N} instances this is a DIRECTIONAL check "
              f"only; the full re-test\nrides on the model grid or a larger "
              f"sample")

    rows = []
    rng = np.random.default_rng(42)
    print(f"\n{'readout':<20}{'P(=Western)':>13}{'P(=own)':>9}{'pull':>8}"
          f"{'95% CI (respondent bootstrap)':>32}")
    for arm in arms:
        g = joined[joined.arm == arm]
        if g.empty:
            continue
        hit_bloc = (g.pred == g.bloc_mode).to_numpy().astype(float)
        hit_local = (g.pred == g.local_mode).to_numpy().astype(float)
        pull = hit_bloc - hit_local
        bs = pull[rng.integers(0, len(pull), size=(2000, len(pull)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{arm:<20}{hit_bloc.mean():>13.3f}{hit_local.mean():>9.3f}"
              f"{pull.mean():>8.3f}         [{lo:+.3f}, {hi:+.3f}]")
        rows.append({"arm": arm, "n": len(g),
                     "n_countries": g.country.nunique(),
                     "p_bloc": hit_bloc.mean(), "p_local": hit_local.mean(),
                     "pull": pull.mean(), "lo": lo, "hi": hi,
                     "directional_only": len(g) < MIN_ANCHOR_N})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"western_anchoring_readout_{tag}.csv", index=False)
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=None,
                    help="one results file; default is every readout_results_*"
                         " and readout_grid_* file present")
    ap.add_argument("--input", type=Path, default=SCALE / "readout_set.jsonl")
    args = ap.parse_args()

    files = ([args.results] if args.results else
             sorted(RESULTS_DIR.glob("readout_results_*.jsonl"))
             + sorted(RESULTS_DIR.glob("readout_grid_*.jsonl")))
    if not files:
        sys.exit("no readout results files found")

    meta = load_meta(args.input)
    resp_country = pd.read_csv(
        ANALYSIS / "marginal_recovery" / "respondent_country.csv",
        dtype=str, keep_default_na=False)
    anchor_cells = pd.read_csv(ANALYSIS / "western_anchoring" / "cells.csv",
                               dtype=str)
    OUT.mkdir(parents=True, exist_ok=True)

    for path in files:
        tag = tag_of(path)
        print("\n" + "=" * 74)
        print(f"model file {path.name}")
        print("=" * 74 + "\n")
        df = load_predictions(path, meta)
        if df.empty:
            print("no usable predictions; skipped")
            continue
        arms = [a for a in ARM_ORDER if (df.arm == a).any()]
        marginal_recovery(df, arms, tag, resp_country)
        default_rates(df, arms, tag)
        western_anchoring(df, arms, tag, anchor_cells)

    print(f"\nwrote marginal_recovery_readout_*, default_rates_readout_*, "
          f"western_anchoring_readout_* to {OUT}")


if __name__ == "__main__":
    main()
