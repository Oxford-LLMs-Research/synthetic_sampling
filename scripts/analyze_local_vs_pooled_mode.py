#!/usr/bin/env python3
"""Is the majority these models follow the respondent's country's, or the question's?

The dissenter analysis splits respondents by their question-country modal answer,
which credits the models with any local knowledge they have. That split cannot
distinguish a model tracking a country-specific mode from one tracking the
question's overall mode, because the two coincide in most cells.

This isolates the cells where they differ and asks which answer the model emits,
and it re-runs the majority/dissenter accuracy split against the pooled mode.

Outputs analysis/local_vs_pooled/{by_model.csv,summary.csv}.
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repair_ids import repair

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"C:\Users\murrn\cursor\synthetic_sampling"
OUT = os.path.join(ROOT, "analysis", "local_vs_pooled")
MIN_N = 20          # matches the dissenter analysis
PROFILE = "s6m4"
N_BOOT = 2000
MODELS = ["qwen3-32b", "deepseek", "llama3.1_8b_instruct", "olmo3_7b_dpo",
          "gemma3-27b", "qwen3-4b", "llama3.1_70b_instruct", "gpt-oss",
          "olmo3_32b_base", "olmo3_32b_dpo", "olmo3_7b_base",
          "llama3.1_70b_base", "llama3.1_8b_base"]


def load() -> pd.DataFrame:
    frames = []
    for m in MODELS:
        df = pd.read_csv(os.path.join(ROOT, "analysis", m, "results_data.csv"),
                         dtype=str,
                         usecols=["example_id", "survey", "respondent_id",
                                  "target_code", "profile_type", "predicted",
                                  "ground_truth"])
        df = repair(df[df["profile_type"] == PROFILE], verbose=False)
        df["model"] = m
        frames.append(df)
    a = pd.concat(frames, ignore_index=True)
    rc = pd.read_csv(os.path.join(ROOT, "analysis", "marginal_recovery",
                                  "respondent_country.csv"),
                     dtype=str, keep_default_na=False)
    a = a.merge(rc, on=["survey", "respondent_id"], how="left")
    return a[a["country"].notna() & (a["country"] != "")]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    a = load()

    # Modes are a property of the recorded answers, so one model's rows suffice.
    truth = a[a["model"] == MODELS[0]]
    cell = (truth.groupby(["survey", "target_code", "country"])["ground_truth"]
            .agg(n="size", local_mode=lambda s: s.value_counts().idxmax())
            .reset_index())
    cell = cell[cell["n"] >= MIN_N]
    pooled = (truth.groupby(["survey", "target_code"])["ground_truth"]
              .agg(pooled_mode=lambda s: s.value_counts().idxmax()).reset_index())
    cell = cell.merge(pooled, on=["survey", "target_code"])
    cell["differs"] = cell["local_mode"] != cell["pooled_mode"]

    n_cells, n_q = len(cell), cell.groupby(["survey", "target_code"]).ngroups
    share_differ = cell["differs"].mean()
    print(f"cells (n>={MIN_N}): {n_cells:,} over {n_q} questions")
    print(f"country mode differs from pooled mode in {share_differ:.1%} "
          f"({int(cell['differs'].sum()):,} cells)\n")

    d = a.merge(cell, on=["survey", "target_code", "country"], how="inner")
    disc = d[d["differs"]].copy()
    disc["says_local"] = disc["predicted"] == disc["local_mode"]
    disc["says_pooled"] = disc["predicted"] == disc["pooled_mode"]

    by = (disc.groupby("model")[["says_local", "says_pooled"]].mean()
          .sort_values("says_local", ascending=False))
    by["diff"] = by["says_local"] - by["says_pooled"]
    print("Where the two modes differ, share of predictions matching each:")
    print(by.round(4).to_string())

    # Cluster the bootstrap on country: cells from one country share a mode.
    countries = disc["country"].unique()
    rng = np.random.default_rng(0)
    g = {c: sub for c, sub in disc.groupby("country")}
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draw = pd.concat([g[c] for c in rng.choice(countries, len(countries))],
                         ignore_index=True)
        diffs[b] = draw["says_local"].mean() - draw["says_pooled"].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = disc["says_local"].mean() - disc["says_pooled"].mean()
    print(f"\n13-model mean: local {disc['says_local'].mean():.4f}  "
          f"pooled {disc['says_pooled'].mean():.4f}")
    print(f"difference {point:+.4f}  95% country-clustered CI "
          f"[{lo:+.4f}, {hi:+.4f}]")
    print(f"models favouring the local mode: {(by['diff'] > 0).sum()} of {len(by)}")

    # Does the accuracy gap depend on which mode defines "majority"?
    d["correct"] = d["predicted"] == d["ground_truth"]
    rows = []
    for label, key in (("country", "local_mode"), ("pooled", "pooled_mode")):
        d["agrees"] = d["ground_truth"] == d[key]
        t = d.groupby(["model", "agrees"])["correct"].mean().unstack()
        rows.append({"split": label, "share_agreeing": d["agrees"].mean(),
                     "acc_agree": t[True].mean(), "acc_disagree": t[False].mean(),
                     "gap": (t[True] - t[False]).mean()})
    gaps = pd.DataFrame(rows)
    print("\nMajority/dissenter accuracy split, by which mode defines majority:")
    print(gaps.round(4).to_string(index=False))

    # The sharpest form: is the majority/dissenter gap present at all when the
    # country's mode is NOT the common answer? Discordant cells have flatter
    # answers (lower modal share), which would shrink any gap mechanically, so
    # the two kinds of cell are compared within bands of modal share.
    share = (truth.groupby(["survey", "target_code", "country"])["ground_truth"]
             .agg(modal_share=lambda s: s.value_counts(normalize=True).iloc[0])
             .reset_index())
    d2 = d.merge(share, on=["survey", "target_code", "country"], how="left")
    d2["concordant"] = d2["local_mode"] == d2["pooled_mode"]
    d2["holds_local"] = d2["ground_truth"] == d2["local_mode"]
    print("\nMajority/dissenter gap by mode concordance, within modal-share bands:")
    rows = []
    for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        for conc in (True, False):
            sub = d2[(d2["concordant"] == conc) & (d2["modal_share"] >= lo)
                     & (d2["modal_share"] < hi)]
            cells = sub.groupby(["survey", "target_code", "country"]).ngroups
            t = sub.groupby(["model", "holds_local"])["correct"].mean().unstack()
            gap = ((t[True] - t[False]).mean()
                   if cells >= 20 and True in t and False in t else float("nan"))
            rows.append({"band": f"{lo:.1f}-{hi:.1f}", "concordant": conc,
                         "cells": cells, "gap": gap})
    bands = pd.DataFrame(rows)
    print(bands.round(3).to_string(index=False))
    bands.to_csv(os.path.join(OUT, "gap_by_concordance.csv"), index=False)

    by.to_csv(os.path.join(OUT, "by_model.csv"))
    pd.DataFrame([{"n_cells": n_cells, "n_questions": n_q,
                   "share_modes_differ": share_differ,
                   "says_local": disc["says_local"].mean(),
                   "says_pooled": disc["says_pooled"].mean(),
                   "diff": point, "ci_lo": lo, "ci_hi": hi,
                   "n_models_favour_local": int((by["diff"] > 0).sum())}
                  ]).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    gaps.to_csv(os.path.join(OUT, "gap_by_split.csv"), index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
