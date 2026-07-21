"""Equity audit: prediction quality by demographic subgroup.

Joins harmonized demographics (extract_respondent_demographics.py) onto the
rich-profile evaluation results and asks: for whom are synthetic respondents
least accurate?

Three layers:
  1. Descriptive: mean normalized accuracy by gender / age band / education /
     urban-rural, pooled over the 13 models (per-model values saved too).
  2. Adjusted: the same contrasts within (survey, question, country) cells,
     removing composition differences (which questions and which countries a
     subgroup answers). Implemented by demeaning within cells, i.e. a fixed-
     effects estimate; CIs from a respondent-cluster bootstrap.
  3. Dissenter intersection: each subgroup's share of dissenters (respondents
     whose answer differs from their question x country modal answer) and the
     dissenter penalty within the subgroup.

Outputs to analysis/equity_audit/:
  subgroup_descriptive.csv, subgroup_descriptive_by_model.csv,
  subgroup_adjusted.csv, subgroup_dissenter.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
MR = ANALYSIS / "marginal_recovery"
OUT = ANALYSIS / "equity_audit"

PROFILE = "s6m4"
MIN_N = 20
N_BOOT = 1000
SEED = 42

MODELS = [
    "deepseek", "gemma3-27b", "gpt-oss",
    "llama3.1_70b_base", "llama3.1_70b_instruct",
    "llama3.1_8b_base", "llama3.1_8b_instruct",
    "olmo3_32b_base", "olmo3_32b_dpo",
    "olmo3_7b_base", "olmo3_7b_dpo",
    "qwen3-32b", "qwen3-4b",
]

DIMS = {
    "gender": ["male", "female"],
    "age_band": ["18-29", "30-49", "50+"],
    "education": ["low", "mid", "high"],
    "urban": ["urban", "rural"],
}


def build_instance_table() -> pd.DataFrame:
    """One row per (survey, respondent, target) at rich profiles with the
    13-model mean correctness, plus per-model correctness columns."""
    base = None
    for m in MODELS:
        df = pd.read_csv(
            ANALYSIS / m / "results_data.csv",
            usecols=["survey", "respondent_id", "target_code", "profile_type",
                     "ground_truth", "correct"],
            dtype={"survey": str, "respondent_id": str, "target_code": str,
                   "profile_type": str, "ground_truth": str},
        )
        df = df[df["profile_type"] == PROFILE].drop(columns="profile_type")
        df["correct"] = df["correct"].astype(str).str.lower().eq("true").astype(np.float32)
        df = df.rename(columns={"correct": f"c_{m}"})
        if base is None:
            base = df
        else:
            base = base.merge(df.drop(columns="ground_truth"),
                              on=["survey", "respondent_id", "target_code"],
                              how="inner")
        print(f"  merged {m}: {len(base):,} rows")

    model_cols = [f"c_{m}" for m in MODELS]
    base["mean_correct"] = base[model_cols].mean(axis=1)

    nopt = (pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv")
            .groupby(["survey", "target_code"])["n_options"].first().reset_index())
    base = base.merge(nopt, on=["survey", "target_code"], how="left")
    base["norm_acc"] = ((base["mean_correct"] - 1 / base["n_options"])
                        / (1 - 1 / base["n_options"]))

    rc = pd.read_csv(MR / "respondent_country.csv", dtype=str)
    base = base.merge(rc, on=["survey", "respondent_id"], how="left")
    demo = pd.read_csv(OUT / "respondent_demographics.csv", dtype=str)
    base = base.merge(demo, on=["survey", "respondent_id"], how="left")
    return base


def descriptive(df: pd.DataFrame) -> None:
    rows, rows_pm = [], []
    for dim, levels in DIMS.items():
        for lv in levels:
            sub = df[df[dim] == lv]
            rows.append({"dim": dim, "level": lv, "n_instances": len(sub),
                         "norm_acc": sub["norm_acc"].mean()})
            for m in MODELS:
                col = f"c_{m}"
                norm = ((sub[col] - 1 / sub["n_options"])
                        / (1 - 1 / sub["n_options"])).mean()
                rows_pm.append({"dim": dim, "level": lv, "model": m,
                                "norm_acc": norm})
    pd.DataFrame(rows).to_csv(OUT / "subgroup_descriptive.csv", index=False)
    pd.DataFrame(rows_pm).to_csv(OUT / "subgroup_descriptive_by_model.csv", index=False)
    print("\ndescriptive (13-model mean normalized accuracy):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def adjusted(df: pd.DataFrame) -> None:
    """Within-(survey, target, country) contrasts vs the reference level.

    CIs from a respondent-cluster bootstrap: respondents are resampled with
    multinomial weights and each contributes their per-level (sum, count)
    aggregate, so a draw is a weighted version of the point estimate.
    """
    rng = np.random.default_rng(SEED)
    df = df[df["country"].notna()].copy()
    cell = ["survey", "target_code", "country"]
    df["cell_mean"] = df.groupby(cell)["norm_acc"].transform("mean")
    df["cell_n"] = df.groupby(cell)["norm_acc"].transform("size")
    df = df[df["cell_n"] >= MIN_N]
    df["dev"] = df["norm_acc"] - df["cell_mean"]

    rows = []
    for dim, levels in DIMS.items():
        ref = levels[0]
        sub = df[df[dim].notna()]
        # Per-respondent aggregates; a respondent has one level per dimension
        agg = (sub.groupby(["survey", "respondent_id", dim], observed=True)["dev"]
               .agg(["sum", "size"]).reset_index())
        lv_arr = agg[dim].to_numpy()
        S = agg["sum"].to_numpy()
        n = agg["size"].to_numpy()
        R = len(agg)
        W = rng.multinomial(R, np.full(R, 1 / R), size=N_BOOT)  # (N_BOOT, R)

        def level_mean(weights, level):
            mask = lv_arr == level
            return (weights[:, mask] @ S[mask]) / (weights[:, mask] @ n[mask])

        ref_point = S[lv_arr == ref].sum() / n[lv_arr == ref].sum()
        ref_boot = level_mean(W, ref)
        for lv in levels[1:]:
            est = S[lv_arr == lv].sum() / n[lv_arr == lv].sum() - ref_point
            boots = level_mean(W, lv) - ref_boot
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"dim": dim, "contrast": f"{lv} - {ref}",
                         "estimate": est, "ci_lo": lo, "ci_hi": hi})
            print(f"adjusted {dim}: {lv} - {ref} = {est:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(rows).to_csv(OUT / "subgroup_adjusted.csv", index=False)


def dissenter_intersection(df: pd.DataFrame) -> None:
    df = df[df["country"].notna()].copy()
    cell = ["survey", "target_code", "country"]
    stats = (df.groupby(cell)["ground_truth"]
             .agg(n="size", mode=lambda s: s.mode().iloc[0]).reset_index())
    stats = stats[stats["n"] >= MIN_N]
    df = df.merge(stats[cell + ["mode"]], on=cell, how="inner")
    df["is_modal"] = df["ground_truth"] == df["mode"]

    rows = []
    for dim, levels in DIMS.items():
        for lv in levels:
            sub = df[df[dim] == lv]
            rows.append({
                "dim": dim, "level": lv,
                "dissenter_share": 1 - sub["is_modal"].mean(),
                "norm_acc_modal": sub.loc[sub["is_modal"], "norm_acc"].mean(),
                "norm_acc_dissenter": sub.loc[~sub["is_modal"], "norm_acc"].mean(),
            })
    out = pd.DataFrame(rows)
    out["penalty"] = out["norm_acc_modal"] - out["norm_acc_dissenter"]
    out.to_csv(OUT / "subgroup_dissenter.csv", index=False)
    print("\ndissenter intersection:")
    print(out.round(3).to_string(index=False))


def main() -> None:
    df = build_instance_table()
    print(f"\ninstance table: {len(df):,} rows")
    descriptive(df)
    adjusted(df)
    dissenter_intersection(df)


if __name__ == "__main__":
    main()
