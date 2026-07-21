"""Dissenter penalty: whose views disappear when models flatten?

For each (survey, target question, country) cell with >= MIN_N respondents at
rich profiles, we identify the cell's modal answer (the most common recorded
answer). Respondents are 'modal' if their answer is the cell mode and
'dissenters' otherwise. We then compare model accuracy on the two groups.

Heterogeneity flattening predicts a strong asymmetry: models that collapse
toward modal answers look adequate on modal respondents and fail on
dissenters. This quantifies the claim that minority-within-group views are
systematically misrepresented.

Because the dissenter share mechanically depends on how dominant the mode is,
results are also reported stratified by modal-share bins.

Outputs to analysis/equity_audit/:
  dissenter_penalty_by_model.csv     per model: acc modal vs dissenter
  dissenter_penalty_by_share.csv     same, stratified by modal-share bin
  dissenter_cells.csv                per-cell accuracy for both groups (pooled models)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
MR = ANALYSIS / "marginal_recovery"
OUT = ANALYSIS / "equity_audit"
OUT.mkdir(exist_ok=True)

MIN_N = 20
PROFILE = "s6m4"

MODELS = [
    "deepseek", "gemma3-27b", "gpt-oss",
    "llama3.1_70b_base", "llama3.1_70b_instruct",
    "llama3.1_8b_base", "llama3.1_8b_instruct",
    "olmo3_32b_base", "olmo3_32b_dpo",
    "olmo3_7b_base", "olmo3_7b_dpo",
    "qwen3-32b", "qwen3-4b",
]

SHARE_BINS = [0.0, 0.4, 0.6, 0.8, 1.01]
SHARE_LABELS = ["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"]


def load_n_options() -> pd.DataFrame:
    q = pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv")
    return q.groupby(["survey", "target_code"])["n_options"].first().reset_index()


def process_model(model: str, resp_country: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(
        ANALYSIS / model / "results_data.csv",
        usecols=["survey", "respondent_id", "target_code", "profile_type",
                 "ground_truth", "correct"],
        dtype={"survey": str, "respondent_id": str, "target_code": str,
               "profile_type": str, "ground_truth": str},
    )
    df = df[df["profile_type"] == PROFILE]
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    df = df.merge(resp_country, on=["survey", "respondent_id"], how="left")
    df = df[df["country"].notna()]

    # Cell mode and modal share from recorded answers
    grp = df.groupby(["survey", "target_code", "country"])
    stats = grp["ground_truth"].agg(
        n="size", mode=lambda s: s.mode().iloc[0],
        modal_n=lambda s: s.value_counts().iloc[0],
    ).reset_index()
    stats = stats[stats["n"] >= MIN_N]
    stats["modal_share"] = stats["modal_n"] / stats["n"]

    df = df.merge(stats[["survey", "target_code", "country", "mode", "modal_share"]],
                  on=["survey", "target_code", "country"], how="inner")
    df["is_modal"] = df["ground_truth"] == df["mode"]
    df["model"] = model
    return df[["survey", "target_code", "country", "model", "correct",
               "is_modal", "modal_share"]]


def main() -> None:
    resp_country = pd.read_csv(MR / "respondent_country.csv", dtype=str)
    nopt = load_n_options()

    frames = [process_model(m, resp_country) for m in MODELS]
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.merge(nopt, on=["survey", "target_code"], how="left")
    all_df["norm_acc"] = ((all_df["correct"].astype(float) - 1 / all_df["n_options"])
                          / (1 - 1 / all_df["n_options"]))
    print(f"instances in qualifying cells: {len(all_df):,} "
          f"({all_df['is_modal'].mean():.1%} modal)")

    by_model = (all_df.groupby(["model", "is_modal"])
                .agg(acc=("correct", "mean"), norm=("norm_acc", "mean"))
                .unstack())
    by_model.columns = [f"{a}_{'modal' if b else 'dissenter'}"
                        for a, b in by_model.columns]
    by_model["penalty_norm"] = by_model["norm_modal"] - by_model["norm_dissenter"]
    by_model = by_model.sort_values("norm_modal", ascending=False)
    by_model.to_csv(OUT / "dissenter_penalty_by_model.csv")

    all_df["share_bin"] = pd.cut(all_df["modal_share"], SHARE_BINS,
                                 labels=SHARE_LABELS, right=False)
    by_share = (all_df.groupby(["share_bin", "is_modal"], observed=True)["correct"]
                .agg(["mean", "size"]).unstack())
    by_share.to_csv(OUT / "dissenter_penalty_by_share.csv")

    cells = (all_df.groupby(["survey", "target_code", "country", "is_modal"])
             ["correct"].mean().unstack()
             .rename(columns={True: "acc_modal", False: "acc_dissenter"})
             .reset_index())
    cells.to_csv(OUT / "dissenter_cells.csv", index=False)

    # Respondent-free cluster bootstrap over cells for the pooled penalty
    rng = np.random.default_rng(42)
    cell_pen = (cells["acc_modal"] - cells["acc_dissenter"]).dropna().to_numpy()
    idx = rng.integers(0, len(cell_pen), size=(10_000, len(cell_pen)))
    boots = cell_pen[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    pooled_modal = all_df.loc[all_df["is_modal"], "correct"].mean()
    pooled_diss = all_df.loc[~all_df["is_modal"], "correct"].mean()
    print(f"\npooled accuracy: modal {pooled_modal:.3f} vs dissenter {pooled_diss:.3f}")
    print(f"cell-level mean penalty: {cell_pen.mean():.3f} [{lo:.3f}, {hi:.3f}] "
          f"over {len(cell_pen)} cells")
    print("\nby model:")
    print(by_model.round(3).to_string())
    print("\nby modal-share bin (acc means, then n):")
    print(by_share.round(3).to_string())


if __name__ == "__main__":
    main()
