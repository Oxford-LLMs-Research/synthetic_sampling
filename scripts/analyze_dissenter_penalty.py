"""Dissenter penalty: whose views disappear when models flatten?

A respondent is a 'dissenter' if their recorded answer is not the majority
answer. Which majority defines that split matters, and this computes both:

  question   the question's modal answer over all countries surveyed. This is
             what the models actually emit (analyze_local_vs_pooled_mode.py),
             so it is the split the mechanism claim is about, and it is the
             one the main-text figure uses.
  country    the modal answer within the respondent's (survey, question,
             country) cell. Retained as a comparison: it credits the models
             with local knowledge they turn out not to use, and the equity
             audit reads dissent in this sense as disagreeing with the people
             around you.

Because the dissenter share mechanically depends on how dominant the majority
answer is, results are also stratified by that share.

Outputs to analysis/equity_audit/, suffixed by split:
  dissenter_penalty_by_model{,_country}.csv   per model: acc majority vs dissenter
  dissenter_penalty_by_share{,_country}.csv   same, stratified by majority share
  dissenter_cells{,_country}.csv              per-cell accuracy for both groups
"""

from pathlib import Path

import numpy as np
import pandas as pd

from repair_ids import repair

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

# The country split groups within country; the question split pools over them.
GROUPS = {"country": ["survey", "target_code", "country"],
          "question": ["survey", "target_code"]}


def load_n_options() -> pd.DataFrame:
    q = pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv")
    return q.groupby(["survey", "target_code"])["n_options"].first().reset_index()


def load_model(model: str, resp_country: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(
        ANALYSIS / model / "results_data.csv",
        usecols=["example_id", "survey", "respondent_id", "target_code",
                 "profile_type", "ground_truth", "correct"],
        dtype={"example_id": str, "survey": str, "respondent_id": str,
               "target_code": str, "profile_type": str, "ground_truth": str},
    )
    df = df[df["profile_type"] == PROFILE]
    df = repair(df, verbose=False)
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    df = df.merge(resp_country, on=["survey", "respondent_id"], how="left")
    df = df[df["country"].notna()]
    df["model"] = model
    return df[["survey", "target_code", "country", "model", "correct",
               "ground_truth"]]


def apply_split(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """Attach the majority answer and its share for one definition of majority."""
    keys = GROUPS[split]
    one = raw[raw["model"] == MODELS[0]]          # answers do not vary by model
    stats = (one.groupby(keys)["ground_truth"]
             .agg(n="size", mode=lambda s: s.mode().iloc[0],
                  modal_n=lambda s: s.value_counts().iloc[0])
             .reset_index())
    # The country split needs enough respondents per cell to name a mode; the
    # question split pools every country, so every question qualifies.
    if split == "country":
        stats = stats[stats["n"] >= MIN_N]
    stats["modal_share"] = stats["modal_n"] / stats["n"]

    df = raw.merge(stats[keys + ["mode", "modal_share"]], on=keys, how="inner")
    df["is_modal"] = df["ground_truth"] == df["mode"]
    return df


def summarise(all_df: pd.DataFrame, split: str) -> None:
    suffix = "" if split == "question" else "_country"
    keys = GROUPS[split]

    by_model = (all_df.groupby(["model", "is_modal"])
                .agg(acc=("correct", "mean"), norm=("norm_acc", "mean"))
                .unstack())
    by_model.columns = [f"{a}_{'modal' if b else 'dissenter'}"
                        for a, b in by_model.columns]
    by_model["penalty_norm"] = by_model["norm_modal"] - by_model["norm_dissenter"]
    by_model = by_model.sort_values("norm_modal", ascending=False)
    by_model.to_csv(OUT / f"dissenter_penalty_by_model{suffix}.csv")

    all_df["share_bin"] = pd.cut(all_df["modal_share"], SHARE_BINS,
                                 labels=SHARE_LABELS, right=False)
    by_share = (all_df.groupby(["share_bin", "is_modal"], observed=True)
                .agg(acc=("correct", "mean"), norm=("norm_acc", "mean"),
                     n=("correct", "size")).unstack())
    by_share.to_csv(OUT / f"dissenter_penalty_by_share{suffix}.csv")

    cells = (all_df.groupby(keys + ["is_modal"])["correct"].mean().unstack()
             .rename(columns={True: "acc_modal", False: "acc_dissenter"})
             .reset_index())
    cells.to_csv(OUT / f"dissenter_cells{suffix}.csv", index=False)

    rng = np.random.default_rng(42)
    cell_pen = (cells["acc_modal"] - cells["acc_dissenter"]).dropna().to_numpy()
    idx = rng.integers(0, len(cell_pen), size=(10_000, len(cell_pen)))
    boots = cell_pen[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n{'=' * 62}\nSPLIT: {split} majority   "
          f"({all_df['is_modal'].mean():.1%} in the majority group, "
          f"{len(all_df):,} instances)")
    print(f"pooled accuracy: majority "
          f"{all_df.loc[all_df['is_modal'], 'correct'].mean():.3f} vs dissenter "
          f"{all_df.loc[~all_df['is_modal'], 'correct'].mean():.3f}")
    print(f"normalized penalty, 13-model mean: {by_model['penalty_norm'].mean():.3f}"
          f"   range {by_model['penalty_norm'].min():.3f} to "
          f"{by_model['penalty_norm'].max():.3f}")
    print(f"cell-level mean penalty (raw): {cell_pen.mean():.3f} "
          f"[{lo:.3f}, {hi:.3f}] over {len(cell_pen):,} cells")
    print("\nby model:")
    print(by_model.round(3).to_string())
    print("\nby majority-share bin:")
    print(by_share.round(3).to_string())


def main() -> None:
    resp_country = pd.read_csv(MR / "respondent_country.csv", dtype=str,
                               keep_default_na=False)
    nopt = load_n_options()
    raw = pd.concat([load_model(m, resp_country) for m in MODELS],
                    ignore_index=True)

    for split in ("question", "country"):
        df = apply_split(raw, split).merge(nopt, on=["survey", "target_code"],
                                           how="left")
        df["norm_acc"] = ((df["correct"].astype(float) - 1 / df["n_options"])
                          / (1 - 1 / df["n_options"]))
        summarise(df, split)


if __name__ == "__main__":
    main()
