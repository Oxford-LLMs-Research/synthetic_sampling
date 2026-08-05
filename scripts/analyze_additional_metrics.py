"""Model metrics the paper claims but never tabulates.

Four blocks, all at rich profiles (s6m4) unless noted:

  1. Per-survey normalized accuracy, 7 surveys x 13 models. Reviewers asked
     whether pooling across heterogeneous instruments hides variation.
  2. Accuracy by number of response options and by response format. This is the
     direct check on the objection that averaging over questions with different
     option counts is misleading: normalized accuracy should be roughly flat in
     the option count, raw accuracy should not.
  3. Macro-F1, Brier, top-2 and top-3 accuracy, read from each model's
     summary.json. The main text claims Macro-F1 improves for all 13 models and
     Brier for 12 of 13 with no table anywhere; this produces it.
  4. "Don't know" hedging and yes/no bias, which back the topic-variation
     claims. These had no persisted artifact at all: the numbers in the draft
     came from an ad hoc inspection.

A prediction counts as a "don't know" if it begins with one of the hedging
phrases in DK_PATTERN (the survey option text itself, e.g. "Don't know",
"Don't know/Haven't heard"). Matching is anchored so that substantive answers
containing the words are not swept in.

Anchoring alone was not enough. "Don't know him" is a substantive answer to
questions asking how much a respondent knows about a named figure, and it
begins with a hedging phrase, so DK_PATTERN swept in 19k of them. That put the
model rates in this table on a wider base than the human rate quoted beside
them, overstating the models' hedging by 0.2-1.2 points and making the human
rate look 0.7 points higher than the 1.1% reported in the appendix text.
DK_EXCLUDE removes it, and the two sides now share a base. It is the only
substantive string DK_PATTERN matches: the other six are all genuine
non-response options.

Outputs to analysis/additional_metrics/:
  by_survey.csv  by_n_options.csv  headline_metrics.csv
  dk_hedging.csv  dk_targets.csv  yes_no_bias.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from repair_ids import repair

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
OUT = ANALYSIS / "additional_metrics"
OUT.mkdir(parents=True, exist_ok=True)

PROFILE = "s6m4"
DK_THRESHOLD = 0.80

MODELS = {
    "qwen3-32b": "Qwen 3 32B", "deepseek": "DeepSeek-V3.1",
    "llama3.1_8b_instruct": "Llama 3.1 8B inst.", "gpt-oss": "GPT-OSS 120B",
    "llama3.1_70b_instruct": "Llama 3.1 70B inst.", "olmo3_7b_dpo": "OLMo 3 7B inst.",
    "gemma3-27b": "Gemma 3 27B", "qwen3-4b": "Qwen 3 4B",
    "olmo3_32b_dpo": "OLMo 3 32B inst.", "olmo3_7b_base": "OLMo 3 7B base",
    "olmo3_32b_base": "OLMo 3 32B base", "llama3.1_70b_base": "Llama 3.1 70B base",
    "llama3.1_8b_base": "Llama 3.1 8B base",
}

SURVEY_NAMES = {
    "wvs": "World Values Survey", "ess_wave_10": "ESS Wave 10",
    "ess_wave_11": "ESS Wave 11", "afrobarometer": "Afrobarometer",
    "arabbarometer": "Arab Barometer", "asianbarometer": "Asian Barometer",
    "latinobarometer": "Latinobar\\'{o}metro",
}

DK_PATTERN = re.compile(
    r"^\s*(don.?t know|do not know|can.?t choose|cannot choose|not sure|"
    r"no opinion|haven.?t heard|don.?t care)", re.IGNORECASE)

# "Don't know him" answers a question about a named figure; it states a
# position rather than declining to. See the module docstring.
DK_EXCLUDE = re.compile(r"^\s*(don.?t|do not) know (him|her|them|what)\b",
                        re.IGNORECASE)


def is_dk(s: pd.Series) -> pd.Series:
    return s.str.match(DK_PATTERN) & ~s.str.match(DK_EXCLUDE)


def load_n_options() -> pd.DataFrame:
    q = pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv")
    return (q.groupby(["survey", "target_code"])["n_options"].first()
            .reset_index())


def norm(correct: pd.Series, n_options: pd.Series) -> pd.Series:
    chance = 1.0 / n_options
    return (correct - chance) / (1.0 - chance)


def headline_metrics() -> pd.DataFrame:
    """Macro-F1, Brier, top-k from summary.json, sparse and rich."""
    rows = []
    for key, disp in MODELS.items():
        s = json.loads((ANALYSIS / key / "summary.json").read_text())
        sparse, rich = s["by_profile_type"]["s3m2"], s["by_profile_type"]["s6m4"]
        rows.append({
            "model": disp,
            "accuracy": rich["accuracy"],
            "top2": rich["top2_accuracy"], "top3": rich["top3_accuracy"],
            "macro_f1": rich["macro_f1"], "brier": rich["brier_score"],
            "macro_f1_sparse": sparse["macro_f1"], "brier_sparse": sparse["brier_score"],
        })
    df = pd.DataFrame(rows)
    df["macro_f1_delta"] = df["macro_f1"] - df["macro_f1_sparse"]
    df["brier_delta"] = df["brier"] - df["brier_sparse"]
    return df.sort_values("accuracy", ascending=False)


def main() -> None:
    nopt = load_n_options()
    survey_rows, nopt_rows, dk_rows, yn_rows = [], [], [], []
    dk_target_rows = []

    for key, disp in MODELS.items():
        df = pd.read_csv(
            ANALYSIS / key / "results_data.csv",
            usecols=["example_id", "survey", "respondent_id", "target_code",
                     "profile_type", "ground_truth", "predicted", "correct"],
            dtype={"example_id": str, "survey": str, "respondent_id": str,
                   "target_code": str, "profile_type": str,
                   "ground_truth": str, "predicted": str},
            encoding="utf-8", encoding_errors="replace", low_memory=False)
        df = df[df["profile_type"] == PROFILE]
        df = repair(df, verbose=False)
        df["correct"] = df["correct"].astype(str).str.lower().eq("true").astype(float)
        df = df.merge(nopt, on=["survey", "target_code"], how="left")
        df = df[df["n_options"] > 1]
        df["norm_acc"] = norm(df["correct"], df["n_options"])

        for survey, g in df.groupby("survey"):
            survey_rows.append({"model": disp, "survey": survey, "n": len(g),
                                "raw_acc": g["correct"].mean(),
                                "norm_acc": g["norm_acc"].mean()})

        for n_opt, g in df.groupby("n_options"):
            nopt_rows.append({"model": disp, "n_options": int(n_opt), "n": len(g),
                              "raw_acc": g["correct"].mean(),
                              "norm_acc": g["norm_acc"].mean()})

        # --- Don't know hedging -------------------------------------------
        df["pred_dk"] = is_dk(df["predicted"].fillna(""))
        df["true_dk"] = is_dk(df["ground_truth"].fillna(""))
        per_target = df.groupby(["survey", "target_code"]).agg(
            pred_dk=("pred_dk", "mean"), true_dk=("true_dk", "mean"),
            n=("pred_dk", "size")).reset_index()
        # Only targets that actually offer a DK option can be hedged toward.
        offers_dk = per_target[per_target["pred_dk"] > 0]
        hedged = per_target[per_target["pred_dk"] >= DK_THRESHOLD]
        dk_rows.append({
            "model": disp,
            "pred_dk_rate": df["pred_dk"].mean(),
            "true_dk_rate": df["true_dk"].mean(),
            "n_targets_offering_dk": len(offers_dk),
            "n_targets_hedged": len(hedged),
            "mean_true_dk_on_hedged": hedged["true_dk"].mean() if len(hedged) else np.nan,
        })
        for _, r in hedged.iterrows():
            dk_target_rows.append({"model": disp, "survey": r["survey"],
                                   "target_code": r["target_code"],
                                   "pred_dk": r["pred_dk"], "true_dk": r["true_dk"]})

        # --- Yes/No bias on binary yes/no items ---------------------------
        opts = df.groupby(["survey", "target_code"])["ground_truth"].unique()
        yn_keys = [k for k, v in opts.items()
                   if {str(x).strip().lower() for x in v} <= {"yes", "no"}
                   and len({str(x).strip().lower() for x in v}) == 2]
        if yn_keys:
            idx = pd.MultiIndex.from_tuples(yn_keys, names=["survey", "target_code"])
            yn = df.set_index(["survey", "target_code"]).loc[idx].reset_index()
            yn_rows.append({
                "model": disp, "n_questions": len(yn_keys), "n": len(yn),
                "pred_yes": yn["predicted"].str.strip().str.lower().eq("yes").mean(),
                "true_yes": yn["ground_truth"].str.strip().str.lower().eq("yes").mean(),
                "accuracy": yn["correct"].mean(),
            })
        print(f"  processed {disp}")

    pd.DataFrame(survey_rows).to_csv(OUT / "by_survey.csv", index=False)
    pd.DataFrame(nopt_rows).to_csv(OUT / "by_n_options.csv", index=False)
    pd.DataFrame(dk_rows).to_csv(OUT / "dk_hedging.csv", index=False)
    pd.DataFrame(dk_target_rows).to_csv(OUT / "dk_targets.csv", index=False)
    pd.DataFrame(yn_rows).to_csv(OUT / "yes_no_bias.csv", index=False)
    head = headline_metrics()
    head.to_csv(OUT / "headline_metrics.csv", index=False)

    print("\n=== headline metrics (rich profiles) ===")
    print(head[["model", "accuracy", "top2", "top3", "macro_f1", "brier"]]
          .round(3).to_string(index=False))
    print(f"\nMacro-F1 improves sparse->rich: {(head['macro_f1_delta'] > 0).sum()}/13")
    print(f"Brier improves (decreases) sparse->rich: {(head['brier_delta'] < 0).sum()}/13")
    print("exceptions:", head.loc[head["brier_delta"] >= 0, "model"].tolist())

    sv = pd.DataFrame(survey_rows).pivot_table(index="survey", values="norm_acc",
                                               aggfunc=["mean", "min", "max"])
    print("\n=== normalized accuracy by survey (across 13 models) ===")
    print(sv.round(3).to_string())

    no = pd.DataFrame(nopt_rows).groupby("n_options")[["raw_acc", "norm_acc"]].mean()
    no["n_instances"] = pd.DataFrame(nopt_rows).groupby("n_options")["n"].sum() / 13
    print("\n=== accuracy by option count (13-model mean) ===")
    print(no.round(3).to_string())

    print("\n=== Don't know hedging ===")
    print(pd.DataFrame(dk_rows).round(3).to_string(index=False))
    print("\n=== yes/no bias ===")
    print(pd.DataFrame(yn_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
