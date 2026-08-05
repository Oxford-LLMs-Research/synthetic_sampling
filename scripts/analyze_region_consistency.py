"""Consistency of the regional performance hierarchy across models.

The draft asserted that the regional ranking agrees across models at r > 0.95.
No code computed that number. This script measures it.

For each model we take its vector of per-region mean accuracy over the 19 world
regions and correlate it (Spearman) with every other model's vector. We report
the distribution of the 78 pairwise correlations, the same statistic aggregated
to model families, and the per-model mean agreement, which identifies outliers.

Both metrics are computed:
  raw         per-region raw accuracy, from the disaggregated analysis
  normalized  per-region normalized accuracy, filtered exactly as Figure 3(a)

Normalized is the metric the paper reports, so it is the one to quote.

Outputs to analysis/region_consistency/:
  pairwise_<metric>.csv   one row per model pair
  summary.csv             distribution summaries for both metrics and levels
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
BY_REGION = ROOT / "synthetic_sampling/analysis/disaggregated/by_region.json"
MEM_DATA = ROOT / "synthetic_sampling/analysis/mixed_effects/mixed_effects_data.csv"
OUT = ROOT / "analysis/region_consistency"
OUT.mkdir(parents=True, exist_ok=True)

# by_region.json and mixed_effects_data.csv use different model spellings.
CANON = {
    "deepseek-v3p1-terminus": "deepseek", "deepseek": "deepseek",
    "gemma-3-27b-instruct": "gemma3-27b", "gemma3-27b": "gemma3-27b",
    "gpt_oss": "gpt-oss", "gpt-oss": "gpt-oss",
    "llama3.1-70b-base": "llama3.1_70b_base", "llama3.1_70b_base": "llama3.1_70b_base",
    "llama3.1-70b-instruct": "llama3.1_70b_instruct",
    "llama3.1_70b_instruct": "llama3.1_70b_instruct",
    "llama3.1-8b-base": "llama3.1_8b_base", "llama3.1_8b_base": "llama3.1_8b_base",
    "llama3.1-8b-instruct": "llama3.1_8b_instruct",
    "llama3.1_8b_instruct": "llama3.1_8b_instruct",
    "olmo3-32b-base": "olmo3_32b_base", "olmo3_32b_base": "olmo3_32b_base",
    "olmo3-32b-dpo": "olmo3_32b_dpo", "olmo3_32b_dpo": "olmo3_32b_dpo",
    "olmo3-7b-base": "olmo3_7b_base", "olmo3_7b_base": "olmo3_7b_base",
    "olmo3-7b-dpo": "olmo3_7b_dpo", "olmo3_7b_dpo": "olmo3_7b_dpo",
    "qwen3-32b": "qwen3-32b", "qwen3-4b": "qwen3-4b",
}

FAMILY = {
    "deepseek": "DeepSeek", "gemma3-27b": "Gemma", "gpt-oss": "GPT-OSS",
    "llama3.1_70b_base": "Llama", "llama3.1_70b_instruct": "Llama",
    "llama3.1_8b_base": "Llama", "llama3.1_8b_instruct": "Llama",
    "olmo3_32b_base": "OLMo", "olmo3_32b_dpo": "OLMo",
    "olmo3_7b_base": "OLMo", "olmo3_7b_dpo": "OLMo",
    "qwen3-32b": "Qwen", "qwen3-4b": "Qwen",
}

DISPLAY = {
    "deepseek": "DeepSeek-V3.1", "gemma3-27b": "Gemma 3 27B", "gpt-oss": "GPT-OSS 120B",
    "llama3.1_70b_base": "Llama 3.1 70B base",
    "llama3.1_70b_instruct": "Llama 3.1 70B inst.",
    "llama3.1_8b_base": "Llama 3.1 8B base",
    "llama3.1_8b_instruct": "Llama 3.1 8B inst.",
    "olmo3_32b_base": "OLMo 3 32B base", "olmo3_32b_dpo": "OLMo 3 32B inst.",
    "olmo3_7b_base": "OLMo 3 7B base", "olmo3_7b_dpo": "OLMo 3 7B inst.",
    "qwen3-32b": "Qwen 3 32B", "qwen3-4b": "Qwen 3 4B",
}


def raw_matrix() -> pd.DataFrame:
    """region x model matrix of raw accuracy."""
    data = json.loads(BY_REGION.read_text())
    rows = []
    for model, regions in data.items():
        for region, stats in regions.items():
            if region == "Unknown":
                continue
            rows.append({"model": CANON[model], "region": region,
                         "value": stats["accuracy"]})
    return pd.DataFrame(rows).pivot(index="region", columns="model", values="value")


def normalized_matrix() -> pd.DataFrame:
    """region x model matrix of normalized accuracy, filtered as in Figure 3(a)."""
    df = pd.read_csv(MEM_DATA, encoding="latin-1",
                     usecols=["model", "correct", "region", "n_options"])
    df = df[df["region"] != "Unknown"]
    df["n_options"] = pd.to_numeric(df["n_options"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct", "n_options"])
    df = df[df["n_options"] > 1]
    chance = 1.0 / df["n_options"]
    df["norm_acc"] = (df["correct"] - chance) / (1.0 - chance)
    df["model"] = df["model"].map(CANON)
    agg = df.groupby(["region", "model"])["norm_acc"].mean().reset_index()
    return agg.pivot(index="region", columns="model", values="norm_acc")


def pairwise(mat: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for a, b in combinations(sorted(mat.columns), 2):
        rho = spearmanr(mat[a], mat[b]).statistic
        rows.append({"level": label, "a": a, "b": b, "spearman": rho})
    return pd.DataFrame(rows)


def describe(pw: pd.DataFrame, metric: str, level: str) -> dict:
    r = pw["spearman"]
    return {
        "metric": metric, "level": level, "n_pairs": len(r),
        "mean": r.mean(), "median": r.median(), "min": r.min(), "max": r.max(),
        "pct_above_095": (r > 0.95).mean() * 100,
    }


def main() -> None:
    summaries, pairwise_frames = [], []

    for metric, mat in [("raw", raw_matrix()), ("normalized", normalized_matrix())]:
        print(f"\n=== {metric} accuracy: {mat.shape[0]} regions x {mat.shape[1]} models ===")

        pw = pairwise(mat, "model")
        pw["metric"] = metric
        summaries.append(describe(pw, metric, "model"))
        pairwise_frames.append(pw)

        fam = mat.T.groupby(mat.columns.map(FAMILY)).mean().T
        pwf = pairwise(fam, "family")
        pwf["metric"] = metric
        summaries.append(describe(pwf, metric, "family"))
        pairwise_frames.append(pwf)

        # Per-model mean agreement with the other twelve: identifies outliers.
        agree = {}
        for m in mat.columns:
            sub = pw[(pw["a"] == m) | (pw["b"] == m)]
            agree[m] = sub["spearman"].mean()
        agree = pd.Series(agree).sort_values()
        print("per-model mean agreement with the other 12 (ascending):")
        for m, v in agree.items():
            print(f"  {DISPLAY[m]:<22} {v:.3f}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(pairwise_frames).to_csv(OUT / "pairwise.csv", index=False)

    print("\n=== summary (this is what goes in the paper) ===")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
