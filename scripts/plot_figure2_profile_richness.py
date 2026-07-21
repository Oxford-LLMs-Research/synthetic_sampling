#!/usr/bin/env python3
"""
Figure 2: Profile richness — normalized accuracy across s3m2 → s4m3 → s6m4.

Line plot with one trajectory per model; XGBoost and majority-class reference lines.

Sources:
  analysis/normalized_accuracy/per_question_norm_acc.csv
  analysis/xgboost_baseline/results_merged.csv
  analysis/normalized_accuracy/majority_class_norm_acc.csv
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
NORM_ACC_PATH = ROOT / "analysis/normalized_accuracy/per_question_norm_acc.csv"
XGB_PATH = ROOT / "analysis/xgboost_baseline/results_merged.csv"
MAJ_PATH = ROOT / "analysis/normalized_accuracy/majority_class_norm_acc.csv"
OUT_DIR = ROOT / "analysis/figures/emnlp_revision"
LATEX_FIG_DIR = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
                 "/latex/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Display names and families for coloring
MODEL_META = {
    "deepseek":              ("DeepSeek-V3",          "DeepSeek",  "#1f77b4"),
    "gemma3-27b":            ("Gemma 3 27B",           "Gemma",     "#aec7e8"),
    "gpt-oss":               ("GPT-OSS 120B",          "GPT-OSS",   "#ff7f0e"),
    "llama3.1_70b_base":     ("Llama 3.1 70B Base",    "Llama",     "#2ca02c"),
    "llama3.1_70b_instruct": ("Llama 3.1 70B Inst.",   "Llama",     "#98df8a"),
    "llama3.1_8b_base":      ("Llama 3.1 8B Base",     "Llama",     "#d62728"),
    "llama3.1_8b_instruct":  ("Llama 3.1 8B Inst.",    "Llama",     "#ff9896"),
    "olmo3_32b_base":        ("OLMo 3 32B Base",       "OLMo",      "#9467bd"),
    "olmo3_32b_dpo":         ("OLMo 3 32B Inst.",      "OLMo",      "#c5b0d5"),
    "olmo3_7b_base":         ("OLMo 3 7B Base",        "OLMo",      "#8c564b"),
    "olmo3_7b_dpo":          ("OLMo 3 7B Inst.",       "OLMo",      "#c49c94"),
    "qwen3-32b":             ("Qwen 3 32B",            "Qwen",      "#e377c2"),
    "qwen3-4b":              ("Qwen 3 4B",             "Qwen",      "#f7b6d2"),
}

PROFILE_LABELS = {"s3m2": "Sparse\n(6 feat.)", "s4m3": "Medium\n(12 feat.)", "s6m4": "Rich\n(24 feat.)"}
PROFILE_ORDER = ["s3m2", "s4m3", "s6m4"]
PROFILE_X = {p: i for i, p in enumerate(PROFILE_ORDER)}

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    # ---- load per-question norm acc ----
    df = pd.read_csv(NORM_ACC_PATH)

    # Aggregate mean norm_acc by (model, profile_type)
    agg = (df.groupby(["model", "profile_type"])["norm_acc"]
             .mean()
             .reset_index()
             .rename(columns={"norm_acc": "mean_norm_acc"}))

    # ---- XGBoost reference (s6m4 mean across all surveys) ----
    xgb_norm_acc = None
    if XGB_PATH.exists():
        xgb = pd.read_csv(XGB_PATH)
        xgb_s6m4 = xgb[xgb["profile_type"] == "s6m4"]["xgb_norm_acc"].mean()
        xgb_norm_acc = xgb_s6m4
        print(f"XGBoost s6m4 norm_acc: {xgb_norm_acc:.4f}")

    # ---- Majority-class reference (s6m4) ----
    maj_norm_acc = None
    if MAJ_PATH.exists():
        maj = pd.read_csv(MAJ_PATH)
        maj_s6m4 = maj[maj["profile_type"] == "s6m4"]["majority_norm_acc"].mean()
        maj_norm_acc = maj_s6m4
        print(f"Majority-class s6m4 norm_acc: {maj_norm_acc:.4f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for model, (display, family, color) in MODEL_META.items():
        rows = agg[agg["model"] == model].copy()
        if rows.empty:
            continue
        rows["x"] = rows["profile_type"].map(PROFILE_X)
        rows = rows.sort_values("x")
        xs = rows["x"].tolist()
        ys = rows["mean_norm_acc"].tolist()
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.85, marker="o",
                markersize=3.5, label=display)

    # Reference lines at right edge (x=2 = s6m4)
    x_ref = PROFILE_X["s6m4"]
    if maj_norm_acc is not None:
        ax.axhline(maj_norm_acc, color="black", linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"Majority class ({maj_norm_acc:.3f})")
    if xgb_norm_acc is not None:
        ax.axhline(xgb_norm_acc, color="black", linestyle="-.", linewidth=1.0,
                   alpha=0.7, label=f"XGBoost ({xgb_norm_acc:.3f})")

    # Axes
    ax.set_xticks(list(PROFILE_X.values()))
    ax.set_xticklabels([PROFILE_LABELS[p] for p in PROFILE_ORDER], fontsize=9)
    ax.set_ylabel("Normalized Accuracy", fontsize=10)
    ax.set_xlim(-0.25, 2.25)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)

    # Legend: split into two columns to keep it compact
    legend = ax.legend(
        loc="upper left",
        fontsize=7,
        ncol=2,
        frameon=True,
        framealpha=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.4,
    )
    legend.get_frame().set_linewidth(0.4)

    plt.tight_layout(pad=0.5)

    out_base = OUT_DIR / "figure2_profile_richness"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_base}.pdf/.png")

    # Copy to latex figures
    if LATEX_FIG_DIR.exists():
        dest = LATEX_FIG_DIR / "figure_profile_richness.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")


if __name__ == "__main__":
    main()
