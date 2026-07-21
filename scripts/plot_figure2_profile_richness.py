#!/usr/bin/env python3
"""
Figure 2: Profile richness — normalized accuracy, variance ratio, and JSD
across sparse (6 feat.) → medium (12 feat.) → rich (24 feat.) profiles.

Three-panel layout matching Figure 1's structure:
  (a) Normalized accuracy — with majority class and XGBoost reference lines
  (b) Variance ratio       — with VR=1.0 (human level) reference
  (c) Jensen-Shannon Divergence — lower = closer to human distribution

Lines: colored by model family (same palette as Figure 1).
       Solid = instruct/fine-tuned,  Dashed = base model.
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NORM_ACC_PATH = ROOT / "analysis/normalized_accuracy/per_question_norm_acc.csv"
XGB_PATH      = ROOT / "analysis/xgboost_baseline/results_merged.csv"
MAJ_PATH      = ROOT / "analysis/normalized_accuracy/majority_class_norm_acc.csv"
VR_JSD_PATH   = ROOT / "analysis/figures/emnlp_revision/vr_jsd_by_model.csv"
OUT_DIR       = ROOT / "analysis/figures/emnlp_revision"
LATEX_FIG_DIR = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
                 "/latex/figures")

PROFILE_ORDER  = ["s3m2", "s4m3", "s6m4"]
PROFILE_X      = {p: i for i, p in enumerate(PROFILE_ORDER)}
PROFILE_LABELS = {"s3m2": "Sparse\n(6 feat.)", "s4m3": "Medium\n(12 feat.)", "s6m4": "Rich\n(24 feat.)"}

# Same family color palette as Figure 1
FAMILY_COLORS = {
    "Llama":    "#2E86AB",
    "OLMo":     "#A23B72",
    "Qwen":     "#F18F01",
    "GPT-OSS":  "#C73E1D",
    "DeepSeek": "#6A994E",
    "Gemma":    "#BC4749",
}

# (display_name, family, is_instruct)
MODEL_META = {
    "llama3.1_8b_base":      ("Llama 3.1 8B base",   "Llama",    False),
    "llama3.1_8b_instruct":  ("Llama 3.1 8B inst.",  "Llama",    True),
    "llama3.1_70b_base":     ("Llama 3.1 70B base",  "Llama",    False),
    "llama3.1_70b_instruct": ("Llama 3.1 70B inst.", "Llama",    True),
    "olmo3_7b_base":         ("OLMo 3 7B base",      "OLMo",     False),
    "olmo3_7b_dpo":          ("OLMo 3 7B inst.",     "OLMo",     True),
    "olmo3_32b_base":        ("OLMo 3 32B base",     "OLMo",     False),
    "olmo3_32b_dpo":         ("OLMo 3 32B inst.",    "OLMo",     True),
    "qwen3-4b":              ("Qwen 3 4B",           "Qwen",     True),
    "qwen3-32b":             ("Qwen 3 32B",          "Qwen",     True),
    "gpt-oss":               ("GPT-OSS 120B",        "GPT-OSS",  True),
    "deepseek":              ("DeepSeek-V3",         "DeepSeek", True),
    "gemma3-27b":            ("Gemma 3 27B",         "Gemma",    True),
}

plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def load_norm_acc():
    df = pd.read_csv(NORM_ACC_PATH)
    return (df.groupby(["model", "profile_type"])["norm_acc"]
              .mean().reset_index()
              .rename(columns={"norm_acc": "mean_norm_acc"}))


def load_references():
    xgb_by_pt, maj_by_pt = {}, {}
    if XGB_PATH.exists():
        xgb = pd.read_csv(XGB_PATH)
        xgb_by_pt = xgb.groupby("profile_type")["xgb_norm_acc"].mean().to_dict()
    if MAJ_PATH.exists():
        maj = pd.read_csv(MAJ_PATH)
        col = "majority_norm_acc" if "majority_norm_acc" in maj.columns else "majority_acc"
        maj_by_pt = maj.groupby("profile_type")[col].mean().to_dict()
    return xgb_by_pt, maj_by_pt


def load_vr_jsd():
    if VR_JSD_PATH.exists():
        return pd.read_csv(VR_JSD_PATH)
    return pd.DataFrame()


def draw_trajectory_panel(ax, data_dict, metric, title, ylabel,
                          ref_lines=None, y_zero=False):
    """
    data_dict: {model_key: {profile_type: value}}
    ref_lines: list of (value, linestyle, label)
    """
    for model, pt_vals in data_dict.items():
        if model not in MODEL_META:
            continue
        _, family, is_instruct = MODEL_META[model]
        color = FAMILY_COLORS.get(family, "#888888")
        ls = "-" if is_instruct else "--"
        lw = 1.3 if is_instruct else 1.1

        xs = sorted((PROFILE_X[pt], v)
                    for pt, v in pt_vals.items() if pt in PROFILE_X and not np.isnan(v))
        if not xs:
            continue
        xs_plot, ys_plot = zip(*xs)
        ax.plot(xs_plot, ys_plot, color=color, linestyle=ls, linewidth=lw,
                alpha=0.85, marker="o", markersize=3.5, zorder=3)

    if ref_lines:
        for val, ls, label in ref_lines:
            if not np.isnan(val):
                ax.axhline(val, color="black", linestyle=ls, linewidth=1.1,
                           alpha=0.75, label=label, zorder=4)
        ax.legend(fontsize=7.5, loc="upper left",
                  frameon=True, framealpha=0.9,
                  handlelength=1.4, borderpad=0.4)

    if y_zero:
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

    ax.set_xticks(list(PROFILE_X.values()))
    ax.set_xticklabels([PROFILE_LABELS[p] for p in PROFILE_ORDER], fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlim(-0.3, 2.3)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)


def main():
    norm_agg = load_norm_acc()
    xgb_by_pt, maj_by_pt = load_references()
    vr_jsd = load_vr_jsd()

    # Build nested dict: model → profile_type → metric
    norm_dict, vr_dict, jsd_dict = {}, {}, {}
    for _, row in norm_agg.iterrows():
        norm_dict.setdefault(row["model"], {})[row["profile_type"]] = row["mean_norm_acc"]

    if not vr_jsd.empty:
        for _, row in vr_jsd.iterrows():
            vr_dict.setdefault(row["model"], {})[row["profile_type"]] = row["mean_vr"]
            jsd_dict.setdefault(row["model"], {})[row["profile_type"]] = row["mean_jsd"]

    maj_s6 = maj_by_pt.get("s6m4", float("nan"))
    xgb_s6 = xgb_by_pt.get("s6m4", float("nan"))

    # Three-panel figure, same width as Figure 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.subplots_adjust(wspace=0.30)

    # (a) Normalized accuracy
    draw_trajectory_panel(
        axes[0], norm_dict, "mean_norm_acc",
        "(a) Normalized accuracy\n(0 = random chance, 1 = perfect)",
        "Normalized accuracy",
        ref_lines=[
            (maj_s6, "--", f"Majority class ({maj_s6:.3f})"),
            (xgb_s6, "-.", f"XGBoost ({xgb_s6:.3f})"),
        ],
        y_zero=True,
    )

    # (b) Variance ratio
    if vr_dict:
        draw_trajectory_panel(
            axes[1], vr_dict, "mean_vr",
            "(b) Variance ratio\n(VR < 1 = model flattens response diversity)",
            "Variance ratio",
            ref_lines=[(1.0, "--", "Human variance (VR = 1)")],
        )
    else:
        axes[1].text(0.5, 0.5, "VR data not found", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("(b) Variance ratio", fontsize=9)

    # (c) JSD
    if jsd_dict:
        draw_trajectory_panel(
            axes[2], jsd_dict, "mean_jsd",
            "(c) Jensen-Shannon Divergence\n(lower = closer to human distribution)",
            "Mean JSD",
        )
    else:
        axes[2].text(0.5, 0.5, "JSD data not found", ha="center", va="center",
                     transform=axes[2].transAxes)
        axes[2].set_title("(c) JSD", fontsize=9)

    # Bottom legend: family colors + line style for instruct vs base
    family_patches = [mpatches.Patch(facecolor=c, label=f)
                      for f, c in FAMILY_COLORS.items()]
    inst_line = mlines.Line2D([], [], color="gray", linestyle="-", linewidth=1.4,
                              marker="o", markersize=5, label="Instruct")
    base_line = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.2,
                              marker="o", markersize=5, label="Base")
    fig.legend(handles=family_patches + [inst_line, base_line],
               loc="lower center", ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_base = OUT_DIR / "figure2_profile_richness"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_base}.pdf/.png")

    if LATEX_FIG_DIR.exists():
        dest = LATEX_FIG_DIR / "figure_profile_richness.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")


if __name__ == "__main__":
    main()
