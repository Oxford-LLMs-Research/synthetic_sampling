#!/usr/bin/env python3
"""
Figure 1 v3 — parallel dot plots sharing one y-axis.

Panel (a) [wider]: Normalized accuracy, models sorted best→worst (top→bottom).
    - Reference lines: majority-class (dashed) and XGBoost (dash-dot)
    - y-axis: model names

Panel (b) [narrower, same model order, no y-labels]: Variance ratio.
    - Reference line: VR = 1 (human-level diversity)
    - Rank shifts between panels reveal independence of the two failure modes
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
ANALYSIS   = ROOT / "analysis"
NORM_ACC   = ANALYSIS / "normalized_accuracy" / "per_question_norm_acc.csv"
MAJORITY   = ANALYSIS / "normalized_accuracy" / "majority_class_norm_acc.csv"
XGB        = ANALYSIS / "xgboost_baseline" / "results_merged.csv"
VR_CACHE   = ANALYSIS / "figures" / "emnlp_revision" / "vr_jsd_by_model.csv"
OUT_DIR    = ANALYSIS / "figures" / "emnlp_revision"
LATEX_FIGS = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
              "/latex/figures")

PT = "s6m4"

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

FAMILY_COLORS = {
    "Llama":    "#2E86AB",
    "OLMo":     "#A23B72",
    "Qwen":     "#F18F01",
    "GPT-OSS":  "#C73E1D",
    "DeepSeek": "#6A994E",
    "Gemma":    "#BC4749",
}


def load_data():
    acc = (pd.read_csv(NORM_ACC)
           .query("profile_type == @PT")
           .groupby("model")["norm_acc"].mean()
           .rename("norm_acc"))
    maj_val = (pd.read_csv(MAJORITY)
               .query("profile_type == @PT")["majority_norm_acc"].mean())
    xgb_val = (pd.read_csv(XGB)
               .query("profile_type == @PT")["xgb_norm_acc"].mean())
    vr = (pd.read_csv(VR_CACHE)
          .query("profile_type == @PT")
          .set_index("model")["mean_vr"])

    df = pd.DataFrame({"norm_acc": acc, "vr": vr})
    df = df[df.index.isin(MODEL_META)].copy()
    df["display"]     = df.index.map(lambda m: MODEL_META[m][0])
    df["family"]      = df.index.map(lambda m: MODEL_META[m][1])
    df["is_instruct"] = df.index.map(lambda m: MODEL_META[m][2])
    df["color"]       = df["family"].map(FAMILY_COLORS)
    df["face"]        = df.apply(lambda r: r["color"] if r["is_instruct"] else "white", axis=1)
    df = df.sort_values("norm_acc", ascending=True)   # bottom→top = worst→best
    return df, maj_val, xgb_val


def make_figure(df, maj_val, xgb_val):
    plt.rcParams.update({
        "font.family":       "serif",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        300,
    })

    fig = plt.figure(figsize=(12.0, 5.6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1.6], wspace=0.06)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    y_pos = np.arange(len(df))
    n = len(df)

    # ------------------------------------------------------------------
    # Panel (a): normalized accuracy
    # ------------------------------------------------------------------
    for i, (_, row) in enumerate(df.iterrows()):
        ax_a.scatter(row["norm_acc"], y_pos[i],
                     color=row["face"], edgecolors=row["color"],
                     s=82, linewidths=1.9, zorder=4)

    ax_a.axvline(maj_val, color="black",   linestyle="--", linewidth=1.5, zorder=3,
                 label=f"Majority class ({maj_val:.3f})")
    ax_a.axvline(xgb_val, color="#555555", linestyle="-.", linewidth=1.5, zorder=3,
                 label=f"XGBoost ({xgb_val:.3f})")
    ax_a.axvline(0,       color="black",   linestyle=":",  linewidth=0.9,
                 alpha=0.45, zorder=2)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(df["display"], fontsize=8.8)
    ax_a.set_xlabel("Normalized accuracy  (0 = random chance)", fontsize=9.5)
    ax_a.set_title("(a)  Normalized accuracy — rich profile (24 features)",
                   fontsize=9.2, pad=7, loc="left")
    ax_a.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.5)
    ax_a.legend(fontsize=7.8, loc="lower right", framealpha=0.92)
    ax_a.set_xlim(min(df["norm_acc"].min() - 0.015, -0.02),
                  maj_val + 0.025)
    ax_a.set_ylim(-0.6, n - 0.4)

    # Light horizontal guides connecting the two panels
    for yi in y_pos:
        ax_a.axhline(yi, color="#e8e8e8", linewidth=0.4, zorder=0)

    # ------------------------------------------------------------------
    # Panel (b): variance ratio (same model order, no y-labels)
    # ------------------------------------------------------------------
    for i, (_, row) in enumerate(df.iterrows()):
        ax_b.scatter(row["vr"], y_pos[i],
                     color=row["face"], edgecolors=row["color"],
                     s=82, linewidths=1.9, zorder=4)
        ax_b.axhline(y_pos[i], color="#e8e8e8", linewidth=0.4, zorder=0)

    ax_b.axvline(1.0, color="black", linestyle="--", linewidth=1.3,
                 alpha=0.75, zorder=3, label="VR = 1")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([])
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlabel("Variance ratio", fontsize=9.5)
    ax_b.set_title("(b)  Variance ratio\n(VR < 1: flattens diversity)",
                   fontsize=9.2, pad=7, loc="left")
    ax_b.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.5)
    ax_b.legend(fontsize=7.8, loc="lower right", framealpha=0.92)
    ax_b.set_xlim(df["vr"].min() - 0.05, df["vr"].max() + 0.08)
    ax_b.set_ylim(-0.6, n - 0.4)

    # ------------------------------------------------------------------
    # Shared bottom legend: family colors + instruct/base
    # ------------------------------------------------------------------
    family_patches = [
        mpatches.Patch(facecolor=FAMILY_COLORS[f], label=f, alpha=0.9)
        for f in sorted(FAMILY_COLORS)
    ]
    inst_m = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                           markersize=7, label="Instruct / fine-tuned")
    base_m = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                           markersize=7, markerfacecolor="white",
                           markeredgewidth=1.5, label="Base")
    fig.legend(handles=family_patches + [inst_m, base_m],
               loc="lower center", ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in [(".pdf", 300), (".png", 150)]:
        fig.savefig(OUT_DIR / f"figure1_v3{suffix}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'figure1_v3.pdf'}")

    for ext in (".pdf", ".png"):
        shutil.copy2(OUT_DIR / f"figure1_v3{ext}",
                     LATEX_FIGS / f"figure1_v3{ext}")
    print(f"Copied to {LATEX_FIGS}")


def main():
    print("Loading data...")
    df, maj_val, xgb_val = load_data()
    print(f"  {len(df)} models  maj={maj_val:.4f}  xgb={xgb_val:.4f}")
    print("Generating figure...")
    make_figure(df, maj_val, xgb_val)
    print("Done.")


if __name__ == "__main__":
    main()
