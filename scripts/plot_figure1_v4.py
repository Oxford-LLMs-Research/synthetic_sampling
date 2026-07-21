#!/usr/bin/env python3
"""
Figure 1 v4 — single scatter: normalized accuracy (x) vs. variance ratio (y).

Reference lines:
  - Vertical: majority-class and XGBoost norm_acc (x-axis extended to show gap)
  - Horizontal: VR = 1 (human-level diversity)

Labels placed with adjustText.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "DeepSeek": "#6A994E",
    "GPT-OSS":  "#C73E1D",
    "Gemma":    "#BC4749",
    "Llama":    "#2E86AB",
    "OLMo":     "#A23B72",
    "Qwen":     "#F18F01",
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
    return df, maj_val, xgb_val


def make_figure(df, maj_val, xgb_val):
    plt.rcParams.update({
        "font.family":       "serif",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        300,
    })

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    # Shade the "gap" region between best LLM and XGBoost baseline (now vertical)
    best_llm = df["norm_acc"].max()
    ax.axhspan(best_llm, xgb_val, color="#f0f0f0", zorder=0)

    # Reference lines: baselines are horizontal ceilings, VR=1 is a vertical boundary
    ax.axhline(maj_val, color="black",   linestyle="--", linewidth=1.5, zorder=3,
               label=f"Majority class ({maj_val:.3f})")
    ax.axhline(xgb_val, color="#555555", linestyle="-.", linewidth=1.5, zorder=3,
               label=f"XGBoost ({xgb_val:.3f})")
    ax.axvline(1.0,     color="black",   linestyle=":",  linewidth=1.2,
               alpha=0.6, zorder=2, label="VR = 1  (human diversity)")

    # Scatter — axes swapped: x=VR, y=norm_acc
    xs, ys, texts = [], [], []
    for _, row in df.iterrows():
        ax.scatter(row["vr"], row["norm_acc"],
                   color=row["face"], edgecolors=row["color"],
                   s=90, linewidths=1.9, zorder=5)
        xs.append(row["vr"])
        ys.append(row["norm_acc"])
        texts.append(
            ax.text(row["vr"], row["norm_acc"], row["display"],
                    fontsize=7.2, color=row["color"],
                    ha="center", va="bottom", zorder=6)
        )

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax, x=xs, y=ys,
            expand_points=(1.8, 2.2),
            expand_text=(1.4, 1.6),
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5),
            force_text=(0.5, 0.7),
            force_points=(0.3, 0.4),
            lim=500,
        )
    except ImportError:
        pass

    ax.set_xlabel("Variance ratio  (VR < 1: flattens diversity)", fontsize=10)
    ax.set_ylabel("Normalized accuracy  (0 = random chance)", fontsize=10)
    ax.grid(linestyle="--", alpha=0.2, linewidth=0.4)

    ax.set_xlim(0.38, df["vr"].max() + 0.10)
    ax.set_ylim(-0.01, maj_val + 0.025)

    # Annotate the gap (now vertical)
    gap_mid = (best_llm + xgb_val) / 2
    gap_x   = df["vr"].min() + 0.02
    ax.annotate("", xy=(gap_x, xgb_val - 0.003), xytext=(gap_x, best_llm + 0.003),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.0))
    ax.text(gap_x + 0.012, gap_mid, "LLM–XGBoost gap",
            ha="left", va="center", fontsize=7.5, color="#666666")

    # Legend: reference lines + family colors + instruct/base
    ref_handles = [
        mlines.Line2D([], [], color="black",   linestyle="--", linewidth=1.4,
                      label=f"Majority class ({maj_val:.3f})"),
        mlines.Line2D([], [], color="#555555", linestyle="-.", linewidth=1.4,
                      label=f"XGBoost ({xgb_val:.3f})"),
        mlines.Line2D([], [], color="black",   linestyle=":",  linewidth=1.2,
                      alpha=0.7, label="VR = 1  (human diversity)"),
    ]
    family_patches = [
        mpatches.Patch(facecolor=FAMILY_COLORS[f], label=f, alpha=0.9)
        for f in sorted(FAMILY_COLORS)
    ]
    inst_m = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                           markersize=7, label="Instruct / fine-tuned")
    base_m = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                           markersize=7, markerfacecolor="white",
                           markeredgewidth=1.5, label="Base")

    ax.legend(handles=ref_handles + family_patches + [inst_m, base_m],
              fontsize=7.5, loc="upper left", framealpha=0.92,
              ncol=2, handlelength=1.6, borderpad=0.6)

    plt.tight_layout(pad=0.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in [(".pdf", 300), (".png", 150)]:
        fig.savefig(OUT_DIR / f"figure1_v4{suffix}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'figure1_v4.pdf'}")

    for ext in (".pdf", ".png"):
        shutil.copy2(OUT_DIR / f"figure1_v4{ext}",
                     LATEX_FIGS / f"figure1_v4{ext}")
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
