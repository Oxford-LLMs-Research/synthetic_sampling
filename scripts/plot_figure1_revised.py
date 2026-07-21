"""
Revised Figure 1 for EMNLP submission.

Three-panel figure (rich profiles only — information scaling goes in figure_profile_richness):
  Panel (a): Normalized accuracy per model (rich profile, 24 features)
             Two reference lines: majority-class (dashed) and XGBoost (dash-dot)
             Colors by model family; filled = instruct, open circle = base
  Panel (b): Variance ratio (predicted entropy / human entropy)
             VR < 1 = model predicts less diversity than humans show
  Panel (c): Jensen-Shannon Divergence (lower = closer to human distribution)

Usage:
    python plot_figure1_revised.py                # rich profiles only (default)
    python plot_figure1_revised.py --all-profiles # include all 3 richness levels
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
NORM_ACC_DIR = ANALYSIS_DIR / "normalized_accuracy"
XGB_DIR = ANALYSIS_DIR / "xgboost_baseline"
OUT_DIR = ANALYSIS_DIR / "figures/emnlp_revision"
# Destination in paper latex folder
LATEX_FIGS = ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference/latex/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_TYPES = ["s3m2", "s4m3", "s6m4"]
PROFILE_LABELS = {"s3m2": "Sparse (6)", "s4m3": "Medium (12)", "s6m4": "Rich (24)"}

# Model folder name -> (display name, family, is_instruct)
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


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_norm_acc():
    per_q = pd.read_csv(NORM_ACC_DIR / "per_question_norm_acc.csv")
    agg = per_q.groupby(["model", "profile_type"])["norm_acc"].mean().reset_index()
    agg.columns = ["model", "profile_type", "mean_norm_acc"]
    return agg


def load_majority_norm_acc():
    maj = pd.read_csv(NORM_ACC_DIR / "majority_class_norm_acc.csv")
    return maj.groupby("profile_type")["majority_norm_acc"].mean().to_dict()


def load_xgb_norm_acc():
    for fname in ["results_merged.csv", "results.csv"]:
        p = XGB_DIR / fname
        if p.exists():
            print(f"  Loading XGBoost from {fname}")
            xgb = pd.read_csv(p)
            return xgb.groupby("profile_type")["xgb_norm_acc"].mean().to_dict()
    print("  WARNING: XGBoost results not found")
    return None


def compute_variance_ratio(predictions, ground_truth):
    from collections import Counter
    def cat_entropy(vals):
        c = Counter(vals)
        total = sum(c.values())
        probs = np.array([v / total for v in c.values()])
        return float(entropy(probs))
    pred_h = cat_entropy(predictions)
    gt_h = cat_entropy(ground_truth)
    return pred_h / gt_h if gt_h > 0 else np.nan


def compute_jsd(predictions, ground_truth, options):
    from collections import Counter
    pred_counts = Counter(predictions)
    gt_counts = Counter(ground_truth)
    all_opts = sorted(set(options))
    p = np.array([pred_counts.get(o, 0) for o in all_opts], dtype=float)
    q = np.array([gt_counts.get(o, 0) for o in all_opts], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p /= p.sum(); q /= q.sum()
    return float(jensenshannon(p, q, base=2))


def load_vr_jsd(profile_types):
    results = []
    model_dirs = sorted(d for d in (ANALYSIS_DIR).iterdir()
                        if (ANALYSIS_DIR / d.name / "results_data.csv").exists())
    for md in model_dirs:
        model = md.name
        print(f"  VR/JSD: {model}")
        df = pd.read_csv(md / "results_data.csv")
        for pt in profile_types:
            sub = df[df["profile_type"] == pt]
            if sub.empty:
                continue
            vr_vals, jsd_vals = [], []
            for (survey, tcode), grp in sub.groupby(["survey", "target_code"]):
                opts = sorted(grp["ground_truth"].unique())
                if len(opts) < 2:
                    continue
                vr = compute_variance_ratio(grp["predicted"].tolist(), grp["ground_truth"].tolist())
                js = compute_jsd(grp["predicted"].tolist(), grp["ground_truth"].tolist(), opts)
                if not np.isnan(vr): vr_vals.append(vr)
                if not np.isnan(js):  jsd_vals.append(js)
            results.append({"model": model, "profile_type": pt,
                            "mean_vr": np.mean(vr_vals) if vr_vals else np.nan,
                            "mean_jsd": np.mean(jsd_vals) if jsd_vals else np.nan})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def make_figure(norm_agg, majority_by_pt, xgb_by_pt, vr_jsd_df,
                show_all_profiles=False):
    """
    Build three-panel Figure 1.
    Default (show_all_profiles=False): one dot per model, rich profile only.
    All-profiles mode: three connected lines per model across sparse/medium/rich.
    """
    PRIMARY_PT = "s6m4"  # rich profile

    # Model display order: sorted by rich-profile normalized accuracy, descending
    rich_df = norm_agg[norm_agg["profile_type"] == PRIMARY_PT].copy()
    rich_df = rich_df[rich_df["model"].isin(MODEL_META)]
    rich_df = rich_df.sort_values("mean_norm_acc", ascending=True)  # ascending for bottom-to-top
    ordered_models = rich_df["model"].tolist()
    n_models = len(ordered_models)

    display_names = [MODEL_META[m][0] for m in ordered_models]
    families = [MODEL_META[m][1] for m in ordered_models]
    is_instruct = [MODEL_META[m][2] for m in ordered_models]
    colors = [FAMILY_COLORS.get(f, "#888888") for f in families]
    markers = ["o" if inst else "o" for inst in is_instruct]  # filled vs open below
    marker_face = [c if inst else "white" for c, inst in zip(colors, is_instruct)]

    y_pos = np.arange(n_models)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(wspace=0.35)

    maj_val = majority_by_pt.get(PRIMARY_PT, np.nan)
    xgb_val = (xgb_by_pt.get(PRIMARY_PT) if xgb_by_pt else None) or np.nan

    # ------------------------------------------------------------------
    # Helper: draw one horizontal panel
    # ------------------------------------------------------------------
    def draw_panel(ax, metric_key, df, title, xlabel, ref_lines=None,
                   show_ylabels=True):
        """
        df: DataFrame with columns [model, profile_type, <metric_key>]
        ref_lines: list of (value, linestyle, color, label)
        show_ylabels: if False, suppress y-axis tick labels (used for panels b/c)
        """
        if show_all_profiles:
            # Three connected points per model, left→right = sparse→rich
            pt_x_map = {"s3m2": 0, "s4m3": 1, "s6m4": 2}
            for i, model in enumerate(ordered_models):
                xs, ys = [], []
                for pt in PROFILE_TYPES:
                    sub = df[(df["model"] == model) & (df["profile_type"] == pt)]
                    if sub.empty: continue
                    val = sub[metric_key].values[0]
                    if not np.isnan(val):
                        xs.append(pt_x_map[pt])
                        ys.append(val)
                fc = marker_face[i]
                ec = colors[i]
                ax.plot(xs, ys, color=ec, alpha=0.7, linewidth=1.2, zorder=2)
                for x, y in zip(xs, ys):
                    ax.scatter(x, y, color=fc, edgecolors=ec, s=60, zorder=3,
                               linewidths=1.5)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Sparse\n(6)", "Medium\n(12)", "Rich\n(24)"], fontsize=9)
            ax.set_xlabel("Profile richness", fontsize=10)
            if ref_lines:
                for val, ls, rc, lbl in ref_lines:
                    if not np.isnan(val):
                        ax.axhline(val, color=rc, linestyle=ls, linewidth=1.8, label=lbl, zorder=5)
        else:
            # One dot per model, sorted horizontally
            pt_df = df[df["profile_type"] == PRIMARY_PT].set_index("model")
            values = [pt_df.loc[m, metric_key] if m in pt_df.index else np.nan
                      for m in ordered_models]
            for i, (val, fc, ec) in enumerate(zip(values, marker_face, colors)):
                if not np.isnan(val):
                    ax.scatter(val, y_pos[i], color=fc, edgecolors=ec, s=80, zorder=4,
                               linewidths=1.8)
            ax.set_yticks(y_pos)
            if show_ylabels:
                ax.set_yticklabels(display_names, fontsize=8.5)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            ax.set_xlabel(xlabel, fontsize=10)
            if ref_lines:
                for val, ls, rc, lbl in ref_lines:
                    if not np.isnan(val):
                        ax.axvline(val, color=rc, linestyle=ls, linewidth=1.8, label=lbl, zorder=5)
            ax.set_xlim(left=min(-0.02, min(v for v in values if not np.isnan(v)) - 0.02))

        ax.set_title(title, fontsize=9, pad=6)
        ax.grid(axis="x" if not show_all_profiles else "y", alpha=0.3)

    # ------------------------------------------------------------------
    # Panel (a): Normalized accuracy
    # ------------------------------------------------------------------
    ref_a = [
        (maj_val, "--",  "black", f"Majority class (norm. acc. = {maj_val:.3f})"),
        (xgb_val, "-.",  "#555555", f"XGBoost (norm. acc. = {xgb_val:.3f})"),
    ]
    draw_panel(axes[0], "mean_norm_acc", norm_agg,
               "(a) Normalized accuracy\n(0 = random chance, 1 = perfect)",
               "Normalized accuracy", ref_a)
    if not show_all_profiles:
        axes[0].axvline(0, color="black", linewidth=0.8, zorder=2)
    axes[0].legend(fontsize=7.5, loc="lower right" if not show_all_profiles else "upper left")

    # ------------------------------------------------------------------
    # Panel (b): Variance ratio
    # ------------------------------------------------------------------
    if vr_jsd_df is not None and not vr_jsd_df.empty:
        ref_b = [(1.0, "--", "black", "Human variance (VR = 1)")]
        draw_panel(axes[1], "mean_vr", vr_jsd_df,
                   "(b) Variance ratio\n(VR < 1 = model flattens response diversity)",
                   "Variance ratio", ref_b, show_ylabels=False)
        axes[1].legend(fontsize=7.5)
    else:
        axes[1].text(0.5, 0.5, "VR/JSD not computed\n(run without --skip-vr-jsd)",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("(b) Variance ratio", fontsize=9)

    # ------------------------------------------------------------------
    # Panel (c): JSD
    # ------------------------------------------------------------------
    if vr_jsd_df is not None and not vr_jsd_df.empty:
        draw_panel(axes[2], "mean_jsd", vr_jsd_df,
                   "(c) Jensen-Shannon Divergence\n(lower = closer to human distribution)",
                   "Mean JSD", ref_lines=None, show_ylabels=False)
    else:
        axes[2].text(0.5, 0.5, "VR/JSD not computed\n(run without --skip-vr-jsd)",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("(c) JSD", fontsize=9)

    # ------------------------------------------------------------------
    # Bottom legend: model family colors + instruct/base marker style
    # ------------------------------------------------------------------
    family_patches = [mpatches.Patch(facecolor=c, label=f)
                      for f, c in FAMILY_COLORS.items()]
    # Filled = instruct, open = base (already visible in y-axis labels; keep as
    # a compact visual aid without redundant legend text entries)
    inst_marker = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                                markersize=7, label="Instruct")
    base_marker = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                                markersize=7, markerfacecolor="white", markeredgewidth=1.5,
                                label="Base")
    fig.legend(handles=family_patches + [inst_marker, base_marker],
               loc="lower center", ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save
    suffix = "_all_profiles" if show_all_profiles else "_s6m4"
    pdf_path = OUT_DIR / f"figure1{suffix}.pdf"
    png_path = OUT_DIR / f"figure1{suffix}.png"
    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved {pdf_path}")

    # Copy rich-profile version to latex figures folder (the one paper.tex references)
    if not show_all_profiles and LATEX_FIGS.exists():
        dest = LATEX_FIGS / "figure1.pdf"
        shutil.copy2(pdf_path, dest)
        print(f"Copied to {dest}")

    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-profiles", action="store_true",
                        help="Show all 3 richness levels (default: rich only)")
    parser.add_argument("--skip-vr-jsd", action="store_true",
                        help="Skip VR/JSD computation (panels b/c will be empty)")
    args = parser.parse_args()

    profile_types_needed = PROFILE_TYPES if args.all_profiles else ["s6m4"]

    print("Loading normalized accuracy data...")
    norm_agg = load_norm_acc()
    majority_by_pt = load_majority_norm_acc()
    xgb_by_pt = load_xgb_norm_acc()
    print(f"  Models found: {norm_agg['model'].nunique()}")
    maj = majority_by_pt.get("s6m4", float("nan"))
    xgb = (xgb_by_pt or {}).get("s6m4", float("nan"))
    print(f"  Majority-class (s6m4): {maj:.4f}")
    print(f"  XGBoost (s6m4):        {xgb}")

    vr_jsd_df = None
    if not args.skip_vr_jsd:
        # Check if cached
        cached = OUT_DIR / "vr_jsd_by_model.csv"
        if cached.exists():
            print(f"Loading cached VR/JSD from {cached}")
            vr_jsd_df = pd.read_csv(cached)
        else:
            print("Computing VR/JSD (a few minutes)...")
            vr_jsd_df = load_vr_jsd(profile_types_needed)
            vr_jsd_df.to_csv(cached, index=False)
            print(f"  Cached to {cached}")

    print("\nGenerating figure...")
    make_figure(norm_agg, majority_by_pt, xgb_by_pt, vr_jsd_df,
                show_all_profiles=args.all_profiles)
    print("Done.")


if __name__ == "__main__":
    main()
