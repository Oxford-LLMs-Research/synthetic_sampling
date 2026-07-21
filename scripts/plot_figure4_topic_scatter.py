#!/usr/bin/env python3
"""
Figure 4: Topic scatter — normalized accuracy (x) vs. variance ratio (y, log scale).

Each of the 31 fine-grained topic tags is one point, colored by its thematic
section (7 sections).  `national_ethnic_identity` is excluded: only 2 ESS
questions, both with n<15 respondents and all responses coded as missing-value
categories (Don't know / Refusal), making the accuracy metric meaningless.

Mirrors the ICML draft scatter but updated for EMNLP:
  - x-axis: mean normalized accuracy (all 13 models, all profile types)
  - y-axis: mean variance ratio, log scale (rich profile s6m4, all 13 models)
  - VR = 1.0 reference line marks human-level diversity
  - x = 0 reference marks random-chance performance
  - adjustText used for non-overlapping topic labels
"""
import json
import shutil
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NORM_ACC_PATH = ROOT / "analysis/normalized_accuracy/per_question_norm_acc.csv"
RESULTS_DIR   = ROOT / "analysis"
ZIP_PATH      = ROOT.parent / "synthetic_sampling UPDATED.zip"
OUT_DIR       = ROOT / "analysis/figures/emnlp_revision"
LATEX_FIG_DIR = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
                 "/latex/figures")

SECTION_COLORS = {
    "contemporary_issues":     "#7B9E87",
    "institutional_trust":     "#C73E1D",
    "political_attitudes":     "#2E86AB",
    "political_participation":  "#F18F01",
    "social_attitudes":        "#A23B72",
    "values_identity":         "#6A994E",
    "wellbeing":               "#BC4749",
}
SECTION_LABELS = {
    "contemporary_issues":     "Contemporary Issues",
    "institutional_trust":     "Institutional Trust",
    "political_attitudes":     "Political Attitudes",
    "political_participation": "Political Participation",
    "social_attitudes":        "Social Attitudes",
    "values_identity":         "Values & Identity",
    "wellbeing":               "Wellbeing",
}
TOPIC_DISPLAY = {
    "civic_action":            "Civic Action",
    "civil_liberties":         "Civil Liberties",
    "climate_environment":     "Climate & Env.",
    "corruption_perceptions":  "Corruption",
    "democratic_values":       "Democratic Values",
    "economic_evaluations":    "Econ. Evaluations",
    "economic_policy":         "Econ. Policy",
    "ethical_norms":           "Ethical Norms",
    "gender_attitudes":        "Gender Attitudes",
    "government_performance":  "Gov. Performance",
    "government_trust":        "Gov. Trust",
    "group_trust":             "Group Trust",
    "health":                  "Health",
    "institutional_confidence":"Institutional Conf.",
    "international_relations": "Intl. Relations",
    "interpersonal_trust":     "Interpersonal Trust",
    "life_satisfaction":       "Life Satisfaction",
    "media_information":       "Media & Info.",
    "migration_attitudes":     "Migration",
    "partisanship":            "Partisanship",
    "political_efficacy":      "Political Efficacy",
    "political_interest":      "Political Interest",
    "political_priorities":    "Political Priorities",
    "regime_preferences":      "Regime Prefs.",
    "religious_values":        "Religious Values",
    "security_safety":         "Security & Safety",
    "service_delivery":        "Service Delivery",
    "sexuality_attitudes":     "Sexuality Attitudes",
    "social_capital":          "Social Capital",
    "traditionalism":          "Traditionalism",
    "voting":                  "Voting",
}

META_PATHS = {
    "afrobarometer":   "synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata/pulled_metadata_afrobarometer.json",
    "arabbarometer":   "synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata/pulled_metadata_arabbarometer.json",
    "asianbarometer":  "synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata/pulled_metadata_asianbarometer.json",
    "ess_wave_10":     "synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata/pulled_metadata_ess10.json",
    "ess_wave_11":     "synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata/pulled_metadata_ess11.json",
    "latinobarometer": "synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata/pulled_metadata_latinobarometer.json",
    "wvs":             "synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata/pulled_metadata_wvs.json",
}

MODELS = [
    "deepseek", "gemma3-27b", "gpt-oss",
    "llama3.1_70b_base", "llama3.1_70b_instruct",
    "llama3.1_8b_base",  "llama3.1_8b_instruct",
    "olmo3_32b_base",    "olmo3_32b_dpo",
    "olmo3_7b_base",     "olmo3_7b_dpo",
    "qwen3-32b",         "qwen3-4b",
]


def build_question_metadata() -> dict:
    mapping = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for surv, path in META_PATHS.items():
            try:
                data = json.loads(z.read(path))
            except KeyError:
                print(f"  [warn] missing zip entry for {surv}")
                continue
            for section, questions in data.items():
                if not isinstance(questions, dict):
                    continue
                for qcode, qdata in questions.items():
                    if not isinstance(qdata, dict):
                        continue
                    hs = qdata.get("harmonized_section") or section
                    tt = qdata.get("topic_tag")
                    if (surv, qcode) not in mapping:
                        mapping[(surv, qcode)] = (hs, tt)
    return mapping


def categorical_variance(series: pd.Series) -> float:
    codes = pd.Categorical(series).codes.astype(float)
    codes = codes[codes >= 0]
    return float(np.var(codes, ddof=0)) if len(codes) >= 2 else np.nan


def compute_vr_by_question(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["survey", "target_code", "profile_type",
                                    "ground_truth", "predicted"])
    df = df[df["profile_type"] == "s6m4"]
    rows = []
    for (survey, tc), grp in df.groupby(["survey", "target_code"]):
        vh = categorical_variance(grp["ground_truth"])
        vp = categorical_variance(grp["predicted"])
        if vh and vh > 0:
            rows.append({"survey": survey, "target_code": tc, "vr": vp / vh})
    return pd.DataFrame(rows)


def load_metrics(q_meta: dict) -> pd.DataFrame:
    # Normalized accuracy: all models, rich profile only (s6m4, 24 features)
    acc = pd.read_csv(NORM_ACC_PATH)
    acc = acc[acc["profile_type"] == "s6m4"]
    acc["topic_tag"] = acc.apply(
        lambda r: q_meta.get((r["survey"], r["target_code"]), (None, None))[1], axis=1)
    acc["section"] = acc.apply(
        lambda r: q_meta.get((r["survey"], r["target_code"]), (None, None))[0], axis=1)
    acc_agg = (acc.dropna(subset=["topic_tag"])
                  .groupby(["section", "topic_tag"])["norm_acc"]
                  .mean().reset_index()
                  .rename(columns={"norm_acc": "mean_norm_acc"}))

    # VR: rich profile, all 13 models
    vr_frames = []
    for model in MODELS:
        path = RESULTS_DIR / model / "results_data.csv"
        if not path.exists():
            continue
        vr_df = compute_vr_by_question(path)
        vr_df["model"] = model
        vr_frames.append(vr_df)
        print(f"  VR: {model} ({len(vr_df)} questions)")

    vr_all = pd.concat(vr_frames, ignore_index=True)
    vr_all["topic_tag"] = vr_all.apply(
        lambda r: q_meta.get((r["survey"], r["target_code"]), (None, None))[1], axis=1)
    vr_all["section"] = vr_all.apply(
        lambda r: q_meta.get((r["survey"], r["target_code"]), (None, None))[0], axis=1)
    vr_agg = (vr_all.dropna(subset=["topic_tag"])
                    .groupby(["section", "topic_tag"])["vr"]
                    .mean().reset_index()
                    .rename(columns={"vr": "mean_vr"}))

    merged = pd.merge(acc_agg, vr_agg, on=["section", "topic_tag"], how="inner")
    # Exclude data-artifact topic: only 2 ESS questions, all responses are
    # missing-value codes (Don't know/Refusal), n<15 respondents each.
    merged = merged[merged["topic_tag"] != "national_ethnic_identity"]
    return merged


def main():
    print("Loading question metadata...")
    q_meta = build_question_metadata()
    print(f"  {len(q_meta)} entries")

    print("\nComputing metrics per topic...")
    merged = load_metrics(q_meta)
    print(f"\n{len(merged)} topics with both metrics")

    plt.rcParams.update({
        "font.family": "serif",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    x_min, x_max = merged["mean_norm_acc"].min() - 0.05, merged["mean_norm_acc"].max() + 0.07
    y_min_log = 0.035   # just below political_priorities (0.053)
    y_max_log = 7.0     # just above group_trust (4.03)

    # Light shading for "worse than chance" region
    ax.axvspan(x_min, 0, color="#f5f5f5", zorder=0)

    # Reference lines
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, zorder=1)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1, alpha=0.7, zorder=2)

    # Scatter + collect text objects
    xs, ys, texts = [], [], []
    for _, row in merged.iterrows():
        sec   = row["section"]
        topic = row["topic_tag"]
        x     = row["mean_norm_acc"]
        y     = row["mean_vr"]
        color = SECTION_COLORS.get(sec, "#888888")
        ax.scatter(x, y, color=color, s=58, zorder=5,
                   alpha=0.92, linewidths=0.4, edgecolors="white")
        label = TOPIC_DISPLAY.get(topic, topic.replace("_", " ").title())
        xs.append(x)
        ys.append(y)
        texts.append(
            ax.text(x, y, label, fontsize=6.8, color=color,
                    ha="center", va="bottom", zorder=6)
        )

    # Log scale y-axis — set BEFORE adjust_text so coordinates are correct
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_log, y_max_log)
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0]))

    # Non-overlapping labels
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            x=xs, y=ys,
            expand_points=(1.6, 2.0),
            expand_text=(1.4, 1.6),
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.45),
            force_text=(0.5, 0.7),
            force_points=(0.3, 0.4),
            lim=500,
        )
    except ImportError:
        print("  [note] adjustText not installed")

    ax.set_xlabel(
        "Mean normalized accuracy  (0 = random chance, positive = above chance)",
        fontsize=9)
    ax.set_ylabel(
        "Mean variance ratio, log scale\n"
        r"(VR $<$ 1: model flattens diversity; VR $=$ 1: human-like spread)",
        fontsize=9)

    ax.grid(axis="both", linestyle="--", alpha=0.18, linewidth=0.4)

    # Section legend
    section_patches = [
        mpatches.Patch(facecolor=SECTION_COLORS[s], label=SECTION_LABELS[s], alpha=0.9)
        for s in sorted(SECTION_COLORS)
    ]
    ax.legend(
        handles=section_patches,
        fontsize=7.5, loc="upper left",
        frameon=True, framealpha=0.92,
        handlelength=1.2, borderpad=0.5,
        ncol=1,
    )

    plt.tight_layout(pad=0.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "figure4_topic_scatter"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out_base}.pdf/.png")

    if LATEX_FIG_DIR.exists():
        dest = LATEX_FIG_DIR / "figure_topic_scatter.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")


if __name__ == "__main__":
    main()
