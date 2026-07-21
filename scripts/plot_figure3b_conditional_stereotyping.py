#!/usr/bin/env python3
"""
Figure 3(b): Conditional stereotyping — does making country explicit help or hurt?

Dumbbell chart comparing mean normalized accuracy by region for:
  - Implicit: profiles WITHOUT explicit country/region mention
  - Explicit: profiles WITH country explicitly in features

Sources:
  synthetic_sampling/analysis/mixed_effects/mixed_effects_data_no_country_in_profile.csv
  synthetic_sampling/analysis/mixed_effects/mixed_effects_data_country_in_profile.csv
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MEM_DIR = ROOT / "synthetic_sampling/analysis/mixed_effects"
IMPLICIT_CSV = MEM_DIR / "mixed_effects_data_no_country_in_profile.csv"
EXPLICIT_CSV = MEM_DIR / "mixed_effects_data_country_in_profile.csv"
OUT_DIR = ROOT / "analysis/figures/emnlp_revision"
LATEX_FIG_DIR = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
                 "/latex/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.5,
    "patch.linewidth": 0.5,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def normalized_accuracy(correct, n_options):
    chance = 1.0 / n_options
    return (correct - chance) / (1.0 - chance)


def load_and_aggregate(csv_path: Path, label: str) -> pd.DataFrame:
    print(f"Loading {label} from {csv_path.name}...")
    cols = ["correct", "region", "n_options"]
    # country_in_profile CSV has extra column
    df = pd.read_csv(csv_path, encoding="latin-1",
                     usecols=lambda c: c in ["correct", "region", "n_options", "example_id"])
    df = df[df["region"] != "Unknown"].copy()
    df["n_options"] = pd.to_numeric(df["n_options"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct", "n_options"])
    df = df[df["n_options"] > 1]
    df["norm_acc"] = normalized_accuracy(df["correct"], df["n_options"])
    df = df.dropna(subset=["norm_acc"])
    agg = (df.groupby("region")["norm_acc"]
             .mean()
             .reset_index()
             .rename(columns={"norm_acc": label}))
    return agg


def main():
    impl = load_and_aggregate(IMPLICIT_CSV, "implicit")
    expl = load_and_aggregate(EXPLICIT_CSV, "explicit")

    df = impl.merge(expl, on="region")
    df["change"] = df["explicit"] - df["implicit"]
    df = df.sort_values("change").reset_index(drop=True)

    print("\nRegion: implicit -> explicit (delta norm_acc)")
    for _, row in df.iterrows():
        sign = "+" if row["change"] > 0 else ""
        print(f"  {row['region']:<22} {row['implicit']:+.4f} -> {row['explicit']:+.4f}  "
              f"({sign}{row['change']:.4f})")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for idx, row in df.iterrows():
        color = "#2ca02c" if row["change"] >= 0 else "#d62728"
        # Connecting line
        ax.plot([row["implicit"], row["explicit"]], [idx, idx],
                color=color, alpha=0.6, linewidth=1.5, zorder=1)
        # Implicit dot (gray)
        ax.scatter(row["implicit"], idx, color="gray", s=70, alpha=0.85, zorder=2)
        # Explicit dot (colored)
        ax.scatter(row["explicit"], idx, color=color, s=70, zorder=3)

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["region"], fontsize=8.5)
    ax.set_xlabel("Mean Normalized Accuracy", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.4)

    # Pad x-axis
    all_vals = pd.concat([df["implicit"], df["explicit"]])
    pad = (all_vals.max() - all_vals.min()) * 0.06
    ax.set_xlim(all_vals.min() - pad, all_vals.max() + pad)

    # Legend
    legend_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=8, label="Implicit (no country)"),
        mlines.Line2D([], [], color="#2ca02c", marker="o", linestyle="-",
                      markersize=8, label="Explicit — helps"),
        mlines.Line2D([], [], color="#d62728", marker="o", linestyle="-",
                      markersize=8, label="Explicit — hurts"),
    ]
    legend = ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
                       frameon=True, framealpha=0.9, handlelength=1.5, handletextpad=0.4)
    legend.get_frame().set_linewidth(0.4)

    plt.tight_layout(pad=0.5)

    out_base = OUT_DIR / "figure3b_conditional_stereotyping"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out_base}.pdf/.png")

    if LATEX_FIG_DIR.exists():
        dest = LATEX_FIG_DIR / "conditional_stereotyping_dumbbell.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")


if __name__ == "__main__":
    main()
