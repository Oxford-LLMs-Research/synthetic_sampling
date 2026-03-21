#!/usr/bin/env python3
"""
Plot conditional stereotyping: How explicit country labels shift model reasoning.

This script compares regional coefficients from mixed-effects models fitted on:
1. Implicit: profiles without explicit country mention
2. Explicit: profiles with country explicitly stated

The dumbbell chart shows how accuracy shifts when country is made explicit.
"""

import json
import os
from pathlib import Path

import matplotlib
# Use non-interactive backend if not in interactive environment
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd


def load_region_coefficients(json_path: Path) -> dict[str, float]:
    """Extract region coefficients from mixed-effects model JSON output."""
    with open(json_path) as f:
        data = json.load(f)
    
    fixed_effects = data["fixed_effects"]
    
    # Extract region coefficients (keys start with "region")
    region_coeffs = {}
    for key, value in fixed_effects.items():
        if key.startswith("region") and key != "regionUnknown":
            # Remove "region" prefix to get clean region name
            region_name = key.replace("region", "")
            # Convert CamelCase/concatenated to readable format
            # e.g., "CentralAfrica" -> "Central Africa"
            import re
            region_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", region_name)
            region_coeffs[region_name] = value
    
    return region_coeffs


def create_comparison_dataframe(implicit_coeffs: dict, explicit_coeffs: dict) -> pd.DataFrame:
    """Create a DataFrame comparing implicit and explicit coefficients."""
    # Get all regions present in both
    all_regions = set(implicit_coeffs.keys()) & set(explicit_coeffs.keys())
    
    data = {
        "Region": list(all_regions),
        "Implicit": [implicit_coeffs[r] for r in all_regions],
        "Explicit": [explicit_coeffs[r] for r in all_regions],
    }
    
    df = pd.DataFrame(data)
    df["Change"] = df["Explicit"] - df["Implicit"]
    df = df.sort_values("Change", ascending=True)
    df = df.reset_index(drop=True)
    
    return df


def plot_conditional_stereotyping(df: pd.DataFrame, output_path: Path, show: bool = False) -> None:
    """Create dumbbell chart showing shift from implicit to explicit."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each region
    for idx, (_, row) in enumerate(df.iterrows()):
        # Color logic: Green if explicit info helps, Red if it hurts/interferes
        color = "#2ca02c" if row["Change"] > 0 else "#d62728"
        
        # The Arrow/Line
        ax.plot(
            [row["Implicit"], row["Explicit"]], [idx, idx],
            color=color, alpha=0.6, linewidth=2, zorder=1
        )
        
        # The "Start" Point (Implicit/No Country) - Neutral Gray
        ax.scatter(row["Implicit"], idx, color="gray", s=80, alpha=0.8, zorder=2)
        
        # The "End" Point (Explicit/Country) - Colored
        ax.scatter(row["Explicit"], idx, color=color, s=80, zorder=3)

    # Formatting
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Region"], fontsize=13, fontfamily="serif")
    ax.set_xlabel(
        "Marginal Probability of Correct Response (Fixed Effect Coefficient)",
        fontsize=18, fontfamily="serif", labelpad=8
    )
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Reference Line (Baseline = Caribbean)
    ax.axvline(0, color="black", linestyle=":", alpha=0.4)

    # Set x-axis limits: start at 0, extend to max value with padding
    x_min = 0
    x_max = max(df["Implicit"].max(), df["Explicit"].max())
    padding = (x_max - x_min) * 0.05  # 5% padding
    ax.set_xlim(x_min, x_max + padding)

    # Custom Legend (larger for readability)
    legend_handles = [
        mlines.Line2D(
            [], [], color="gray", marker="o", linestyle="None",
            markersize=12, label="Implicit"
        ),
        mlines.Line2D(
            [], [], color="#2ca02c", marker="o", linestyle="-",
            markersize=12, label="Explicit (Helps)"
        ),
        mlines.Line2D(
            [], [], color="#d62728", marker="o", linestyle="-",
            markersize=12, label="Explicit (Hurts)"
        ),
    ]
    legend = ax.legend(handles=legend_handles, loc="lower right", frameon=True, 
                       fontsize=18, title_fontsize=18, handlelength=1.5, handletextpad=0.5)

    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path.with_suffix('.pdf')}")
    print(f"Saved figure to {output_path.with_suffix('.png')}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main(show: bool = False):
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    mixed_effects_dir = project_root / "analysis" / "mixed_effects"
    
    implicit_path = mixed_effects_dir / "mixed_effects_model_v2_no_country_in_profile.json"
    explicit_path = mixed_effects_dir / "mixed_effects_model_v2_country_in_profile.json"
    output_path = mixed_effects_dir / "conditional_stereotyping_dumbbell"
    
    # Load coefficients from JSON files
    print(f"Loading implicit model from: {implicit_path}")
    implicit_coeffs = load_region_coefficients(implicit_path)
    
    print(f"Loading explicit model from: {explicit_path}")
    explicit_coeffs = load_region_coefficients(explicit_path)
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(implicit_coeffs, explicit_coeffs)
    
    print("\nRegion coefficient comparison:")
    print(df.to_string(index=False))
    print(f"\nTotal regions: {len(df)}")
    
    # Plot
    plot_conditional_stereotyping(df, output_path, show=show)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot conditional stereotyping dumbbell chart")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    args = parser.parse_args()
    main(show=args.show)
