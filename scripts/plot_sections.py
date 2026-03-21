#!/usr/bin/env python
"""
Plot section accuracy from disaggregated analysis.

Creates a bar plot showing accuracy by thematic section with confidence intervals.

Usage:
    python plot_sections.py                    # All surveys
    python plot_sections.py --country-in-profile-only  # Filtered data
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import sys

# Add scripts directory to path
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_script_dir))


# =============================================================================
# LOAD DATA
# =============================================================================

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_by_dimension(data, metric='accuracy'):
    """
    Aggregate results across models for each dimension (section).
    Filters out sections with no valid data (n=0 or NaN values).
    Returns: {dimension: {'mean': float, 'std': float, 'se': float, 'min': float, 'max': float, 'n': int}}
    """
    dim_values = defaultdict(list)
    dim_n = defaultdict(int)
    
    for model, dims in data.items():
        for dim, metrics in dims.items():
            if dim == 'Unknown':
                continue
            # Skip sections with no data or invalid metrics
            n_obs = metrics.get('n', 0)
            acc_value = metrics.get(metric, 0)
            
            # Only include if we have valid data (n > 0 and valid accuracy)
            if n_obs > 0 and not np.isnan(acc_value) and acc_value is not None:
                dim_values[dim].append(acc_value)
                dim_n[dim] = n_obs  # Same across models
    
    results = {}
    for dim, values in dim_values.items():
        if not values:  # Skip if no valid values
            continue
            
        mean_acc = np.mean(values)
        n_obs = dim_n[dim]
        
        # Standard error of proportion (binomial): SE = sqrt(p*(1-p)/n)
        se = np.sqrt(mean_acc * (1 - mean_acc) / n_obs) if n_obs > 0 else 0
        
        # Also keep std across models for reference
        std_across_models = np.std(values)
        
        results[dim] = {
            'mean': mean_acc,
            'std': std_across_models,
            'se': se,
            'min': np.min(values),
            'max': np.max(values),
            'n': n_obs,
            'n_models': len(values)
        }
    return results


# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot section accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--country-in-profile-only',
        action='store_true',
        help='Use disaggregated results filtered to instances with country/region in profile features '
             '(uses *_country_in_profile.json).'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for figures (default: analysis/figures/disaggregated_analysis)'
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    country_in_profile_only = args.country_in_profile_only
    
    # Load data
    analysis_dir = Path(__file__).parent.parent / 'analysis' / 'disaggregated'
    
    if country_in_profile_only:
        suffix = '_country_in_profile'
        print(f"Loading filtered data (*{suffix}.json)...")
    else:
        suffix = ''
        print("Loading data...")
    
    by_section = load_json(analysis_dir / f'by_section{suffix}.json')
    
    # Aggregate across models (filters out invalid sections)
    print("Aggregating across models...")
    section_stats = aggregate_by_dimension(by_section, 'accuracy')
    
    if not section_stats:
        print("Error: No valid section data found")
        return
    
    sorted_sections = sorted(section_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"Found {len(sorted_sections)} valid sections")
    
    # ICML figure guidelines: lines at least 0.5pt, 9pt font, no titles in figure
    plt.rcParams.update({
        'font.size': 9,
        'axes.linewidth': 0.5,
        'font.family': 'sans-serif',
        'lines.linewidth': 0.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(sorted_sections) * 0.3)))
    
    # Extract data
    sections = [s[0].replace('_', ' ').title() for s in sorted_sections]
    means = [s[1]['mean'] for s in sorted_sections]
    ses = [s[1]['se'] for s in sorted_sections]
    
    # Use 95% confidence intervals: ±1.96*SE
    ci_95 = [1.96 * se for se in ses]
    
    y_pos = np.arange(len(sections))
    
    # Draw horizontal lines from xmin to mean (visual effect)
    xmin = 0.0
    for i, mean in enumerate(means):
        ax.hlines(y=i, xmin=xmin, xmax=mean, color='gray', alpha=0.2, linewidth=0.5)
    
    # Plot points with error bars (horizontal) - dot plot style
    ax.errorbar(means, y_pos, xerr=ci_95, fmt='o', color='black', 
                markersize=6, capsize=2, capthick=0.5, linewidth=0.5,
                elinewidth=0.5, zorder=3)
    
    # Overlay colored points
    for i, mean in enumerate(means):
        ax.scatter(mean, i, color='#2E5090', s=120, edgecolor='black', 
                  linewidth=0.5, zorder=4)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sections, fontsize=8)
    ax.set_xlabel('Accuracy (mean ± 95% CI)', fontsize=9)
    
    # Set x-axis limits with padding, ensuring small values are visible
    if means:
        max_with_ci = max(m + ci for m, ci in zip(means, ci_95))
        min_with_ci = min(m - ci for m, ci in zip(means, ci_95))
        data_range = max_with_ci - min_with_ci
        padding = max(0.02, data_range * 0.05)
        xmin_plot = max(0.0, min_with_ci - padding)  # Allow down to 0 for very low values
        xmax_plot = min(1.0, max_with_ci + padding)
        ax.set_xlim(xmin_plot, xmax_plot)
    
    # Format x-axis ticks to show 2 decimal places
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figures
    if args.output_dir:
        figures_dir = args.output_dir
    else:
        default_subdir = 'disaggregated_analysis_country_in_profile' if country_in_profile_only else 'disaggregated_analysis'
        figures_dir = Path(__file__).parent.parent / 'analysis' / 'figures' / default_subdir
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_suffix = '_country_in_profile' if country_in_profile_only else ''
    filename_base = f'figure_sections{output_suffix}'
    
    plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
    
    label = ' (country_in_profile)' if country_in_profile_only else ''
    print(f"\n[OK] Saved section plot{label}")
    print(f"      PDF: {figures_dir / f'{filename_base}.pdf'}")
    print(f"      PNG: {figures_dir / f'{filename_base}.png'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SECTION ACCURACY SUMMARY")
    print("=" * 80)
    print(f"\n{'Section':<25} {'Mean Acc':>10} {'95% CI':>15} {'N':>10}")
    print("-" * 65)
    for section, stats in sorted_sections:
        ci_str = f"±{1.96*stats['se']:.1%}"
        section_name = section.replace('_', ' ').title()
        print(f"{section_name:<25} {stats['mean']:>9.1%} {ci_str:>15} {stats['n']:>10,}")
    
    best = sorted_sections[0]
    worst = sorted_sections[-1]
    gap = best[1]['mean'] - worst[1]['mean']
    print(f"\nGap between best ({best[0]}) and worst ({worst[0]}): {gap:.1%}")
    
    plt.close()


if __name__ == '__main__':
    main()
