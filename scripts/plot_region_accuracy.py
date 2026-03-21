#!/usr/bin/env python
"""
Plot region accuracy with confidence intervals.

Creates a dot plot showing average prediction accuracy by region,
with 95% confidence intervals and continent grouping.

Usage:
    python plot_region_accuracy.py                    # All surveys
    python plot_region_accuracy.py --country-in-profile-only  # Filtered data
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
    Aggregate results across models for each dimension (region).
    Returns: {dimension: {'mean': float, 'std': float, 'se': float, 'min': float, 'max': float, 'n': int}}
    
    Standard error (se) is computed using binomial standard error: sqrt(p*(1-p)/n)
    where p is the mean accuracy and n is the number of observations.
    """
    dim_values = defaultdict(list)
    dim_n = defaultdict(int)
    
    for model, dims in data.items():
        for dim, metrics in dims.items():
            if dim == 'Unknown':
                continue
            dim_values[dim].append(metrics[metric])
            dim_n[dim] = metrics['n']  # Same across models
    
    results = {}
    for dim, values in dim_values.items():
        mean_acc = np.mean(values)
        n_obs = dim_n[dim]
        
        # Standard error of proportion (binomial): SE = sqrt(p*(1-p)/n)
        se = np.sqrt(mean_acc * (1 - mean_acc) / n_obs) if n_obs > 0 else 0
        
        # Also keep std across models for reference (model-to-model variation)
        std_across_models = np.std(values)
        
        results[dim] = {
            'mean': mean_acc,
            'std': std_across_models,  # Variation across models
            'se': se,  # Standard error based on n observations
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
        description="Plot region accuracy with confidence intervals.",
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
    
    by_region = load_json(analysis_dir / f'by_region{suffix}.json')
    
    # Aggregate across models
    print("Aggregating across models...")
    region_stats = aggregate_by_dimension(by_region, 'accuracy')
    
    # Map regions to continents
    continent_map = {
        'Central Africa': 'Africa', 'East Africa': 'Africa', 'North Africa': 'Africa', 
        'Southern Africa': 'Africa', 'West Africa': 'Africa',
        'Central America': 'Americas', 'North America': 'Americas', 
        'South America': 'Americas', 'Caribbean': 'Americas',
        'Central Asia': 'Asia', 'East Asia': 'Asia', 'South Asia': 'Asia', 
        'Southeast Asia': 'Asia', 'Middle East': 'Asia',
        'Eastern Europe': 'Europe', 'Northern Europe': 'Europe', 
        'Southern Europe': 'Europe', 'Western Europe': 'Europe',
        'Oceania': 'Oceania'
    }
    
    # Prepare data for plotting
    plot_data = []
    for region, stats in region_stats.items():
        continent = continent_map.get(region, 'Other')
        plot_data.append({
            'region': region,
            'continent': continent,
            'mean': stats['mean'],
            'se': stats['se'],
            'n': stats['n']
        })
    
    # Sort by accuracy (ascending for bottom-to-top display)
    plot_data.sort(key=lambda x: x['mean'])
    
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette matching plot_disaggregated.py style
    colors = {
        'Africa': '#8B4513',      # Brown/Sienna
        'Americas': '#4A7C3F',    # Forest green
        'Asia': '#6B4C93',        # Deep purple
        'Europe': '#2E5090',      # Deep blue
        'Oceania': '#e7298a',     # Magenta
        'Other': '#666666'        # Medium gray
    }
    
    # Extract data for plotting
    regions = [d['region'] for d in plot_data]
    means = [d['mean'] for d in plot_data]
    ses = [d['se'] for d in plot_data]
    continent_colors = [colors.get(d['continent'], '#666666') for d in plot_data]
    
    # Calculate 95% confidence intervals: ±1.96*SE
    ci_95 = [1.96 * se for se in ses]
    
    y_pos = np.arange(len(regions))
    
    # Draw horizontal lines from xmin to mean (optional, for visual effect)
    xmin = 0.15
    for i, (mean, color) in enumerate(zip(means, continent_colors)):
        ax.hlines(y=i, xmin=xmin, xmax=mean, color='gray', alpha=0.2, linewidth=0.5)
    
    # Plot points with error bars (horizontal)
    ax.errorbar(means, y_pos, xerr=ci_95, fmt='o', color='black', 
                markersize=6, capsize=2, capthick=0.5, linewidth=0.5,
                elinewidth=0.5, zorder=3)
    
    # Overlay colored points
    for i, (mean, color) in enumerate(zip(means, continent_colors)):
        ax.scatter(mean, i, color=color, s=120, edgecolor='black', 
                  linewidth=0.5, zorder=4)
    
    # Set labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(regions, fontsize=13)
    ax.set_xlabel('Average Prediction Accuracy (mean ± 95% CI)', fontsize=18, labelpad=8)
    ax.tick_params(axis='x', labelsize=12)
    
    # Set x-axis limits with reasonable padding
    max_accuracy = max(means)
    max_with_ci = max(m + ci for m, ci in zip(means, ci_95))
    min_accuracy = min(means)
    min_with_ci = min(m - ci for m, ci in zip(means, ci_95))
    
    # Add padding: ~5% of range on each side
    data_range = max_with_ci - min_with_ci
    padding = max(0.02, data_range * 0.05)  # At least 0.02 padding
    
    # Don't enforce a minimum of 0.15 - use actual data range
    xmin_plot = max(0.10, min_with_ci - padding)  # Only prevent going below 0.10
    xmax_plot = min(0.45, max_with_ci + padding)
    ax.set_xlim(xmin_plot, xmax_plot)
    
    # Format x-axis ticks to show 2 decimal places (e.g., 0.25 instead of 0.250)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Remove title per ICML guidelines (caption serves this function)
    
    # Create legend for continents (larger for readability)
    from matplotlib.patches import Patch
    
    # Create larger patches for the legend
    legend_elements = [
        Patch(facecolor=colors['Africa'], edgecolor='black', linewidth=0.5, label='Africa'),
        Patch(facecolor=colors['Americas'], edgecolor='black', linewidth=0.5, label='Americas'),
        Patch(facecolor=colors['Asia'], edgecolor='black', linewidth=0.5, label='Asia'),
        Patch(facecolor=colors['Europe'], edgecolor='black', linewidth=0.5, label='Europe'),
        Patch(facecolor=colors['Oceania'], edgecolor='black', linewidth=0.5, label='Oceania'),
    ]
    legend = ax.legend(handles=legend_elements, title='Continent', loc='lower right', 
                       fontsize=18, frameon=True, framealpha=0.9, title_fontsize=18,
                       handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
    # Ensure legend title is also readable
    legend.get_title().set_fontsize(18)
    
    plt.tight_layout()
    
    # Save figures
    if args.output_dir:
        figures_dir = args.output_dir
    else:
        default_subdir = 'disaggregated_analysis_country_in_profile' if country_in_profile_only else 'disaggregated_analysis'
        figures_dir = Path(__file__).parent.parent / 'analysis' / 'figures' / default_subdir
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_suffix = '_country_in_profile' if country_in_profile_only else ''
    filename_base = f'figure_region_accuracy{output_suffix}'
    
    plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
    
    label = ' (country_in_profile)' if country_in_profile_only else ''
    print(f"\n[OK] Saved region accuracy plot{label}")
    print(f"      PDF: {figures_dir / f'{filename_base}.pdf'}")
    print(f"      PNG: {figures_dir / f'{filename_base}.png'}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("REGION ACCURACY SUMMARY")
    print("=" * 80)
    print(f"\n{'Region':<25} {'Mean Acc':>10} {'95% CI':>15} {'N':>10}")
    print("-" * 65)
    for d in plot_data:
        ci_str = f"±{1.96*d['se']:.1%}"
        print(f"{d['region']:<25} {d['mean']:>9.1%} {ci_str:>15} {d['n']:>10,}")
    
    best = plot_data[-1]
    worst = plot_data[0]
    gap = best['mean'] - worst['mean']
    print(f"\nGap between best ({best['region']}) and worst ({worst['region']}): {gap:.1%}")
    
    plt.close()


if __name__ == '__main__':
    main()
