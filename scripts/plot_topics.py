#!/usr/bin/env python
"""
Plot topic accuracy (top/bottom N topics) from disaggregated analysis.

Creates a two-panel plot showing easiest and hardest topics with confidence intervals.

Usage:
    python plot_topics.py                    # All surveys
    python plot_topics.py --country-in-profile-only  # Filtered data
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
    Aggregate results across models for each dimension (topic).
    Filters out topics with no valid data (n=0 or NaN values).
    Returns: {dimension: {'mean': float, 'std': float, 'se': float, 'min': float, 'max': float, 'n': int}}
    """
    dim_values = defaultdict(list)
    dim_n = defaultdict(int)
    
    for model, dims in data.items():
        for dim, metrics in dims.items():
            if dim == 'Unknown':
                continue
            # Skip topics with no data or invalid metrics
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
        description="Plot topic accuracy (top/bottom N topics).",
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
    parser.add_argument(
        '--top-n',
        type=int,
        default=15,
        help='Number of top/bottom topics to show (default: 15)'
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    country_in_profile_only = args.country_in_profile_only
    top_n = args.top_n
    
    # Load data
    analysis_dir = Path(__file__).parent.parent / 'analysis' / 'disaggregated'
    
    if country_in_profile_only:
        suffix = '_country_in_profile'
        print(f"Loading filtered data (*{suffix}.json)...")
    else:
        suffix = ''
        print("Loading data...")
    
    by_topic = load_json(analysis_dir / f'by_topic_tag{suffix}.json')
    
    # Aggregate across models (filters out invalid topics)
    print("Aggregating across models...")
    topic_stats = aggregate_by_dimension(by_topic, 'accuracy')
    
    if not topic_stats:
        print("Error: No valid topic data found")
        return
    
    sorted_topics = sorted(topic_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"Found {len(sorted_topics)} valid topics")
    
    # Adjust top_n if we don't have enough topics
    if len(sorted_topics) < top_n * 2:
        top_n = max(1, len(sorted_topics) // 2)
        print(f"Adjusted top_n to {top_n} (not enough topics)")
    
    # Get top and bottom topics
    top_topics = sorted_topics[:top_n]
    bottom_topics = sorted_topics[-top_n:]
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(6, top_n * 0.4)))
    
    # Panel A: Top N easiest topics
    topics_top = [t[0].replace('_', ' ').title() for t in top_topics]
    means_top = [t[1]['mean'] for t in top_topics]
    ses_top = [t[1]['se'] for t in top_topics]
    ci_95_top = [1.96 * se for se in ses_top]
    
    y_pos_top = np.arange(len(topics_top))
    
    # Draw horizontal lines from xmin to mean (visual effect)
    xmin_top = 0.0
    for i, mean in enumerate(means_top):
        ax1.hlines(y=i, xmin=xmin_top, xmax=mean, color='gray', alpha=0.2, linewidth=0.5)
    
    # Plot points with error bars (horizontal) - dot plot style
    ax1.errorbar(means_top, y_pos_top, xerr=ci_95_top, fmt='o', color='black', 
                markersize=6, capsize=2, capthick=0.5, linewidth=0.5,
                elinewidth=0.5, zorder=3)
    
    # Overlay colored points
    for i, mean in enumerate(means_top):
        ax1.scatter(mean, i, color='#2E5090', s=120, edgecolor='black', 
                  linewidth=0.5, zorder=4)
    
    ax1.set_yticks(y_pos_top)
    ax1.set_yticklabels(topics_top, fontsize=7)
    ax1.set_xlabel('Accuracy (mean ± 95% CI)', fontsize=9)
    
    # Set x-axis limits with padding, ensuring small values are visible
    if means_top:
        max_with_ci_top = max(m + ci for m, ci in zip(means_top, ci_95_top))
        min_with_ci_top = min(m - ci for m, ci in zip(means_top, ci_95_top))
        data_range_top = max_with_ci_top - min_with_ci_top
        padding_top = max(0.02, data_range_top * 0.05)
        xmin_plot_top = max(0.0, min_with_ci_top - padding_top)  # Allow down to 0 for very low values
        xmax_plot_top = min(1.0, max_with_ci_top + padding_top)
        ax1.set_xlim(xmin_plot_top, xmax_plot_top)
    
    # Format x-axis ticks to show 2 decimal places
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax1.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.invert_yaxis()
    
    # Panel B: Bottom N hardest topics
    topics_bottom = [t[0].replace('_', ' ').title() for t in bottom_topics]
    means_bottom = [t[1]['mean'] for t in bottom_topics]
    ses_bottom = [t[1]['se'] for t in bottom_topics]
    ci_95_bottom = [1.96 * se for se in ses_bottom]
    
    y_pos_bottom = np.arange(len(topics_bottom))
    
    # Draw horizontal lines from xmin to mean (visual effect)
    xmin_bottom = 0.0
    for i, mean in enumerate(means_bottom):
        ax2.hlines(y=i, xmin=xmin_bottom, xmax=mean, color='gray', alpha=0.2, linewidth=0.5)
    
    # Plot points with error bars (horizontal) - dot plot style
    ax2.errorbar(means_bottom, y_pos_bottom, xerr=ci_95_bottom, fmt='o', color='black', 
                markersize=6, capsize=2, capthick=0.5, linewidth=0.5,
                elinewidth=0.5, zorder=3)
    
    # Overlay colored points
    for i, mean in enumerate(means_bottom):
        ax2.scatter(mean, i, color='#8B4513', s=120, edgecolor='black', 
                  linewidth=0.5, zorder=4)
    
    ax2.set_yticks(y_pos_bottom)
    ax2.set_yticklabels(topics_bottom, fontsize=7)
    ax2.set_xlabel('Accuracy (mean ± 95% CI)', fontsize=9)
    
    # Set x-axis limits with padding, ensuring small values are visible
    if means_bottom:
        max_with_ci_bottom = max(m + ci for m, ci in zip(means_bottom, ci_95_bottom))
        min_with_ci_bottom = min(m - ci for m, ci in zip(means_bottom, ci_95_bottom))
        data_range_bottom = max_with_ci_bottom - min_with_ci_bottom
        padding_bottom = max(0.02, data_range_bottom * 0.05)
        xmin_plot_bottom = max(0.0, min_with_ci_bottom - padding_bottom)  # Allow down to 0 for very low values
        xmax_plot_bottom = min(1.0, max_with_ci_bottom + padding_bottom)
        ax2.set_xlim(xmin_plot_bottom, xmax_plot_bottom)
    
    # Format x-axis ticks to show 2 decimal places
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax2.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.5)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figures
    if args.output_dir:
        figures_dir = args.output_dir
    else:
        default_subdir = 'disaggregated_analysis_country_in_profile' if country_in_profile_only else 'disaggregated_analysis'
        figures_dir = Path(__file__).parent.parent / 'analysis' / 'figures' / default_subdir
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_suffix = '_country_in_profile' if country_in_profile_only else ''
    filename_base = f'figure_topics{output_suffix}'
    
    plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
    
    label = ' (country_in_profile)' if country_in_profile_only else ''
    print(f"\n[OK] Saved topic plot (top/bottom {top_n} topics{label})")
    print(f"      PDF: {figures_dir / f'{filename_base}.pdf'}")
    print(f"      PNG: {figures_dir / f'{filename_base}.png'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TOPIC ACCURACY SUMMARY")
    print("=" * 80)
    print(f"\nTop {top_n} easiest topics:")
    for topic, stats in top_topics:
        print(f"  {topic:<30} {stats['mean']:.1%} (n={stats['n']:,})")
    
    print(f"\nTop {top_n} hardest topics:")
    for topic, stats in bottom_topics:
        print(f"  {topic:<30} {stats['mean']:.1%} (n={stats['n']:,})")
    
    plt.close()


if __name__ == '__main__':
    main()
