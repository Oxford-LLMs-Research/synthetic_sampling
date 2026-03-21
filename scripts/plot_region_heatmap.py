#!/usr/bin/env python
"""
Plot model × region accuracy heatmap from disaggregated analysis.

Creates a heatmap showing prediction accuracy for each model across regions.

Usage:
    python plot_region_heatmap.py                    # All surveys
    python plot_region_heatmap.py --country-in-profile-only  # Filtered data
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
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


def get_geographic_region_order():
    """
    Define a consistent geographic ordering for regions.
    Groups by continent/sub-region in a logical geographic order.
    
    Returns a list of region names in the desired order.
    Regions not in this list will be appended at the end, sorted alphabetically.
    """
    preferred_order = [
        # Americas (North to South)
        'North America',
        'Central America',
        'Caribbean',
        'South America',
        # Europe (North to South, West to East)
        'Northern Europe',
        'Western Europe',
        'Southern Europe',
        'Eastern Europe',
        # Africa (North to South, West to East)
        'North Africa',
        'West Africa',
        'Central Africa',
        'East Africa',
        'Southern Africa',
        # Middle East & Central Asia
        'Middle East',
        'Central Asia',
        # Asia (West to East, North to South)
        'South Asia',
        'Southeast Asia',
        'East Asia',
        # Oceania
        'Oceania',
    ]
    return preferred_order


def sort_regions_geographically(regions):
    """
    Sort regions using a geographic ordering that groups by continent/sub-region.
    Regions in the preferred order come first, then others alphabetically.
    """
    preferred_order = get_geographic_region_order()
    region_list = list(regions)
    
    # Separate into preferred and other regions
    preferred_regions = []
    other_regions = []
    
    for region in region_list:
        if region in preferred_order:
            preferred_regions.append(region)
        else:
            other_regions.append(region)
    
    # Sort preferred regions by their position in preferred_order
    preferred_regions.sort(key=lambda r: preferred_order.index(r))
    
    # Sort other regions alphabetically
    other_regions.sort()
    
    # Combine: preferred first, then others
    return preferred_regions + other_regions


# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot model × region accuracy heatmap.",
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
    
    # Prepare data matrix
    models = list(by_region.keys())
    if not models:
        print("Error: No data available for heatmap")
        return
    
    # Get all unique regions across all models
    all_regions = set()
    for model_data in by_region.values():
        all_regions.update(model_data.keys())
    all_regions.discard('Unknown')
    
    # Use geographic ordering instead of alphabetical
    region_names = sort_regions_geographically(all_regions)
    
    if not region_names:
        print("Error: No regions found in data")
        return
    
    # Model metadata for consistent sorting and display
    model_metadata = {
        'deepseek-v3p1-terminus': ('DeepSeek', 37, 1, 'DeepSeek'),
        'gemma-3-27b-instruct': ('Gemma', 27, 1, 'Gemma 27B'),
        'gpt_oss': ('GPT-OSS', 120, 1, 'GPT-OSS'),
        'llama3.1-70b-base': ('Llama', 70, 0, 'Llama 70B (b)'),
        'llama3.1-70b-instruct': ('Llama', 70, 1, 'Llama 70B (i)'),
        'llama3.1-8b-base': ('Llama', 8, 0, 'Llama 8B (b)'),
        'llama3.1-8b-instruct': ('Llama', 8, 1, 'Llama 8B (i)'),
        'olmo3-32b-base': ('OLMo', 32, 0, 'OLMo 32B (b)'),
        'olmo3-32b-dpo': ('OLMo', 32, 1, 'OLMo 32B (i)'),
        'olmo3-7b-base': ('OLMo', 7, 0, 'OLMo 7B (b)'),
        'olmo3-7b-dpo': ('OLMo', 7, 1, 'OLMo 7B (i)'),
        'qwen3-32b': ('Qwen', 32, 1, 'Qwen 32B'),
        'qwen3-4b': ('Qwen', 4, 1, 'Qwen 4B'),
    }
    
    # Short model names for display
    model_short = {m: metadata[3] for m, metadata in model_metadata.items()}
    
    # Sort models by: family (alphabetically), then size (largest first), then type (base before instruct)
    def get_model_sort_key(model_name):
        if model_name in model_metadata:
            family, size, model_type, _ = model_metadata[model_name]
            return (family, -size, model_type)  # Negative size for descending order
        else:
            # Fallback: sort by name if not in metadata
            return ('ZZZ', 0, 1)  # Put unknown models at the end
    
    sorted_models = sorted(models, key=get_model_sort_key)
    
    # Build matrix
    matrix = np.zeros((len(sorted_models), len(region_names)))
    for i, model in enumerate(sorted_models):
        for j, region in enumerate(region_names):
            if region in by_region[model]:
                matrix[i, j] = by_region[model][region]['accuracy']
    
    # Determine color scale from data
    vmin = np.nanmin(matrix[matrix > 0]) if np.any(matrix > 0) else 0.20
    vmax = np.nanmax(matrix) if np.any(matrix > 0) else 0.42
    
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
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a colormap where green represents high accuracy (good) and red represents low accuracy (bad)
    # RdYlGn goes: red (low/bad) -> yellow -> green (high/good)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(region_names)))
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(sorted_models)))
    ax.set_yticklabels([model_short.get(m, m) for m in sorted_models], fontsize=8)
    
    # Add colorbar with proper formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy', fontsize=9)
    cbar.ax.tick_params(width=0.5, labelsize=8)
    
    plt.tight_layout()
    
    # Save figures
    if args.output_dir:
        figures_dir = args.output_dir
    else:
        default_subdir = 'disaggregated_analysis_country_in_profile' if country_in_profile_only else 'disaggregated_analysis'
        figures_dir = Path(__file__).parent.parent / 'analysis' / 'figures' / default_subdir
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_suffix = '_country_in_profile' if country_in_profile_only else ''
    filename_base = f'figure_heatmap_region{output_suffix}'
    
    plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
    
    label = ' (country_in_profile)' if country_in_profile_only else ''
    print(f"\n[OK] Saved region heatmap{label}")
    print(f"      PDF: {figures_dir / f'{filename_base}.pdf'}")
    print(f"      PNG: {figures_dir / f'{filename_base}.png'}")
    
    plt.close()


if __name__ == '__main__':
    main()
