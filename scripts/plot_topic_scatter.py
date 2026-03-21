#!/usr/bin/env python
"""
Plot topic scatterplot: Accuracy vs. Variance Ratio.

Creates a scatterplot showing the relationship between prediction accuracy
and variance ratio (predicted/human) across topics.

Usage:
    python plot_topic_scatter.py                    # All surveys
    python plot_topic_scatter.py --country-in-profile-only  # Filtered data
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Try to import adjustText for label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Note: adjustText not available. Labels may overlap. Install with: pip install adjustText")

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


# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot topic scatterplot: Accuracy vs. Variance Ratio.",
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
        '--show',
        action='store_true',
        help='Show interactive plot window'
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
    
    by_topic = load_json(analysis_dir / f'by_topic_tag{suffix}.json')
    
    # Aggregate across models (simple mean for visualization)
    records = []
    for model, topics in by_topic.items():
        for topic, metrics in topics.items():
            if topic == 'Unknown':
                continue
            # Use variance_ratio_soft_median to match main figure (which uses median)
            variance = metrics.get('variance_ratio_soft_median', metrics.get('variance_ratio_hard_median', 1.0))
            accuracy = metrics.get('accuracy', 0)
            
            # Skip topics with invalid data
            if accuracy is None or np.isnan(accuracy) or variance is None or np.isnan(variance):
                continue
                
            records.append({
                'Topic': topic,
                'Accuracy': accuracy,
                'Variance': variance
            })
    
    if not records:
        print("Error: No valid data found!")
        return
    
    df = pd.DataFrame(records).groupby('Topic').mean().reset_index()
    
    # Clean topic names for display (replace underscores)
    df['Label'] = df['Topic'].str.replace('_', ' ').str.title()
    
    # Apply label abbreviations for better readability
    df['Label'] = df['Label'].replace({
        'Institutional Confidence': 'Inst. Conf.',
        'Political Participation': 'Pol. Partic.',
        'International Relations': 'Intl. Rel.',
        'Climate Environment': 'Climate'
    })
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if country_in_profile_only:
            output_dir = Path(__file__).parent.parent / 'analysis' / 'figures' / 'disaggregated_analysis_country_in_profile'
        else:
            output_dir = Path(__file__).parent.parent / 'analysis' / 'figures' / 'disaggregated_analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    if country_in_profile_only:
        output_filename = 'figure_topic_scatter_country_in_profile'
    else:
        output_filename = 'figure_topic_scatter'
    
    output_path = output_dir / output_filename
    
    # ICML figure guidelines: lines at least 0.5pt, proper font sizes, no titles in figure
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
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter Plot
    ax.scatter(df['Accuracy'], df['Variance'], s=150, alpha=0.7, color='#4c72b0', 
               edgecolors='black', linewidth=0.5, zorder=3)
    
    # Add Reference Lines
    ax.axhline(1.0, color='#d62728', linestyle='--', linewidth=1.5, 
               label='Human Variance (1.0)', zorder=1)
    ax.axvline(0.25, color='gray', linestyle=':', linewidth=1.5, 
               label='Random Chance', zorder=1)
    
    # Smart Annotation: Label only outliers
    # Criteria: Top/Bottom 3 Accuracy, Top/Bottom 3 Variance
    top_acc = df.nlargest(3, 'Accuracy')['Topic'].tolist()
    bot_acc = df.nsmallest(3, 'Accuracy')['Topic'].tolist()
    top_var = df.nlargest(3, 'Variance')['Topic'].tolist()
    bot_var = df.nsmallest(3, 'Variance')['Topic'].tolist()
    
    # Narrative targets for the story
    narrative_targets = ['partisanship', 'health', 'ethical_norms', 'climate_environment', 
                        'group_trust', 'religious_values']
    
    targets = set(top_acc + bot_acc + top_var + bot_var + narrative_targets)
    
    texts = []
    for i, row in df.iterrows():
        if row['Topic'] in targets:
            texts.append(ax.text(row['Accuracy'], row['Variance'], row['Label'], 
                                fontsize=10, fontweight='bold', zorder=4))
    
    # If adjust_text is available, use it. Otherwise, labels might overlap.
    if HAS_ADJUST_TEXT and texts:
        try:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except Exception as e:
            print(f"Warning: adjust_text failed: {e}")
    
    # Formatting - ICML guidelines: no title in figure, proper axis labels
    ax.set_xlabel('Prediction Accuracy', fontsize=14, labelpad=8)
    ax.set_ylabel('Variance Ratio (Predicted / Human)', fontsize=14, labelpad=8)
    ax.tick_params(axis='both', labelsize=12)
    
    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fontsize=12, 
                       framealpha=0.9, handlelength=1.5)
    legend.get_frame().set_linewidth(0.5)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path.with_suffix('.pdf')}")
    print(f"Saved figure to {output_path.with_suffix('.png')}")
    
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()
