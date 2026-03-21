#!/usr/bin/env python3
"""
Identify baseline/reference categories for fixed effects in mixed effects model.

In R's lme4, when categorical variables are used as fixed effects, the first level
(alphabetically) is used as the baseline/reference category. All other levels are
compared to this baseline, so the baseline itself does not appear in the coefficient
list.

Usage:
    python identify_baseline_categories.py --data analysis/mixed_effects/mixed_effects_data.csv
"""

import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Identify baseline categories for fixed effects'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to CSV data file'
    )
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  Loaded {len(df):,} observations\n")
    
    print("=" * 80)
    print("BASELINE/REFERENCE CATEGORIES FOR FIXED EFFECTS")
    print("=" * 80)
    print("\nIn R's lme4, the first factor level (alphabetically) is used as the baseline.")
    print("All other levels are compared to this baseline.\n")
    
    # Models
    print("MODELS:")
    models = sorted(df['model'].unique())
    print(f"  Total models: {len(models)}")
    print("  All model levels (alphabetical order):")
    for i, model in enumerate(models, 1):
        marker = "  -> " if i == 1 else "     "
        baseline_note = " [BASELINE - not shown in coefficients]" if i == 1 else ""
        print(f"{marker}{i}. {model}{baseline_note}")
    print()
    
    # Regions
    print("REGIONS:")
    regions = sorted(df['region'].unique())
    print(f"  Total regions: {len(regions)}")
    print("  All region levels (alphabetical order):")
    for i, region in enumerate(regions, 1):
        marker = "  -> " if i == 1 else "     "
        baseline_note = " [BASELINE - not shown in coefficients]" if i == 1 else ""
        print(f"{marker}{i}. {region}{baseline_note}")
    print()
    
    # Topic sections
    print("TOPIC SECTIONS:")
    topics = sorted(df['topic_section'].unique())
    print(f"  Total topic sections: {len(topics)}")
    print("  All topic section levels (alphabetical order):")
    for i, topic in enumerate(topics, 1):
        marker = "  -> " if i == 1 else "     "
        baseline_note = " [BASELINE - not shown in coefficients]" if i == 1 else ""
        print(f"{marker}{i}. {topic}{baseline_note}")
    print()
    
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Baseline MODEL: {models[0]}")
    print(f"Baseline REGION: {regions[0]}")
    print(f"Baseline TOPIC_SECTION: {topics[0]}")
    print()
    print("All coefficients in your model output are relative to these baselines.")
    print(f"For example, 'modelgemma-3-27b-instruct' coefficient shows the difference")
    print(f"from the baseline model ({models[0]}).")

if __name__ == '__main__':
    main()
