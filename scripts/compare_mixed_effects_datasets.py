#!/usr/bin/env python3
"""
Compare main mixed effects dataset vs country-in-profile subset.

Shows differences in:
- Instances per survey
- Questions per topic_section
- n_options distribution (proxy for response format types)
- modal_share distribution
"""

import argparse
import pandas as pd
from pathlib import Path


def header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare main vs country-in-profile mixed effects datasets"
    )
    parser.add_argument(
        '--main',
        type=Path,
        default=Path('analysis/mixed_effects/mixed_effects_data.csv'),
        help='Path to main dataset CSV'
    )
    parser.add_argument(
        '--subset',
        type=Path,
        default=Path('analysis/mixed_effects/mixed_effects_data_country_in_profile.csv'),
        help='Path to country-in-profile subset CSV'
    )
    parser.add_argument(
        '--no-country-subset',
        type=Path,
        default=None,
        help='Path to no-country-in-profile subset CSV (optional, for three-way comparison)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading main dataset:  {args.main}")
    print(f"Loading subset dataset: {args.subset}\n")
    
    df_main = pd.read_csv(args.main, low_memory=False)
    df_sub = pd.read_csv(args.subset, low_memory=False)
    
    df_no_country = None
    if args.no_country_subset and args.no_country_subset.exists():
        print(f"Loading no-country-in-profile: {args.no_country_subset}\n")
        df_no_country = pd.read_csv(args.no_country_subset, low_memory=False)
    
    # 1) Instances per survey
    header("INSTANCES PER SURVEY")
    print("\nMain (all rich profiles):")
    vc_main = df_main['survey'].value_counts().sort_index()
    print(vc_main.to_string())
    print(f"Total: {len(df_main):,} instances")
    
    print("\nSubset (country/region in profile):")
    vc_sub = df_sub['survey'].value_counts().sort_index()
    print(vc_sub.to_string())
    print(f"Total: {len(df_sub):,} instances")
    
    print("\nDifference (subset / main):")
    ratio = (vc_sub / vc_main * 100).sort_index()
    for survey in ratio.index:
        main_count = vc_main[survey]
        sub_count = vc_sub.get(survey, 0)
        pct = ratio[survey]
        print(f"  {survey:20s}: {sub_count:8,} / {main_count:8,} = {pct:5.1f}%")
    
    # 2) Questions per section (topic_section)
    header("UNIQUE QUESTIONS PER TOPIC_SECTION")
    print("\nMain:")
    grp_main = df_main.groupby('topic_section')['question'].nunique().sort_index()
    print(grp_main.to_string())
    print(f"Total unique questions: {df_main['question'].nunique():,}")
    
    print("\nSubset:")
    grp_sub = df_sub.groupby('topic_section')['question'].nunique().sort_index()
    print(grp_sub.to_string())
    print(f"Total unique questions: {df_sub['question'].nunique():,}")
    
    print("\nDifference (subset / main):")
    for section in sorted(set(grp_main.index) | set(grp_sub.index)):
        main_q = grp_main.get(section, 0)
        sub_q = grp_sub.get(section, 0)
        pct = (sub_q / main_q * 100) if main_q > 0 else 0
        print(f"  {section:30s}: {sub_q:4d} / {main_q:4d} = {pct:5.1f}%")
    
    # 3) Distribution of n_options (proxy for response format types)
    header("N_OPTIONS DISTRIBUTION (proxy for response format types)")
    print("\nMain - Overall distribution:")
    nopt_main = df_main['n_options'].value_counts().sort_index()
    print(nopt_main.to_string())
    
    print("\nSubset - Overall distribution:")
    nopt_sub = df_sub['n_options'].value_counts().sort_index()
    print(nopt_sub.to_string())
    
    print("\nMain - By topic_section (count / mean / min / max):")
    stats_main = df_main.groupby('topic_section')['n_options'].agg(['count', 'mean', 'min', 'max']).sort_index()
    print(stats_main.to_string(float_format=lambda x: f"{x:.2f}"))
    
    print("\nSubset - By topic_section (count / mean / min / max):")
    stats_sub = df_sub.groupby('topic_section')['n_options'].agg(['count', 'mean', 'min', 'max']).sort_index()
    print(stats_sub.to_string(float_format=lambda x: f"{x:.2f}"))
    
    # 4) modal_share distribution
    header("MODAL_SHARE DISTRIBUTION")
    print("\nMain - Overall statistics:")
    modal_main = df_main['modal_share'].describe()
    print(modal_main.to_string())
    
    print("\nSubset - Overall statistics:")
    modal_sub = df_sub['modal_share'].describe()
    print(modal_sub.to_string())
    
    print("\nMain - By topic_section (mean / std / min / max):")
    modal_stats_main = df_main.groupby('topic_section')['modal_share'].agg(['count', 'mean', 'std', 'min', 'max']).sort_index()
    print(modal_stats_main.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"))
    
    print("\nSubset - By topic_section (mean / std / min / max):")
    modal_stats_sub = df_sub.groupby('topic_section')['modal_share'].agg(['count', 'mean', 'std', 'min', 'max']).sort_index()
    print(modal_stats_sub.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"))
    
    # Show difference in mean modal_share by section
    print("\nDifference in mean modal_share by section (subset - main):")
    for section in sorted(set(modal_stats_main.index) | set(modal_stats_sub.index)):
        main_mean = modal_stats_main.loc[section, 'mean'] if section in modal_stats_main.index else None
        sub_mean = modal_stats_sub.loc[section, 'mean'] if section in modal_stats_sub.index else None
        if main_mean is not None and sub_mean is not None:
            diff = sub_mean - main_mean
            print(f"  {section:30s}: {sub_mean:.3f} - {main_mean:.3f} = {diff:+.3f}")
    
    # Three-way comparison if no-country subset provided
    if df_no_country is not None:
        header("THREE-WAY COMPARISON")
        print("\nMain (all rich profiles):")
        print(f"  Instances: {len(df_main):,}")
        print(f"  Questions: {df_main['question'].nunique():,}")
        
        print("\nCountry-in-profile subset:")
        print(f"  Instances: {len(df_sub):,} ({len(df_sub)/len(df_main)*100:.1f}%)")
        print(f"  Questions: {df_sub['question'].nunique():,}")
        
        print("\nNo-country-in-profile subset:")
        print(f"  Instances: {len(df_no_country):,} ({len(df_no_country)/len(df_main)*100:.1f}%)")
        print(f"  Questions: {df_no_country['question'].nunique():,}")
        
        total_subset = len(df_sub) + len(df_no_country)
        if abs(total_subset - len(df_main)) < 100:  # Allow small rounding differences
            print(f"\n[OK] Verification: {len(df_sub):,} + {len(df_no_country):,} = {total_subset:,} ~= {len(df_main):,} (main)")
        else:
            print(f"\n[WARNING] Subset sizes don't add up: {len(df_sub):,} + {len(df_no_country):,} = {total_subset:,} != {len(df_main):,}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Main dataset:   {len(df_main):,} instances, {df_main['question'].nunique():,} unique questions")
    print(f"Subset dataset: {len(df_sub):,} instances, {df_sub['question'].nunique():,} unique questions")
    print(f"Retention rate: {len(df_sub) / len(df_main) * 100:.1f}% of instances")
    print(f"Question retention: {df_sub['question'].nunique() / df_main['question'].nunique() * 100:.1f}% of questions")


if __name__ == '__main__':
    main()
