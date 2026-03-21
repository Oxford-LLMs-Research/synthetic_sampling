#!/usr/bin/env python3
"""
Create subset of main mixed effects data where country/region are NOT in profile features.

This is the complement of the country_in_profile subset:
  main = country_in_profile ∪ no_country_in_profile

Usage:
    python create_no_country_in_profile_subset.py \\
        --main analysis/mixed_effects/mixed_effects_data.csv \\
        --country-in-profile analysis/mixed_effects/mixed_effects_data_country_in_profile.csv \\
        --output analysis/mixed_effects/mixed_effects_data_no_country_in_profile.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create subset where country/region are NOT in profile features"
    )
    parser.add_argument(
        '--main',
        type=Path,
        default=Path('analysis/mixed_effects/mixed_effects_data.csv'),
        help='Path to main dataset CSV (all rich profiles)'
    )
    parser.add_argument(
        '--country-in-profile',
        type=Path,
        default=Path('analysis/mixed_effects/mixed_effects_data_country_in_profile.csv'),
        help='Path to country-in-profile subset CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/mixed_effects/mixed_effects_data_no_country_in_profile.csv'),
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    print(f"Loading main dataset:           {args.main}")
    print(f"Loading country-in-profile:     {args.country_in_profile}")
    print(f"Output will be saved to:        {args.output}\n")
    
    # Load datasets
    df_main = pd.read_csv(args.main)
    df_country_in_profile = pd.read_csv(args.country_in_profile)
    
    print(f"Main dataset:           {len(df_main):,} instances")
    print(f"Country-in-profile:     {len(df_country_in_profile):,} instances")
    
    # Use example_id if available, otherwise construct unique identifier
    # An instance is uniquely identified by: model + respondent + question
    # (since question = survey_target_code, this uniquely identifies the instance)
    if 'example_id' in df_main.columns and 'example_id' in df_country_in_profile.columns:
        print("Using 'example_id' column as unique identifier")
        id_col = 'example_id'
    else:
        print("Constructing unique identifier from model|respondent|question")
        id_col = '_instance_id'
        df_main['_instance_id'] = (
            df_main['model'].astype(str) + '|' +
            df_main['respondent'].astype(str) + '|' +
            df_main['question'].astype(str)
        )
        df_country_in_profile['_instance_id'] = (
            df_country_in_profile['model'].astype(str) + '|' +
            df_country_in_profile['respondent'].astype(str) + '|' +
            df_country_in_profile['question'].astype(str)
        )
    
    # Find instances in main that are NOT in country_in_profile
    country_in_profile_ids = set(df_country_in_profile[id_col])
    df_no_country = df_main[~df_main[id_col].isin(country_in_profile_ids)].copy()
    
    # Drop the temporary _instance_id column if we created it
    if id_col == '_instance_id':
        df_no_country = df_no_country.drop(columns=['_instance_id'])
    
    print(f"No-country-in-profile:   {len(df_no_country):,} instances")
    print()
    
    # Verify: main should equal country_in_profile + no_country_in_profile
    total_subset = len(df_country_in_profile) + len(df_no_country)
    if total_subset != len(df_main):
        print(f"⚠ WARNING: Subset sizes don't add up!")
        print(f"  Main: {len(df_main):,}")
        print(f"  Country-in-profile: {len(df_country_in_profile):,}")
        print(f"  No-country-in-profile: {len(df_no_country):,}")
        print(f"  Sum: {total_subset:,}")
        print(f"  Difference: {len(df_main) - total_subset:,}")
    else:
        print("[OK] Verification: main = country_in_profile + no_country_in_profile")
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_no_country.to_csv(args.output, index=False)
    print(f"\n[OK] Saved to {args.output}")
    
    # Also save as Parquet if available
    try:
        parquet_path = args.output.with_suffix('.parquet')
        df_no_country.to_parquet(parquet_path, index=False)
        print(f"[OK] Saved to {parquet_path}")
    except ImportError:
        print("  (Parquet export requires pyarrow - skipping)")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Main dataset:")
    print(f"  Total instances: {len(df_main):,}")
    print(f"  Surveys: {df_main['survey'].nunique()}")
    print(f"  Unique questions: {df_main['question'].nunique():,}")
    print(f"  Models: {df_main['model'].nunique()}")
    
    print(f"\nCountry-in-profile subset:")
    print(f"  Total instances: {len(df_country_in_profile):,} ({len(df_country_in_profile)/len(df_main)*100:.1f}%)")
    print(f"  Surveys: {df_country_in_profile['survey'].nunique()}")
    print(f"  Unique questions: {df_country_in_profile['question'].nunique():,}")
    
    print(f"\nNo-country-in-profile subset:")
    print(f"  Total instances: {len(df_no_country):,} ({len(df_no_country)/len(df_main)*100:.1f}%)")
    print(f"  Surveys: {df_no_country['survey'].nunique()}")
    print(f"  Unique questions: {df_no_country['question'].nunique():,}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
