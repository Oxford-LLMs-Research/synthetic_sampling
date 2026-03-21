#!/usr/bin/env python3
"""Verify that region mapping is working correctly after the fix."""
import pandas as pd

df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv', low_memory=False)

print("=" * 80)
print("REGION MAPPING VERIFICATION")
print("=" * 80)

print(f"\nTotal observations: {len(df):,}")
print(f"Unique regions: {df['region'].nunique()}")

unknown = df[df['region'] == 'Unknown']
print(f"\nUnknown region count: {len(unknown):,} ({100*len(unknown)/len(df):.2f}%)")

if len(unknown) > 0:
    print(f"\nWARNING: Still have {len(unknown):,} observations in Unknown region")
    print(f"   Unique countries in Unknown: {unknown['country'].nunique()}")
    print("\n   Top 10 countries in Unknown region:")
    for country, count in unknown['country'].value_counts().head(10).items():
        print(f"     {country}: {count:,}")
else:
    print("\nSUCCESS: No Unknown regions!")

print("\n" + "=" * 80)
print("All regions (sorted by count):")
print("=" * 80)
region_counts = df['region'].value_counts().sort_index()
for region, count in region_counts.items():
    pct = 100 * count / len(df)
    print(f"  {region:20s}: {count:8,} ({pct:5.2f}%)")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print(f"  Total regions: {df['region'].nunique()}")
print(f"  Regions with data: {len(region_counts)}")
print(f"  Unknown percentage: {100*len(unknown)/len(df):.2f}%")
