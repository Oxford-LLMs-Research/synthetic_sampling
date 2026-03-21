#!/usr/bin/env python3
"""Debug why specific country codes aren't mapping."""
import pandas as pd
import json

df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv', low_memory=False)
unknown = df[df['region'] == 'Unknown']

# Load canonical mapping
with open('scripts/country_canonical_mapping.json', 'r') as f:
    canonical = json.load(f)

iso_numeric = canonical.get('iso_numeric', {})

print("Debugging remaining Unknown country codes:")
print("=" * 80)

for code in [417, 462, 446]:
    # Try both integer and string matching
    subset_int = unknown[unknown['country'] == code]
    subset_str = unknown[unknown['country'] == str(code)]
    subset_float = unknown[unknown['country'] == float(code)]
    
    subset = subset_int if len(subset_int) > 0 else (subset_str if len(subset_str) > 0 else subset_float)
    
    if len(subset) == 0:
        print(f"\nCode {code}: No matches found (tried int, str, float)")
        continue
    
    print(f"\nCountry code: {code}")
    print(f"  Observations: {len(subset):,}")
    print(f"  Data type in CSV: {type(subset.iloc[0]['country']).__name__}")
    print(f"  Value as stored: {repr(subset.iloc[0]['country'])}")
    print(f"  Surveys: {set(subset['survey'].unique())}")
    
    # Test normalization
    country_raw = subset.iloc[0]['country']
    country_str = str(country_raw).strip()
    if country_str.endswith(".0"):
        country_str = country_str[:-2]
    
    print(f"  Normalized string: '{country_str}'")
    print(f"  In iso_numeric mapping: {country_str in iso_numeric}")
    if country_str in iso_numeric:
        print(f"  Should map to: {iso_numeric[country_str]}")
    
    # Check if it's a float issue
    if isinstance(country_raw, float):
        print(f"  WARNING: Stored as float! Value: {country_raw}")
        if country_raw == float(code):
            print(f"  Float matches integer code: {code}")
