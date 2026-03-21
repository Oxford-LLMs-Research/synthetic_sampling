#!/usr/bin/env python3
"""Diagnose country mapping issues."""
import json
import pandas as pd
from pathlib import Path

# Load data
df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv', low_memory=False)
unknown = df[df['region'] == 'Unknown']

# Load canonical mapping
with open('scripts/country_canonical_mapping.json', 'r') as f:
    canonical = json.load(f)

by_survey = canonical.get('by_survey', {})
iso_numeric = canonical.get('iso_numeric', {})

print("Survey names in data:")
print(sorted(df['survey'].unique()))
print("\nSurvey names in canonical mapping:")
print(sorted(by_survey.keys()))

# Check specific problematic countries
print("\n\nChecking specific countries that should map:")
test_cases = [
    ('arabbarometer', '10'),  # Should be LB
    ('latinobarometer', '170'),  # Should be CO
    ('latinobarometer', '600'),  # Should be PY
    ('wvs', '410'),  # Should be KR
    ('wvs', '840'),  # Should be US
]

def normalize_country(raw):
    """Normalize country code: strip whitespace, remove .0 suffix."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

for survey, country_code in test_cases:
    country_str = normalize_country(country_code)
    
    # Check by_survey
    survey_map = by_survey.get(survey)
    if survey_map and country_str in survey_map:
        iso2 = survey_map[country_str]
        print(f"\n{survey} / {country_code}: Found in by_survey -> {iso2}")
    elif survey in ('wvs', 'latinobarometer') and country_str in iso_numeric:
        iso2 = iso_numeric[country_str]
        print(f"\n{survey} / {country_code}: Found in iso_numeric -> {iso2}")
    elif country_str in iso_numeric:
        iso2 = iso_numeric[country_str]
        print(f"\n{survey} / {country_code}: Found in iso_numeric (fallback) -> {iso2}")
    else:
        print(f"\n{survey} / {country_code}: NOT FOUND in canonical mapping")
    
    # Check if this exists in unknown data
    survey_unknown = unknown[(unknown['survey'] == survey) & (unknown['country'] == country_code)]
    if len(survey_unknown) > 0:
        print(f"  -> {len(survey_unknown):,} observations in Unknown region")

# Check what's actually in the data for these surveys
print("\n\nActual country codes in Unknown region by survey:")
for survey in ['arabbarometer', 'latinobarometer', 'wvs']:
    survey_unknown = unknown[unknown['survey'] == survey]
    if len(survey_unknown) > 0:
        print(f"\n{survey}:")
        top_countries = survey_unknown['country'].value_counts().head(10)
        for country, count in top_countries.items():
            country_str = normalize_country(country)
            # Try to map it
            survey_map = by_survey.get(survey)
            mapped = None
            if survey_map and country_str in survey_map:
                mapped = survey_map[country_str]
            elif survey in ('wvs', 'latinobarometer') and country_str in iso_numeric:
                mapped = iso_numeric[country_str]
            elif country_str in iso_numeric:
                mapped = iso_numeric[country_str]
            
            status = f"-> {mapped}" if mapped else "-> NOT MAPPED"
            print(f"  {country} ({country_str}): {count:,} {status}")
