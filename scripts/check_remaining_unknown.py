#!/usr/bin/env python3
"""Check why remaining Unknown countries aren't mapping."""
import pandas as pd
import json

df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv', low_memory=False)
unknown = df[df['region'] == 'Unknown']

# Load canonical mapping
with open('scripts/country_canonical_mapping.json', 'r') as f:
    canonical = json.load(f)

by_survey = canonical.get('by_survey', {})
iso_numeric = canonical.get('iso_numeric', {})

def normalize_country(raw):
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

print("Checking remaining Unknown countries:")
print("=" * 80)

for country_code in [417, 462, 446, 909]:
    country_data = unknown[unknown['country'] == country_code]
    if len(country_data) == 0:
        continue
    
    print(f"\nCountry code: {country_code}")
    print(f"  Observations: {len(country_data):,}")
    surveys = country_data['survey'].value_counts()
    print(f"  Surveys: {dict(surveys)}")
    
    # Check mapping
    country_str = normalize_country(country_code)
    print(f"  Normalized: '{country_str}'")
    
    # Check iso_numeric
    if country_str in iso_numeric:
        iso2 = iso_numeric[country_str]
        print(f"  In iso_numeric: {iso2}")
    else:
        print(f"  NOT in iso_numeric")
    
    # Check by_survey for each survey
    for survey in surveys.index:
        survey_map = by_survey.get(survey)
        if survey_map and country_str in survey_map:
            iso2 = survey_map[country_str]
            print(f"  In {survey} by_survey: {iso2}")
        else:
            print(f"  NOT in {survey} by_survey")
