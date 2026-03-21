#!/usr/bin/env python3
"""Check which countries are mapped to 'Unknown' region and why."""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Load the mixed effects data
df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv', low_memory=False)
unknown = df[df['region'] == 'Unknown']

print(f"Unknown region: {len(unknown):,} observations")
print(f"Unique countries in Unknown region: {unknown['country'].nunique()}\n")

# Get top countries by count
print("Top 30 countries by count in Unknown region:")
country_counts = unknown['country'].value_counts()
for country, count in country_counts.head(30).items():
    print(f"  {country}: {count:,} observations")

# Check which surveys these come from
print("\n\nSurveys with Unknown countries:")
survey_counts = unknown['survey'].value_counts()
for survey, count in survey_counts.items():
    print(f"  {survey}: {count:,} observations")

# Load canonical mapping
canonical_path = Path('scripts/country_canonical_mapping.json')
with open(canonical_path, 'r') as f:
    canonical = json.load(f)

# Check which countries from Unknown are in canonical mapping
print("\n\nChecking if Unknown countries are in canonical mapping:")
by_survey = canonical.get('by_survey', {})
iso_numeric = canonical.get('iso_numeric', {})

unknown_countries = set(unknown['country'].unique())
found_in_canonical = set()
missing_from_canonical = set()

for country in unknown_countries:
    country_str = str(country).strip().rstrip('.0')
    
    # Check by_survey
    found = False
    for survey, survey_map in by_survey.items():
        if country_str in survey_map:
            found_in_canonical.add((country, survey, survey_map[country_str]))
            found = True
            break
    
    # Check iso_numeric
    if not found and country_str in iso_numeric:
        found_in_canonical.add((country, 'iso_numeric', iso_numeric[country_str]))
        found = True
    
    if not found:
        missing_from_canonical.add(country)

print(f"\nFound in canonical mapping: {len(found_in_canonical)}")
print(f"Missing from canonical mapping: {len(missing_from_canonical)}")

if found_in_canonical:
    print("\nCountries found in canonical mapping (but still Unknown - mapping issue?):")
    for country, source, iso2 in sorted(found_in_canonical)[:20]:
        print(f"  {country} -> {iso2} (from {source})")

if missing_from_canonical:
    print(f"\n\nCountries missing from canonical mapping (top 20):")
    for country in sorted(list(missing_from_canonical))[:20]:
        count = country_counts.get(country, 0)
        print(f"  {country}: {count:,} observations")

# Check pulled_metadata files for country mappings
print("\n\nChecking pulled_metadata files for country mappings...")
metadata_dir = Path('src/synthetic_sampling/profiles/metadata/pulled_metadata')
metadata_files = {
    'afrobarometer': metadata_dir / 'pulled_metadata_afrobarometer.json',
    'arabbarometer': metadata_dir / 'pulled_metadata_arabbarometer.json',
    'asianbarometer': metadata_dir / 'pulled_metadata_asianbarometer.json',
    'ess_wave_10': metadata_dir / 'pulled_metadata_ess10.json',
    'ess_wave_11': metadata_dir / 'pulled_metadata_ess11.json',
    'latinobarometer': metadata_dir / 'pulled_metadata_latinobarometer.json',
    'wvs': metadata_dir / 'pulled_metadata_wvs.json',
}

metadata_countries = defaultdict(dict)
for survey, filepath in metadata_files.items():
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Look for country field in metadata
        # Structure varies by survey
        if 'demographics' in metadata and 'country' in metadata['demographics']:
            country_values = metadata['demographics']['country'].get('values', {})
            metadata_countries[survey] = country_values
        elif 'B_COUNTRY' in metadata.get('demographics', {}):
            country_values = metadata['demographics']['B_COUNTRY'].get('values', {})
            metadata_countries[survey] = country_values
        elif survey == 'latinobarometer':
            # Latinobarometer might have different structure
            for section in metadata.values():
                if isinstance(section, dict) and 'country' in section:
                    country_values = section['country'].get('values', {})
                    if country_values:
                        metadata_countries[survey] = country_values
                        break

print(f"\nFound country mappings in {len(metadata_countries)} metadata files:")
for survey, country_map in metadata_countries.items():
    print(f"  {survey}: {len(country_map)} country codes")

# Check if any Unknown countries match metadata but not canonical
print("\n\nChecking if Unknown countries exist in metadata but not canonical:")
for survey in unknown['survey'].unique():
    if survey in metadata_countries:
        survey_unknown = unknown[unknown['survey'] == survey]
        survey_countries = set(survey_unknown['country'].unique())
        metadata_map = metadata_countries[survey]
        
        # Normalize country codes
        metadata_keys = {str(k).strip().rstrip('.0'): v for k, v in metadata_map.items()}
        
        found_in_metadata = []
        for country in survey_countries:
            country_str = str(country).strip().rstrip('.0')
            if country_str in metadata_keys:
                country_name = metadata_keys[country_str]
                # Check if it's in canonical
                in_canonical = False
                if survey in by_survey and country_str in by_survey[survey]:
                    in_canonical = True
                elif country_str in iso_numeric:
                    in_canonical = True
                
                if not in_canonical:
                    found_in_metadata.append((country, country_name))
        
        if found_in_metadata:
            print(f"\n  {survey} - Found in metadata but NOT in canonical mapping:")
            for country, name in found_in_metadata[:10]:
                count = country_counts.get(country, 0)
                print(f"    {country} ({name}): {count:,} observations")
