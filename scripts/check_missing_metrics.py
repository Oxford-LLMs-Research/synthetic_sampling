#!/usr/bin/env python3
"""Check which countries are missing VR and JS divergence metrics."""

import json
from pathlib import Path

data_file = Path('analysis/disaggregated/by_country.json')
with open(data_file, 'r') as f:
    data = json.load(f)

# Get all countries
all_countries = set()
for model_data in data.values():
    all_countries.update(model_data.keys())

missing_vr = []
missing_js = []
missing_both = []

for country in sorted(all_countries):
    vr_vals = []
    js_vals = []
    n_vals = []
    
    for model_name, model_data in data.items():
        if country in model_data:
            country_data = model_data[country]
            n_vals.append(country_data.get('n', 0))
            vr = country_data.get('variance_ratio_soft_median')
            js = country_data.get('js_divergence_soft_median')
            if vr is not None:
                vr_vals.append(vr)
            if js is not None:
                js_vals.append(js)
    
    if not vr_vals:
        missing_vr.append((country, max(n_vals) if n_vals else 0))
    if not js_vals:
        missing_js.append((country, max(n_vals) if n_vals else 0))
    if not vr_vals and not js_vals:
        missing_both.append((country, max(n_vals) if n_vals else 0))

print(f"Total countries: {len(all_countries)}")
print(f"\nCountries missing VR (Variance Ratio): {len(missing_vr)}")
for country, n in missing_vr:
    print(f"  {country}: n={n}")

print(f"\nCountries missing JS (Jensen-Shannon Divergence): {len(missing_js)}")
for country, n in missing_js:
    print(f"  {country}: n={n}")

print(f"\nCountries missing both: {len(missing_both)}")
for country, n in missing_both:
    print(f"  {country}: n={n}")
