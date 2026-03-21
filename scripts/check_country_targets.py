#!/usr/bin/env python3
"""Check why certain countries have no VR/JS metrics."""

import json
from pathlib import Path
from collections import defaultdict

data_file = Path('analysis/disaggregated/by_country.json')
with open(data_file, 'r') as f:
    data = json.load(f)

# Countries missing both metrics
missing_countries = ['AD', 'CY', 'MO', 'NZ', 'UZ']

print("Checking why these countries have no VR/JS metrics:\n")

for country in missing_countries:
    print(f"\n{country}:")
    print("-" * 50)
    
    # Check across all models
    total_n = 0
    model_counts = {}
    
    for model_name, model_data in data.items():
        if country in model_data:
            country_data = model_data[country]
            n = country_data.get('n', 0)
            total_n += n
            model_counts[model_name] = n
            print(f"  {model_name}: n={n}")
    
    print(f"  Total n across all models: {total_n}")
    print(f"  Average n per model: {total_n / len(model_counts) if model_counts else 0:.1f}")
    
    # The issue is likely that these countries have very few questions/targets
    # with sufficient sample size per target. The metrics require:
    # - At least 5 instances per target (adaptive_threshold)
    # - At least 2 options per question
    # - Multiple targets to aggregate across
    
    # If a country only has a few questions total, or if each question has very few
    # instances, then no targets meet the threshold and no metrics are computed.
