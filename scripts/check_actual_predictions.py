#!/usr/bin/env python3
"""Check how many targets actually have predictions in results."""

import sys
import re
from pathlib import Path

sys.path.insert(0, 'src')
from synthetic_sampling.evaluation import load_results

def extract_all_targets(md_path):
    """Extract all target codes from the aggregate section."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find aggregate section
    agg_start = content.find("## Aggregate Target Questions:")
    if agg_start == -1:
        return set()
    
    # Get everything from aggregate section to the end (or next major section)
    agg_end = content.find("\n## ", agg_start + 100)
    if agg_end == -1:
        agg_section = content[agg_start:]
    else:
        agg_section = content[agg_start:agg_end]
    
    # Known metadata terms to exclude
    excluded_terms = {
        'target', 'code', 'instances', 'format', 'tag', 'question',
        'binary', 'categorical', 'likert_3', 'likert_4', 'likert_5', 'likert_7', 'likert_10',
        'civic_action', 'civil_liberties', 'climate_environment', 'corruption_perceptions',
        'democratic_values', 'economic_evaluations', 'economic_policy', 'ethical_norms',
        'gender_attitudes', 'government_performance', 'government_trust', 'group_trust',
        'health', 'institutional_confidence', 'international_relations', 'interpersonal_trust',
        'life_satisfaction', 'media_information', 'migration_attitudes', 'national_ethnic_identity',
        'partisanship', 'political_efficacy', 'political_interest', 'political_priorities',
        'regime_preferences', 'religious_values', 'security_safety', 'service_delivery',
        'sexuality_attitudes', 'social_capital', 'traditionalism', 'voting',
    }
    
    # Pattern to match target code rows in tables
    pattern = r'^\|\s*([A-Za-z0-9_\.]+)\s*\|\s*[\d,]+'
    
    targets = set()
    for line in agg_section.split('\n'):
        match = re.match(pattern, line)
        if match:
            target_code = match.group(1).strip()
            # Filter out metadata terms and ensure it looks like a target code
            if (target_code and len(target_code) > 1 and 
                target_code.lower() not in excluded_terms and
                (target_code[0].isupper() or target_code.islower())):
                targets.add(target_code)
    
    return targets

result_dir = Path(r'C:\Users\murrn\cursor\synthetic_sampling\results\deepseek-v3p1-terminus')

# Load all instances
all_instances = []
for f in result_dir.glob('*.jsonl'):
    all_instances.extend(load_results(str(f)))

# Get unique targets from results
result_targets = set([inst.target_code for inst in all_instances])

# Get targets from dataset description
dataset_targets = extract_all_targets(Path('outputs/dataset_description.md'))

# Find missing
missing = dataset_targets - result_targets

print("=" * 70)
print("ACTUAL PREDICTIONS IN RESULTS")
print("=" * 70)
print(f"\nDataset description: {len(dataset_targets)} targets")
print(f"Model results (with fixed parsing): {len(result_targets)} targets")
print(f"Missing from results: {len(missing)} targets")

print(f"\n[RESULT] Actually predicted in results: {len(result_targets)} targets")

if missing:
    print(f"\nMissing (no predictions): {len(missing)} targets")
    print("\nMissing targets:")
    for t in sorted(missing):
        print(f"  - {t}")
