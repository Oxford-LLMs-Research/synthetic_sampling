#!/usr/bin/env python3
"""Summarize missing targets analysis."""

import json
from collections import Counter

with open('missing_targets.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("MISSING TARGETS SUMMARY")
print("=" * 70)

print(f"\nDataset targets: {data['dataset_targets']}")
print(f"Result targets: {data['result_targets']}")
print(f"Missing targets: {len(data['missing_in_results'])}")
print(f"Extra fragments: {len(data['extra_in_results'])}")
print(f"\nNet difference: {data['dataset_targets']} - {data['result_targets']} = {data['dataset_targets'] - data['result_targets']}")

print(f"\nMissing by section:")
sections = Counter([m['section'] for m in data['missing_in_results']])
for s, c in sorted(sections.items()):
    print(f"  {s}: {c} targets")

print(f"\n\nExtra fragments (parsing artifacts):")
print("These are fragments of incorrectly parsed target codes:")
for extra in sorted(data['extra_in_results'], key=lambda x: x['instances'], reverse=True):
    print(f"  {extra['target_code']}: {extra['instances']:,} instances")
