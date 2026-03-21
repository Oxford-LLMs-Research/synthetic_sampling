#!/usr/bin/env python3
"""Count how many real target questions were actually predicted."""

import json

with open('missing_targets.json', 'r') as f:
    data = json.load(f)

# Calculate real targets (excluding parsing artifacts)
real_targets_in_results = data['result_targets'] - len(data['extra_in_results'])

# Calculate successfully predicted targets
successfully_predicted = data['dataset_targets'] - len(data['missing_in_results'])

print("=" * 70)
print("TARGET QUESTIONS PREDICTION SUMMARY")
print("=" * 70)

print(f"\nDataset description: {data['dataset_targets']} unique targets")
print(f"\nModel results breakdown:")
print(f"  Total target codes found: {data['result_targets']}")
print(f"    - Parsing artifacts (fragments): {len(data['extra_in_results'])}")
print(f"    - Real target questions: {real_targets_in_results}")

print(f"\nMissing from results: {len(data['missing_in_results'])} targets")

print(f"\n" + "=" * 70)
print(f"SUCCESSFULLY PREDICTED: {successfully_predicted} out of {data['dataset_targets']} targets")
print(f"Success rate: {successfully_predicted / data['dataset_targets'] * 100:.1f}%")
print("=" * 70)

print(f"\nBreakdown:")
print(f"  - Targets in dataset: {data['dataset_targets']}")
print(f"  - Targets missing: {len(data['missing_in_results'])}")
print(f"  - Targets successfully predicted: {successfully_predicted}")
print(f"  - Parsing artifacts in results: {len(data['extra_in_results'])} (not real targets)")
