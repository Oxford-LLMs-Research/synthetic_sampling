#!/usr/bin/env python3
"""Check target codes in results files."""

import json
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from synthetic_sampling.evaluation import load_results

result_file = r'C:\Users\murrn\cursor\synthetic_sampling\results\deepseek-v3p1-terminus\deepseek-v3p1-terminus_survey_arabbarometer_results.jsonl'

print("Checking example_id format and target codes...")
print("=" * 70)

# Check raw JSON
with open(result_file, 'r') as f:
    lines = [json.loads(l) for l in f.readlines()[:20]]

print("\nFirst 10 example_ids (raw):")
for i, line in enumerate(lines[:10]):
    print(f"  {i+1}. {line.get('example_id', 'N/A')}")

# Check parsed
instances = load_results(result_file)
print(f"\n\nParsed {len(instances)} instances")
print("\nFirst 10 parsed target codes:")
for i, inst in enumerate(instances[:10]):
    print(f"  {i+1}. example_id={inst.example_id}, target_code={inst.target_code}")

# Get unique targets
all_targets = set([inst.target_code for inst in instances])
print(f"\n\nTotal unique target codes in this file: {len(all_targets)}")
print("\nSample target codes:")
for t in sorted(list(all_targets))[:30]:
    print(f"  {t}")
