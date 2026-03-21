#!/usr/bin/env python3
"""
Assess the impact of parsing artifacts on results calculations.

This script checks which calculations are affected by the parsing bug.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from synthetic_sampling.evaluation import load_results, ResultsAnalyzer

# Load missing targets info
with open('missing_targets.json', 'r') as f:
    missing_data = json.load(f)

# Get list of parsing artifacts
artifacts = set([e['target_code'] for e in missing_data['extra_in_results']])
missing_targets = set([m['target_code'] for m in missing_data['missing_in_results']])

print("=" * 70)
print("ASSESSING PARSING ARTIFACT IMPACT ON CALCULATIONS")
print("=" * 70)

# Load one model's results
result_dir = Path(r'C:\Users\murrn\cursor\synthetic_sampling\results\deepseek-v3p1-terminus')
all_instances = []

for jsonl_file in result_dir.glob("*.jsonl"):
    instances = load_results(str(jsonl_file))
    all_instances.extend(instances)

print(f"\nLoaded {len(all_instances):,} instances")

# Create analyzer (uses buggy parsing)
analyzer = ResultsAnalyzer(all_instances)

print(f"\nUnique target codes in ResultsAnalyzer: {len(analyzer.targets)}")

# Check which targets are artifacts
artifact_instances = defaultdict(int)
real_target_instances = defaultdict(int)

for inst in all_instances:
    if inst.target_code in artifacts:
        artifact_instances[inst.target_code] += 1
    else:
        real_target_instances[inst.target_code] += 1

print(f"\nInstances with artifact target codes: {sum(artifact_instances.values()):,}")
print(f"Instances with real target codes: {sum(real_target_instances.values()):,}")

print(f"\n" + "=" * 70)
print("IMPACT ASSESSMENT")
print("=" * 70)

print(f"\n1. Overall Metrics (accuracy, log-loss, etc.)")
print(f"   Impact: MINIMAL - These aggregate across all instances")
print(f"   Reason: Don't group by target_code")

print(f"\n2. Metrics by Target (metrics_by_target())")
print(f"   Impact: HIGH - Artifacts create fake target groups")
print(f"   Affected: {len(artifact_instances)} artifact 'targets' with {sum(artifact_instances.values()):,} instances")
print(f"   Example: Q725_4 instances grouped as '4' instead of 'Q725_4'")

print(f"\n3. Baselines (compute_baselines())")
print(f"   Impact: HIGH - Groups by survey_target_code")
print(f"   Reason: Uses f\"{{survey}}_{{target_code}}\" which includes artifacts")
print(f"   Effect: Baselines calculated per artifact instead of real target")

print(f"\n4. Heterogeneity Analysis (heterogeneity_analysis())")
print(f"   Impact: HIGH - Groups by survey_target_code")
print(f"   Reason: Variance ratios computed per target group")
print(f"   Effect: Artifacts split variance calculations incorrectly")

print(f"\n5. JS Divergence by Target (js_divergence_by_target())")
print(f"   Impact: HIGH - Groups by target_code")
print(f"   Effect: Distribution comparisons done on fragments, not real targets")

print(f"\n6. Metrics by Section/Topic (if metadata enriched)")
print(f"   Impact: MEDIUM - Depends on metadata matching")
print(f"   Reason: If metadata uses correct target codes, mismatch occurs")

print(f"\n" + "=" * 70)
print("AFFECTED CALCULATIONS IN YOUR SCRIPTS")
print("=" * 70)

print(f"\n[IMPACT] generate_main_results_figure.py:")
print(f"   - extract_question_id() has same bug (line 174)")
print(f"   - Variance ratio calculations affected")
print(f"   - Question grouping for baselines affected")

print(f"\n[IMPACT] analyze_results.py:")
print(f"   - Uses ResultsAnalyzer (buggy parsing)")
print(f"   - All target-specific metrics affected")
print(f"   - Baselines, heterogeneity, JS divergence all affected")

print(f"\n[IMPACT] analyze_disaggregated.py:")
print(f"   - Uses ResultsAnalyzer")
print(f"   - Any analysis grouping by target affected")

print(f"\n[IMPACT] compare_models.py:")
print(f"   - Uses ResultsAnalyzer")
print(f"   - Model comparisons by target affected")

print(f"\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print(f"\nThe parsing bug affects ALL calculations that group by target_code.")
print(f"This includes:")
print(f"  - Baselines (random, majority)")
print(f"  - Variance ratios")
print(f"  - JS divergence")
print(f"  - Any disaggregated analysis by target")
print(f"\nOverall metrics (total accuracy, etc.) are NOT affected.")
print(f"\nYou should fix the parse_example_id() function in evaluation.py")
