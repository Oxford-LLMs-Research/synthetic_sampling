#!/usr/bin/env python
"""Find target codes that appear in multiple surveys."""
import pandas as pd
from collections import defaultdict

df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv')

# Extract survey and target_code from question identifier
df['survey'] = df['question'].str.split('_').str[0]
df['target_code'] = df['question'].str.split('_', n=1).str[1]

# Find target codes that appear in multiple surveys
target_code_surveys = defaultdict(set)
for _, row in df.iterrows():
    target_code_surveys[row['target_code']].add(row['survey'])

# Find duplicates
duplicates = {code: surveys for code, surveys in target_code_surveys.items() if len(surveys) > 1}

print(f"Found {len(duplicates)} target codes that appear in multiple surveys:")
print("=" * 80)

for code in sorted(duplicates.keys()):
    surveys = sorted(duplicates[code])
    print(f"\n{code}: appears in {len(surveys)} surveys")
    print(f"  Surveys: {', '.join(surveys)}")
    
    # Show the full question identifiers
    question_ids = sorted(df[df['target_code'] == code]['question'].unique())
    for qid in question_ids:
        print(f"    - {qid}")

print(f"\n\nSummary:")
print(f"  Total unique target codes: {len(target_code_surveys)}")
print(f"  Target codes in multiple surveys: {len(duplicates)}")
print(f"  Target codes in single survey: {len(target_code_surveys) - len(duplicates)}")
print(f"  Total unique question identifiers (survey_target_code): {df['question'].nunique()}")
