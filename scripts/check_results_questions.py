#!/usr/bin/env python
"""Check unique questions in results folder for a specific model."""
import json
from pathlib import Path
from collections import defaultdict
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_sampling.evaluation import load_results

results_dir = Path(r"C:\Users\murrn\cursor\synthetic_sampling\results")

# Get first model
model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
if not model_dirs:
    print("No model directories found!")
    sys.exit(1)

model_dir = model_dirs[0]
print(f"Checking model: {model_dir.name}")
print("=" * 80)

# Load all instances
all_questions = set()
questions_by_survey = defaultdict(set)
total_instances = 0

jsonl_files = sorted(model_dir.glob("*.jsonl"))
print(f"Found {len(jsonl_files)} JSONL files\n")

for jsonl_file in jsonl_files:
    try:
        instances = load_results(str(jsonl_file))
        for inst in instances:
            if hasattr(inst, 'target_code') and hasattr(inst, 'survey'):
                question_id = f"{inst.survey}_{inst.target_code}"
                all_questions.add(question_id)
                questions_by_survey[inst.survey].add(inst.target_code)
                total_instances += 1
    except Exception as e:
        print(f"  Warning: Could not load {jsonl_file.name}: {e}")

print(f"Total instances: {total_instances:,}")
print(f"Total unique questions: {len(all_questions)}")
print(f"\nQuestions by survey:")
for survey in sorted(questions_by_survey.keys()):
    print(f"  {survey}: {len(questions_by_survey[survey])} questions")

print(f"\nAll unique questions ({len(all_questions)}):")
for q in sorted(all_questions):
    print(f"  {q}")

print(f"\nQuestions by survey (detailed):")
for survey in sorted(questions_by_survey.keys()):
    codes = sorted(questions_by_survey[survey])
    print(f"\n{survey} ({len(codes)} questions):")
    for code in codes:
        print(f"  {code}")
