#!/usr/bin/env python
"""Quick script to check question counts in mixed effects data."""
import pandas as pd
from pathlib import Path

df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv')

print(f"Total unique questions: {df['question'].nunique()}")
print(f"\nQuestions by survey:")
for survey in sorted(df['survey'].unique()):
    survey_questions = df[df['survey'] == survey]['question'].unique()
    print(f"  {survey}: {len(survey_questions)} questions")

print(f"\nAll questions (sorted):")
questions = sorted(df['question'].unique())
for q in questions:
    print(f"  {q}")

print(f"\nQuestion breakdown by survey:")
for survey in sorted(df['survey'].unique()):
    survey_questions = sorted(df[df['survey'] == survey]['question'].unique())
    print(f"\n{survey} ({len(survey_questions)} questions):")
    for q in survey_questions:
        print(f"  {q}")

# Check if all questions have predictions from all models
print(f"\n{'='*80}")
print("Checking question coverage across models:")
print(f"{'='*80}")
all_models = sorted(df['model'].unique())
all_questions = sorted(df['question'].unique())

print(f"Total models: {len(all_models)}")
print(f"Total questions: {len(all_questions)}")

# Find questions that don't have predictions from all models
questions_missing_models = []
for question in all_questions:
    question_models = set(df[df['question'] == question]['model'].unique())
    missing_models = set(all_models) - question_models
    if missing_models:
        questions_missing_models.append((question, missing_models))

if questions_missing_models:
    print(f"\nWARNING: Found {len(questions_missing_models)} questions with missing predictions from some models:")
    for question, missing in questions_missing_models:
        print(f"  {question}: missing from {len(missing)} model(s) - {sorted(missing)}")
else:
    print(f"\nOK: All {len(all_questions)} questions have predictions from all {len(all_models)} models")

# Count questions that appear for all models
questions_with_all_models = []
for question in all_questions:
    question_models = set(df[df['question'] == question]['model'].unique())
    if question_models == set(all_models):
        questions_with_all_models.append(question)

print(f"\nQuestions with predictions from ALL models: {len(questions_with_all_models)}")
print(f"Questions missing from some models: {len(all_questions) - len(questions_with_all_models)}")
