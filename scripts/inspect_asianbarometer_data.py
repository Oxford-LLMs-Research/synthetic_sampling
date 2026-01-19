#!/usr/bin/env python
"""
Diagnostic script to inspect Asian Barometer data format.
Checks if data has numeric codes or text labels.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from synthetic_sampling.config.surveys import get_survey_config

def inspect_question(survey_config, data, metadata, question_code):
    """Inspect a specific question to see data format vs metadata."""
    print(f"\n{'='*80}")
    print(f"Question: {question_code}")
    print(f"{'='*80}")
    
    # Get metadata
    q_meta = None
    for section, questions in metadata.items():
        if question_code in questions:
            q_meta = questions[question_code]
            print(f"Section: {section}")
            break
    
    if not q_meta:
        print(f"  ERROR: Question {question_code} not found in metadata")
        return
    
    print(f"Question text: {q_meta.get('question', 'N/A')}")
    print(f"\nMetadata values mapping:")
    values_map = q_meta.get('values', {})
    for code, label in list(values_map.items())[:10]:
        print(f"  {code} -> {label}")
    if len(values_map) > 10:
        print(f"  ... and {len(values_map) - 10} more")
    
    # Check data
    if question_code not in data.columns:
        print(f"\n  ERROR: Question {question_code} not in data columns")
        return
    
    print(f"\nData sample (first 20 non-null values):")
    non_null = data[question_code].dropna()
    unique_values = non_null.unique()[:20]
    
    for val in unique_values:
        val_str = str(val)
        # Check if it's in metadata
        in_metadata = False
        metadata_label = None
        if val_str in values_map:
            in_metadata = True
            metadata_label = values_map[val_str]
        elif val_str in values_map.values():
            in_metadata = True
            metadata_label = val_str  # Already a label
        
        status = "[IN METADATA]" if in_metadata else "[NOT IN METADATA]"
        print(f"  {val_str:30} {status:20}", end="")
        if metadata_label and metadata_label != val_str:
            print(f" -> maps to: {metadata_label}")
        else:
            print()
    
    # Check if values are numeric or text
    print(f"\nData type analysis:")
    numeric_count = 0
    text_count = 0
    in_metadata_count = 0
    not_in_metadata_count = 0
    
    for val in non_null.unique():
        val_str = str(val)
        try:
            float(val_str)
            numeric_count += 1
        except (ValueError, TypeError):
            text_count += 1
        
        if val_str in values_map or val_str in values_map.values():
            in_metadata_count += 1
        else:
            not_in_metadata_count += 1
    
    print(f"  Numeric values: {numeric_count}")
    print(f"  Text values: {text_count}")
    print(f"  Values in metadata: {in_metadata_count}")
    print(f"  Values NOT in metadata: {not_in_metadata_count}")
    
    # Check country-specific if applicable
    if survey_config.country_col in data.columns:
        print(f"\nCountry-specific analysis (sample countries):")
        countries = data[survey_config.country_col].dropna().unique()[:3]
        for country in countries:
            country_data = data[data[survey_config.country_col] == country]
            country_values = country_data[question_code].dropna().unique()[:5]
            print(f"  {country}: {list(country_values)}")

def main():
    survey_id = 'asianbarometer'
    survey_config = get_survey_config(survey_id)
    
    # Load metadata
    repo_root = Path(__file__).parent.parent
    metadata_path = repo_root / 'src' / 'synthetic_sampling' / 'profiles' / 'metadata' / survey_config.metadata_path
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Find data file
    # Try common locations
    data_dirs = [
        Path('C:/Users/murrn/cursor/synthetic_sampling/data'),
        Path('data'),
        Path('../data'),
    ]
    
    data_file = None
    for data_dir in data_dirs:
        survey_dir = data_dir / survey_config.folder_name
        if survey_dir.exists():
            for pattern in survey_config.get_file_patterns():
                matches = list(survey_dir.glob(pattern))
                if matches:
                    data_file = matches[0]
                    break
            if data_file:
                break
    
    if not data_file:
        print(f"ERROR: Could not find data file for {survey_id}")
        print(f"  Looked in: {[str(d / survey_config.folder_name) for d in data_dirs]}")
        return
    
    print(f"Loading data from: {data_file}")
    
    # Load data
    if data_file.suffix == '.csv':
        data = pd.read_csv(data_file, encoding=survey_config.encoding, low_memory=False)  # Load full file
    elif data_file.suffix == '.dta':
        data = pd.read_stata(data_file)
    elif data_file.suffix == '.sav':
        import pyreadstat
        data, _ = pyreadstat.read_sav(str(data_file))
    else:
        print(f"ERROR: Unsupported format: {data_file.suffix}")
        return
    
    print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    
    # Check if questions exist (case-insensitive)
    print(f"\nChecking for question columns...")
    all_cols = list(data.columns)
    questions_to_check_upper = ['Q185', 'Q97', 'Q30', 'Q156']
    found_questions = []
    for q_upper in questions_to_check_upper:
        q_lower = q_upper.lower()
        # Try both cases
        if q_upper in all_cols:
            found_questions.append(q_upper)
        elif q_lower in all_cols:
            print(f"  Found {q_upper} as: {q_lower}")
            found_questions.append(q_lower)
        else:
            print(f"  WARNING: {q_upper} not found in columns")
    
    # Show sample of columns that start with Q
    q_cols = [c for c in all_cols if str(c).upper().startswith('Q')][:20]
    print(f"\nSample Q columns: {q_cols}")
    
    # Inspect problematic questions (use found names)
    questions_to_check = found_questions if found_questions else ['q185', 'q97', 'q30', 'q156']
    
    for q_code in questions_to_check:
        inspect_question(survey_config, data, metadata, q_code)

if __name__ == '__main__':
    main()
