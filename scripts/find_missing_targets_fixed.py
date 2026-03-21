#!/usr/bin/env python3
"""
Find missing target questions - fixed version that correctly parses target codes with underscores.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List

# Import evaluation module
import sys
script_dir = Path(__file__).parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from synthetic_sampling.evaluation import load_results


def parse_example_id_fixed(example_id: str) -> tuple:
    """
    Correctly parse example_id into components, handling target codes with underscores.
    
    Format: "{survey}_{respondent_id}_{target_code}_{profile_type}"
    Example: "arabbarometer_700512_Q725_5_s3m2"
    
    Returns: (survey, respondent_id, target_code, profile_type)
    """
    # Profile type is always at the end: s\dm\d
    match = re.match(r'^(.+)_(s\d+m\d+)$', example_id)
    if not match:
        raise ValueError(f"Cannot parse example_id: {example_id}")
    
    prefix, profile_type = match.groups()
    
    # Known surveys
    known_surveys = [
        'afrobarometer', 'arabbarometer', 'asianbarometer', 'latinobarometer',
        'wvs', 'ess_wave_10', 'ess_wave_11'
    ]
    
    # Find which survey this is
    survey = None
    survey_prefix = None
    for s in known_surveys:
        if prefix.startswith(s + '_'):
            survey = s
            survey_prefix = s + '_'
            break
    
    if not survey:
        # Fallback: try to extract survey (first part before respondent)
        parts = prefix.split('_')
        if len(parts) >= 3:
            # Could be single-word survey or multi-word
            # Try common patterns
            if prefix.startswith('ess_'):
                survey = 'ess_wave_10' if 'wave_10' in prefix else 'ess_wave_11'
                survey_prefix = survey + '_'
            else:
                survey = parts[0]
                survey_prefix = survey + '_'
    
    if survey_prefix:
        remainder = prefix[len(survey_prefix):]
        # Now remainder is: {respondent_id}_{target_code}
        # Split on last underscore
        last_underscore = remainder.rfind('_')
        if last_underscore > 0:
            respondent_id = remainder[:last_underscore]
            target_code = remainder[last_underscore + 1:]
            return (survey, respondent_id, target_code, profile_type)
    
    # Fallback
    parts = prefix.rsplit('_', 2)
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2], profile_type)
    elif len(parts) == 2:
        return (parts[0], '', parts[1], profile_type)
    else:
        return (prefix, '', '', profile_type)


def extract_targets_from_markdown(markdown_path: Path) -> Dict[str, Dict[str, str]]:
    """Extract target codes from dataset description markdown."""
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    aggregate_start = content.find("## Aggregate Target Questions:")
    if aggregate_start == -1:
        return {}
    
    # Get everything from aggregate section to the end (or next major section)
    agg_end = content.find("\n## ", aggregate_start + 100)
    if agg_end == -1:
        aggregate_section = content[aggregate_start:]
    else:
        aggregate_section = content[aggregate_start:agg_end]
    
    # Extract metadata for each target
    target_metadata = {}
    pattern = r'^\|\s*([A-Za-z0-9_\.]+)\s*\|\s*([\d,]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|'
    
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
    
    current_section = 'unknown'
    for line in aggregate_section.split('\n'):
        # Check for section header
        section_match = re.match(r'^####\s+(.+)$', line.strip())
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        
        # Check for target row
        match = re.match(pattern, line)
        if match:
            target_code = match.group(1).strip()
            if (target_code and len(target_code) > 1 and 
                target_code.lower() not in excluded_terms and
                (target_code[0].isupper() or target_code.islower())):
                target_metadata[target_code] = {
                    'section': current_section,
                    'instances': match.group(2).strip(),
                    'response_format': match.group(3).strip(),
                    'topic_tag': match.group(4).strip(),
                    'question': match.group(5).strip(),
                }
    
    return target_metadata


def extract_targets_from_results_fixed(result_paths: List[Path]) -> Dict[str, int]:
    """Extract target codes from results using fixed parsing from evaluation module."""
    all_targets = defaultdict(int)
    
    for result_path in result_paths:
        if result_path.is_dir():
            jsonl_files = list(result_path.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                print(f"  Loading: {jsonl_file.name}")
                # Use load_results which uses the fixed parse_example_id
                instances = load_results(str(jsonl_file))
                for inst in instances:
                    all_targets[inst.target_code] += 1
        elif result_path.suffix == '.jsonl':
            print(f"  Loading: {result_path.name}")
            # Use load_results which uses the fixed parse_example_id
            instances = load_results(str(result_path))
            for inst in instances:
                all_targets[inst.target_code] += 1
    
    return dict(all_targets)


def main():
    parser = argparse.ArgumentParser(description='Find missing target questions (fixed)')
    parser.add_argument('dataset_description', type=str)
    parser.add_argument('results', nargs='+', type=str)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    dataset_desc_path = Path(args.dataset_description)
    result_paths = [Path(p) for p in args.results]
    
    print("=" * 70)
    print("FINDING MISSING TARGET QUESTIONS (FIXED PARSING)")
    print("=" * 70)
    
    print(f"\n[1/3] Extracting targets from dataset description...")
    dataset_targets = extract_targets_from_markdown(dataset_desc_path)
    print(f"  Found {len(dataset_targets)} unique target questions")
    
    print(f"\n[2/3] Extracting targets from model results (with fixed parsing)...")
    result_targets = extract_targets_from_results_fixed(result_paths)
    print(f"  Found {len(result_targets)} unique target questions in results")
    
    print(f"\n[3/3] Comparing targets...")
    dataset_codes = set(dataset_targets.keys())
    result_codes = set(result_targets.keys())
    
    missing = dataset_codes - result_codes
    extra = result_codes - dataset_codes
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nDataset description: {len(dataset_codes)} targets")
    print(f"Model results:       {len(result_codes)} targets")
    print(f"Missing in results:   {len(missing)} targets")
    print(f"Extra in results:     {len(extra)} targets")
    
    if missing:
        print(f"\n" + "=" * 70)
        print("MISSING TARGET QUESTIONS")
        print("=" * 70)
        
        missing_by_section = defaultdict(list)
        for code in sorted(missing):
            info = dataset_targets[code]
            missing_by_section[info['section']].append((code, info))
        
        for section in sorted(missing_by_section.keys()):
            targets = missing_by_section[section]
            print(f"\n### {section} ({len(targets)} targets)")
            print(f"\n| Target Code | Instances | Response Format | Topic Tag | Question |")
            print(f"|-------------|-----------|-----------------|----------|----------|")
            for code, info in sorted(targets):
                q = info['question'][:80] + "..." if len(info['question']) > 80 else info['question']
                print(f"| {code} | {info['instances']} | {info['response_format']} | {info['topic_tag']} | {q} |")
    
    if args.output:
        output_data = {
            'dataset_targets': len(dataset_codes),
            'result_targets': len(result_codes),
            'missing_in_results': [
                {'target_code': code, **dataset_targets[code]}
                for code in sorted(missing)
            ],
            'extra_in_results': [
                {'target_code': code, 'instances': result_targets[code]}
                for code in sorted(extra)
            ],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved analysis to {args.output}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
