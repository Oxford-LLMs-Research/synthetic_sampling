#!/usr/bin/env python3
"""
Find missing target questions between dataset description and model results.

This script compares the target questions listed in the dataset description
with those found in model results to identify which questions are missing.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List

# Import evaluation module to load results
import sys
script_dir = Path(__file__).parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from synthetic_sampling.evaluation import load_results, ResultsAnalyzer


def extract_targets_from_markdown(markdown_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Extract target codes and their metadata from the dataset description markdown.
    
    Returns dict mapping target_code -> {section, question, topic_tag, response_format}
    """
    targets = {}
    
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the aggregate summary section
    aggregate_start = content.find("## Aggregate Target Questions:")
    if aggregate_start == -1:
        print("Warning: Could not find aggregate target questions section")
        return targets
    
    # Extract from aggregate section
    aggregate_section = content[aggregate_start:]
    
    # Find all target code entries in tables
    # Pattern: | Target Code | Instances | Response Format | Topic Tag | Question |
    pattern = r'\|\s*([A-Z0-9_\.]+)\s*\|\s*(\d+(?:,\d+)?)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|'
    
    matches = re.finditer(pattern, aggregate_section)
    
    for match in matches:
        target_code = match.group(1).strip()
        instances = match.group(2).strip()
        response_format = match.group(3).strip()
        topic_tag = match.group(4).strip()
        question = match.group(5).strip()
        
        # Extract section from context (look for #### section_name before this match)
        section = 'unknown'
        section_match = re.search(r'####\s+([^\n]+)', aggregate_section[:match.start()])
        if section_match:
            section = section_match.group(1).strip()
        
        targets[target_code] = {
            'section': section,
            'question': question,
            'topic_tag': topic_tag,
            'response_format': response_format,
            'instances': instances,
        }
    
    return targets


def extract_targets_from_results(result_paths: List[Path]) -> Dict[str, int]:
    """
    Extract target codes from model results files.
    
    Returns dict mapping target_code -> count of instances
    """
    all_targets = defaultdict(int)
    
    for result_path in result_paths:
        if result_path.is_dir():
            # Load all JSONL files in directory
            jsonl_files = list(result_path.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                print(f"  Loading: {jsonl_file.name}")
                instances = load_results(str(jsonl_file))
                for inst in instances:
                    all_targets[inst.target_code] += 1
        elif result_path.suffix == '.jsonl':
            print(f"  Loading: {result_path.name}")
            instances = load_results(str(result_path))
            for inst in instances:
                all_targets[inst.target_code] += 1
    
    return dict(all_targets)


def main():
    parser = argparse.ArgumentParser(
        description='Find missing target questions between dataset and results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s outputs/dataset_description.md outputs/results/
  %(prog)s outputs/dataset_description.md outputs/results/*.jsonl
        """
    )
    parser.add_argument(
        'dataset_description',
        type=str,
        help='Path to dataset description markdown file'
    )
    parser.add_argument(
        'results',
        nargs='+',
        type=str,
        help='Path(s) to model results file(s) or directory containing them'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save missing targets analysis (JSON)'
    )
    
    args = parser.parse_args()
    
    dataset_desc_path = Path(args.dataset_description)
    result_paths = [Path(p) for p in args.results]
    
    print("=" * 70)
    print("FINDING MISSING TARGET QUESTIONS")
    print("=" * 70)
    
    # Extract targets from dataset description
    print(f"\n[1/3] Extracting targets from dataset description...")
    print(f"  Reading: {dataset_desc_path}")
    dataset_targets = extract_targets_from_markdown(dataset_desc_path)
    print(f"  Found {len(dataset_targets)} unique target questions")
    
    # Extract targets from results
    print(f"\n[2/3] Extracting targets from model results...")
    result_targets = extract_targets_from_results(result_paths)
    print(f"  Found {len(result_targets)} unique target questions in results")
    
    # Find missing targets
    print(f"\n[3/3] Comparing targets...")
    dataset_target_codes = set(dataset_targets.keys())
    result_target_codes = set(result_targets.keys())
    
    missing_in_results = dataset_target_codes - result_target_codes
    extra_in_results = result_target_codes - dataset_target_codes
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nDataset description: {len(dataset_target_codes)} targets")
    print(f"Model results:       {len(result_target_codes)} targets")
    print(f"Missing in results:   {len(missing_in_results)} targets")
    print(f"Extra in results:     {len(extra_in_results)} targets")
    
    if missing_in_results:
        print(f"\n" + "=" * 70)
        print("MISSING TARGET QUESTIONS (in dataset but not in results)")
        print("=" * 70)
        
        # Group by section
        missing_by_section = defaultdict(list)
        for target_code in sorted(missing_in_results):
            info = dataset_targets[target_code]
            section = info['section']
            missing_by_section[section].append((target_code, info))
        
        for section in sorted(missing_by_section.keys()):
            targets = missing_by_section[section]
            print(f"\n### {section} ({len(targets)} targets)")
            print(f"\n| Target Code | Instances | Response Format | Topic Tag | Question |")
            print(f"|-------------|-----------|-----------------|----------|----------|")
            for target_code, info in sorted(targets):
                question = info['question'][:80] + "..." if len(info['question']) > 80 else info['question']
                resp_format = info.get('response_format', '-')
                topic_tag = info.get('topic_tag', '-')
                instances = info.get('instances', '-')
                print(f"| {target_code} | {instances} | {resp_format} | {topic_tag} | {question} |")
    
    if extra_in_results:
        print(f"\n" + "=" * 70)
        print("EXTRA TARGET QUESTIONS (in results but not in dataset description)")
        print("=" * 70)
        print("\nThese targets appear in results but not in the dataset description:")
        for target_code in sorted(extra_in_results):
            count = result_targets[target_code]
            print(f"  {target_code}: {count:,} instances")
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            'dataset_targets': len(dataset_target_codes),
            'result_targets': len(result_target_codes),
            'missing_in_results': [
                {
                    'target_code': code,
                    **dataset_targets[code]
                }
                for code in sorted(missing_in_results)
            ],
            'extra_in_results': [
                {
                    'target_code': code,
                    'instances': result_targets[code]
                }
                for code in sorted(extra_in_results)
            ],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Saved analysis to {output_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
