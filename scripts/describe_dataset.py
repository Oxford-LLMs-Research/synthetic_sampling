#!/usr/bin/env python
"""
Describe generated dataset instances.

Provides statistics about:
- Total instances per survey
- Instances per profile type
- Target question breakdown (section, topic_tag, response_format)
"""

import json
import sys
import argparse
import io
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# Set up UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def describe_dataset(jsonl_path: str, include_targets: bool = True, markdown_file=None):
    """
    Describe a dataset JSONL file.
    
    Parameters
    ----------
    jsonl_path : str
        Path to JSONL file
    include_targets : bool
        Whether to include detailed target breakdown
    markdown_file : file-like object, optional
        If provided, write markdown output to this file
    """
    def write_output(text):
        """Write to both console and markdown file if provided."""
        print(text)
        if markdown_file:
            markdown_file.write(text + '\n')
    
    write_output(f"Analyzing: {jsonl_path}")
    write_output("=" * 70)
    
    # Statistics
    total_instances = 0
    instances_by_survey = Counter()
    instances_by_profile_type = Counter()
    base_ids = set()
    targets_info = defaultdict(lambda: {
        'count': 0,
        'section': None,
        'topic_tag': None,
        'response_format': None,
        'question': None
    })
    response_formats = Counter()
    topic_tags = Counter()
    
    # Read instances
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            instance = json.loads(line)
            total_instances += 1
            instances_by_survey[instance['survey']] += 1
            instances_by_profile_type[instance['profile_type']] += 1
            base_ids.add(instance['base_id'])
            
            # Collect target information
            if include_targets:
                target_code = instance['target_code']
                if target_code not in targets_info:
                    targets_info[target_code]['section'] = instance.get('target_section', 'unknown')
                    targets_info[target_code]['question'] = instance.get('target_question', '')
                    targets_info[target_code]['topic_tag'] = instance.get('target_topic_tag')
                    targets_info[target_code]['response_format'] = instance.get('target_response_format')
                targets_info[target_code]['count'] += 1
                
                # Count response formats and topic tags
                resp_format = instance.get('target_response_format') or 'unknown'
                response_formats[resp_format] += 1
                topic_tag = instance.get('target_topic_tag')
                if topic_tag:
                    topic_tags[topic_tag] += 1
    
    # Print summary
    write_output(f"\n## Summary")
    write_output(f"\n- **Total Instances**: {total_instances:,}")
    write_output(f"- **Unique Base IDs**: {len(base_ids):,}")
    write_output(f"- **Profile Types per Base ID**: {len(instances_by_profile_type)}")
    
    write_output(f"\n### Instances by Survey")
    write_output(f"\n| Survey | Instances |")
    write_output(f"|--------|-----------|")
    for survey, count in sorted(instances_by_survey.items()):
        write_output(f"| {survey} | {count:,} |")
    
    write_output(f"\n### Instances by Profile Type")
    write_output(f"\n| Profile Type | Instances |")
    write_output(f"|--------------|-----------|")
    for profile_type, count in sorted(instances_by_profile_type.items()):
        write_output(f"| {profile_type} | {count:,} |")
    
    if include_targets and targets_info:
        write_output(f"\n## Target Questions: {len(targets_info)} unique targets")
        
        # Show response format breakdown
        if response_formats:
            write_output(f"\n### Response Formats")
            write_output(f"\n| Format | Instances |")
            write_output(f"|--------|----------|")
            for fmt, count in sorted(response_formats.items(), key=lambda x: x[1], reverse=True):
                write_output(f"| {fmt} | {count:,} |")
        
        # Show topic tag breakdown
        if topic_tags:
            write_output(f"\n### Topic Tags: {len(topic_tags)} unique")
            write_output(f"\n| Tag | Instances |")
            write_output(f"|-----|----------|")
            for tag, count in sorted(topic_tags.items(), key=lambda x: x[1], reverse=True):
                write_output(f"| {tag} | {count:,} |")
        
        # Group by section
        targets_by_section = defaultdict(list)
        for target_code, info in targets_info.items():
            section = info['section'] or 'unknown'
            targets_by_section[section].append((target_code, info))
        
        write_output(f"\n### Targets by Section")
        for section in sorted(targets_by_section.keys()):
            targets = targets_by_section[section]
            total_instances_section = sum(info['count'] for _, info in targets)
            write_output(f"\n#### {section}")
            write_output(f"\n*{len(targets)} targets, {total_instances_section:,} total instances*")
            write_output(f"\n| Target Code | Instances | Response Format | Topic Tag | Question |")
            write_output(f"|-------------|-----------|-----------------|----------|----------|")
            for target_code, info in sorted(targets, key=lambda x: x[1]['count'], reverse=True):
                question = info['question'][:80] + "..." if len(info['question']) > 80 else info['question']
                resp_format = info.get('response_format') or '-'
                topic_tag = info.get('topic_tag') or '-'
                write_output(f"| {target_code} | {info['count']:,} | {resp_format} | {topic_tag} | {question} |")
    
    write_output("\n" + "=" * 70)


def describe_multiple_files(jsonl_paths: List[str], include_targets: bool = True, markdown_file=None):
    """Describe multiple JSONL files and provide aggregate statistics."""
    def write_output(text):
        """Write to both console and markdown file if provided."""
        print(text)
        if markdown_file:
            markdown_file.write(text + '\n')
    
    write_output("# Dataset Description")
    write_output("=" * 70)
    
    all_instances_by_survey = Counter()
    all_instances_by_profile = Counter()
    all_targets = defaultdict(lambda: {'count': 0, 'section': None, 'question': None, 'topic_tag': None, 'response_format': None})
    all_response_formats = Counter()
    all_topic_tags = Counter()
    total_instances = 0
    total_base_ids = set()
    
    # Process each file
    for jsonl_path in jsonl_paths:
        write_output(f"\n")
        describe_dataset(jsonl_path, include_targets=include_targets, markdown_file=markdown_file)
        
        # Aggregate statistics
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                instance = json.loads(line)
                total_instances += 1
                all_instances_by_survey[instance['survey']] += 1
                all_instances_by_profile[instance['profile_type']] += 1
                total_base_ids.add(instance['base_id'])
                
                if include_targets:
                    target_code = instance['target_code']
                    if target_code not in all_targets:
                        all_targets[target_code]['section'] = instance.get('target_section', 'unknown')
                        all_targets[target_code]['question'] = instance.get('target_question', '')
                        all_targets[target_code]['topic_tag'] = instance.get('target_topic_tag')
                        all_targets[target_code]['response_format'] = instance.get('target_response_format')
                    all_targets[target_code]['count'] += 1
                    
                    # Count response formats and topic tags
                    resp_format = instance.get('target_response_format') or 'unknown'
                    all_response_formats[resp_format] += 1
                    topic_tag = instance.get('target_topic_tag')
                    if topic_tag:
                        all_topic_tags[topic_tag] += 1
    
    # Print aggregate summary
    write_output("\n" + "=" * 70)
    write_output("# Aggregate Summary")
    write_output("=" * 70)
    write_output(f"\n## Overall Statistics")
    write_output(f"\n- **Total Instances (all files)**: {total_instances:,}")
    write_output(f"- **Unique Base IDs (all files)**: {len(total_base_ids):,}")
    
    write_output(f"\n## Aggregate by Survey")
    write_output(f"\n| Survey | Instances |")
    write_output(f"|--------|-----------|")
    for survey, count in sorted(all_instances_by_survey.items()):
        write_output(f"| {survey} | {count:,} |")
    
    write_output(f"\n## Aggregate by Profile Type")
    write_output(f"\n| Profile Type | Instances |")
    write_output(f"|--------------|-----------|")
    for profile_type, count in sorted(all_instances_by_profile.items()):
        write_output(f"| {profile_type} | {count:,} |")
    
    if include_targets and all_targets:
        write_output(f"\n## Aggregate Target Questions: {len(all_targets)} unique targets")
        
        # Show aggregate response format breakdown
        if all_response_formats:
            write_output(f"\n### Response Formats (Aggregate)")
            write_output(f"\n| Format | Instances |")
            write_output(f"|--------|----------|")
            for fmt, count in sorted(all_response_formats.items(), key=lambda x: x[1], reverse=True):
                write_output(f"| {fmt} | {count:,} |")
        
        # Show aggregate topic tag breakdown
        if all_topic_tags:
            write_output(f"\n### Topic Tags (Aggregate): {len(all_topic_tags)} unique")
            write_output(f"\n| Tag | Instances |")
            write_output(f"|-----|----------|")
            for tag, count in sorted(all_topic_tags.items(), key=lambda x: x[1], reverse=True):
                write_output(f"| {tag} | {count:,} |")
        
        # Group by section
        targets_by_section = defaultdict(list)
        for target_code, info in all_targets.items():
            section = info['section'] or 'unknown'
            targets_by_section[section].append((target_code, info))
        
        write_output(f"\n### Targets by Section (Aggregate)")
        for section in sorted(targets_by_section.keys()):
            targets = targets_by_section[section]
            total_in_section = sum(info['count'] for _, info in targets)
            write_output(f"\n#### {section}")
            write_output(f"\n*{len(targets)} targets, {total_in_section:,} total instances*")
            write_output(f"\n| Target Code | Instances | Response Format | Topic Tag | Question |")
            write_output(f"|-------------|-----------|-----------------|----------|----------|")
            for target_code, info in sorted(targets, key=lambda x: x[1]['count'], reverse=True):
                question = info['question'][:80] + "..." if len(info['question']) > 80 else info['question']
                resp_format = info.get('response_format') or '-'
                topic_tag = info.get('topic_tag') or '-'
                write_output(f"| {target_code} | {info['count']:,} | {resp_format} | {topic_tag} | {question} |")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Describe generated dataset instances'
    )
    parser.add_argument(
        'jsonl_files',
        nargs='+',
        type=str,
        help='Path(s) to JSONL file(s) to analyze'
    )
    parser.add_argument(
        '--no-targets',
        action='store_true',
        help='Skip detailed target breakdown'
    )
    parser.add_argument(
        '--output',
        '--markdown',
        type=str,
        help='Output markdown file path (optional)'
    )
    
    args = parser.parse_args()
    
    markdown_file = None
    if args.output:
        markdown_file = open(args.output, 'w', encoding='utf-8')
        try:
            if len(args.jsonl_files) == 1:
                describe_dataset(args.jsonl_files[0], include_targets=not args.no_targets, markdown_file=markdown_file)
            else:
                describe_multiple_files(args.jsonl_files, include_targets=not args.no_targets, markdown_file=markdown_file)
        finally:
            markdown_file.close()
            print(f"\nMarkdown output written to: {args.output}")
    else:
        if len(args.jsonl_files) == 1:
            describe_dataset(args.jsonl_files[0], include_targets=not args.no_targets)
        else:
            describe_multiple_files(args.jsonl_files, include_targets=not args.no_targets)
