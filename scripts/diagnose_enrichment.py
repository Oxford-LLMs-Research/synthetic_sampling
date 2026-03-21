#!/usr/bin/env python3
"""
Diagnose why enrichment rate is low.

Checks which example_ids in results match main_data, and reports:
- Target codes in results vs main_data
- Profile types coverage
- Survey coverage
- Sample mismatches
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

def analyze_enrichment(results_dir: Path, main_data_dir: Path, survey_filter: str = None):
    """Analyze enrichment coverage."""
    
    # Load all example_ids from main_data
    main_data_ids = set()
    main_data_by_survey = defaultdict(set)
    main_data_targets = defaultdict(Counter)
    main_data_profiles = Counter()
    
    for jsonl_file in main_data_dir.glob("*_instances.jsonl"):
        survey = jsonl_file.name.replace("_instances.jsonl", "")
        if survey_filter and survey_filter not in survey:
            continue
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    eid = obj.get('example_id', '')
                    if eid:
                        main_data_ids.add(eid)
                        main_data_by_survey[survey].add(eid)
                        
                        # Parse target from example_id
                        if '_' in eid:
                            parts = eid.rsplit('_', 2)
                            if len(parts) >= 2:
                                target = parts[-2]
                                profile = parts[-1]
                                main_data_targets[survey][target] += 1
                                main_data_profiles[profile] += 1
                except:
                    pass
    
    print(f"Main data: {len(main_data_ids):,} total example_ids")
    print(f"  Surveys: {list(main_data_by_survey.keys())}")
    print(f"  Profile types: {dict(main_data_profiles)}")
    print()
    
    # Check results
    results_by_model = {}
    
    for model_folder in results_dir.iterdir():
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        results_ids = set()
        results_targets = defaultdict(Counter)
        results_profiles = Counter()
        results_by_survey = defaultdict(set)
        
        for jsonl_file in model_folder.glob("*_results.jsonl"):
            survey = None
            for s in main_data_by_survey.keys():
                if s in jsonl_file.name:
                    survey = s
                    break
            
            if survey_filter and survey_filter not in (survey or ""):
                continue
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        eid = obj.get('example_id', '')
                        if eid:
                            results_ids.add(eid)
                            if survey:
                                results_by_survey[survey].add(eid)
                            
                            # Parse target
                            if '_' in eid:
                                parts = eid.rsplit('_', 2)
                                if len(parts) >= 2:
                                    target = parts[-2]
                                    profile = parts[-1]
                                    results_targets[survey or 'unknown'][target] += 1
                                    results_profiles[profile] += 1
                    except:
                        pass
        
        if results_ids:
            matched = results_ids & main_data_ids
            results_by_model[model_name] = {
                'total': len(results_ids),
                'matched': len(matched),
                'match_rate': len(matched) / len(results_ids) if results_ids else 0,
                'targets': dict(results_targets),
                'profiles': dict(results_profiles),
                'surveys': list(results_by_survey.keys()),
            }
    
    # Report
    print("=" * 70)
    print("ENRICHMENT DIAGNOSIS")
    print("=" * 70)
    print()
    
    for model_name, data in results_by_model.items():
        print(f"Model: {model_name}")
        print(f"  Results: {data['total']:,} instances")
        print(f"  Matched: {data['matched']:,} ({data['match_rate']:.1%})")
        print(f"  Surveys: {data['surveys']}")
        print(f"  Profile types: {dict(data['profiles'])}")
        print()
        
        # Compare targets per survey
        for survey in data['surveys']:
            if survey not in main_data_targets:
                continue
            
            results_t = set(data['targets'].get(survey, {}).keys())
            main_t = set(main_data_targets[survey].keys())
            
            print(f"  {survey}:")
            print(f"    Results targets: {len(results_t)}")
            print(f"    Main data targets: {len(main_t)}")
            print(f"    Overlap: {len(results_t & main_t)}")
            
            missing_in_results = main_t - results_t
            missing_in_main = results_t - main_t
            
            if missing_in_results:
                print(f"    Targets in main_data but NOT in results ({len(missing_in_results)}):")
                for t in sorted(list(missing_in_results))[:10]:
                    print(f"      {t} (main_data: {main_data_targets[survey][t]:,} instances)")
                if len(missing_in_results) > 10:
                    print(f"      ... and {len(missing_in_results) - 10} more")
            
            if missing_in_main:
                print(f"    Targets in results but NOT in main_data ({len(missing_in_main)}):")
                for t in sorted(list(missing_in_main))[:10]:
                    print(f"      {t} (results: {data['targets'][survey][t]:,} instances)")
                if len(missing_in_main) > 10:
                    print(f"      ... and {len(missing_in_main) - 10} more")
            print()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Diagnose enrichment coverage')
    ap.add_argument('--results-dir', type=Path, required=True, help='Results directory')
    ap.add_argument('--main-data', type=Path, required=True, help='Main data directory')
    ap.add_argument('--survey', type=str, default=None, help='Filter to specific survey')
    args = ap.parse_args()
    
    analyze_enrichment(args.results_dir, args.main_data, args.survey)
