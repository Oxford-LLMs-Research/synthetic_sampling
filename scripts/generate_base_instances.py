#!/usr/bin/env python3
"""
Generate base prediction instances from multiple surveys.

This is the PRIMARY data generation script for the project. It uses 
DatasetBuilder to create prediction instances from survey data. The output
is used for:
  - Main experimental analysis (model evaluation)
  - Validation test generation (surface form, options context, feature order)

Usage:
    python scripts/generate_base_instances.py [options]

Options:
    --config PATH      Path to YAML config file (default: configs/local.yaml)
    --surveys LIST     Comma-separated list of surveys (default: all configured)
    --output FILE      Output filename (default: auto-generated with timestamp)
    --n_respondents N  Override n_respondents_per_survey from config
    --n_targets N      Override n_targets_per_respondent from config
    --n_sections N     Number of survey sections to sample features from
    --n_features N     Number of features per section (profile richness)
    --seed S           Override seed from config

Profile Configuration:
    The profile richness is controlled by n_sections × n_features:
    - Sparse:  2 sections × 1 feature  = 2 features  (s2m1)
    - Medium:  3 sections × 2 features = 6 features  (s3m2)
    - Rich:    5 sections × 3 features = 15 features (s5m3)

Output:
    - JSONL file with base prediction instances in output directory

Examples:
    # Generate from all default surveys
    python scripts/generate_base_instances.py
    
    # Quick test with single survey
    python scripts/generate_base_instances.py --surveys wvs --n_respondents 100
    
    # Full generation for experiments
    python scripts/generate_base_instances.py --n_respondents 1000 --n_targets 5
    
    # Configure profile richness
    python scripts/generate_base_instances.py --n_sections 3 --n_features 2
    
    # Then generate validation tests from the output:
    python scripts/generate_validation_tests.py --input output/base_instances_XXX.jsonl
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from synthetic_sampling import DatasetBuilder
from synthetic_sampling.config import load_config


# Default surveys to include
DEFAULT_SURVEYS = [
    'wvs',
    'ess_wave_10',
    'ess_wave_11', 
    'afrobarometer',
    'arabbarometer',
    'asianbarometer',
    'latinobarometer',
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate base prediction instances from survey data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from all default surveys
    python scripts/generate_base_instances.py
    
    # Generate from specific surveys
    python scripts/generate_base_instances.py --surveys wvs,ess_wave_10
    
    # Override sample size
    python scripts/generate_base_instances.py --n_respondents 500 --n_targets 5
    
    # Configure profile richness (3 sections × 2 features = 6 features per profile)
    python scripts/generate_base_instances.py --n_sections 3 --n_features 2
    
    # Rich profiles (5 sections × 3 features = 15 features per profile)
    python scripts/generate_base_instances.py --n_sections 5 --n_features 3
    
    # Use custom config
    python scripts/generate_base_instances.py --config configs/cluster.yaml
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/local.yaml",
        help="Path to YAML config file (default: configs/local.yaml)"
    )
    parser.add_argument(
        "--surveys", "-s",
        type=str,
        help=f"Comma-separated list of surveys (default: {','.join(DEFAULT_SURVEYS)})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output filename (default: auto-generated)"
    )
    parser.add_argument(
        "--n_respondents",
        type=int,
        help="Override n_respondents_per_survey from config"
    )
    parser.add_argument(
        "--n_targets",
        type=int,
        help="Override n_targets_per_respondent from config"
    )
    parser.add_argument(
        "--n_sections",
        type=int,
        help="Number of survey sections to sample features from (default: from config)"
    )
    parser.add_argument(
        "--n_features", "--m_features",
        type=int,
        dest="m_features",
        help="Number of features per section (default: from config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override seed from config"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = repo_root / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    paths = config['paths']
    dataset_config = config['dataset']
    generator_config = config['generator']
    
    # Apply overrides
    if args.n_respondents:
        dataset_config.n_respondents_per_survey = args.n_respondents
    if args.n_targets:
        dataset_config.n_targets_per_respondent = args.n_targets
    if args.n_sections:
        dataset_config.n_sections = args.n_sections
    if args.m_features:
        dataset_config.m_features_per_section = args.m_features
    if args.seed:
        dataset_config.seed = args.seed
    
    # Determine surveys
    if args.surveys:
        surveys = [s.strip() for s in args.surveys.split(',')]
    else:
        surveys = DEFAULT_SURVEYS
    
    # Print configuration
    print()
    print("=" * 70)
    print("BASE INSTANCE GENERATION")
    print("=" * 70)
    print(f"\nPaths:")
    print(f"  raw_data:  {paths.raw_data_dir}")
    print(f"  metadata:  {paths.metadata_dir}")
    print(f"  output:    {paths.output_dir}")
    
    print(f"\nDataset config:")
    print(f"  n_respondents_per_survey:  {dataset_config.n_respondents_per_survey}")
    print(f"  n_targets_per_respondent:  {dataset_config.n_targets_per_respondent}")
    print(f"  n_sections:                {dataset_config.n_sections}")
    print(f"  m_features_per_section:    {dataset_config.m_features_per_section}")
    print(f"  profile_type:              {dataset_config.profile_type_code}")
    print(f"  seed:                      {dataset_config.seed}")
    
    print(f"\nSurveys: {surveys}")
    print()
    
    # Create builder
    builder = DatasetBuilder(paths, dataset_config, generator_config)
    
    # Build dataset
    print("-" * 70)
    print("Building dataset...")
    print("-" * 70)
    
    instances = builder.build_dataset(surveys)
    
    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total instances: {len(instances):,}")
    
    # Count by survey
    survey_counts = Counter(inst['survey'] for inst in instances)
    print("\nBy survey:")
    for survey, count in sorted(survey_counts.items()):
        print(f"  {survey}: {count:,}")
    
    # Count by profile type
    profile_counts = Counter(inst.get('profile_type', 'unknown') for inst in instances)
    print("\nBy profile type:")
    for ptype, count in sorted(profile_counts.items()):
        print(f"  {ptype}: {count:,}")
    
    # Show sample instances
    print("\nSample instance IDs:")
    seen_surveys = set()
    for inst in instances:
        if inst['survey'] not in seen_surveys:
            print(f"  {inst['example_id']}")
            seen_surveys.add(inst['survey'])
            if len(seen_surveys) >= 5:
                break
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        survey_str = '_'.join(surveys[:3])
        if len(surveys) > 3:
            survey_str += f"_plus{len(surveys)-3}"
        output_file = f"base_instances_{survey_str}_{timestamp}.jsonl"
    
    # Ensure output directory exists
    output_path = paths.output_dir / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    builder.save_jsonl(instances, output_path)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Print next steps
    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("For main analysis (model evaluation):")
    print(f"  Use {output_path} as input to inference pipeline")
    print()
    print("For validation tests:")
    print(f"  python scripts/generate_validation_tests.py --input {output_path}")
    print()
    print("For quick validation test (smaller sample):")
    print(f"  python scripts/generate_validation_tests.py --input {output_path} --n_samples 200")


if __name__ == "__main__":
    main()