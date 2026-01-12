#!/usr/bin/env python3
"""
Generate validation test instances from base prediction instances.

This script takes base instances (output from generate_base_instances.py)
and generates test instances for the three validation tests:

  1. Surface Form Sensitivity - Tests stability across answer phrasings
  2. Options Context - Tests whether showing options affects predictions
  3. Feature Order Robustness - Tests sensitivity to profile feature ordering

Usage:
    python scripts/generate_validation_tests.py --input <base_instances.jsonl> [options]

Options:
    --input PATH       Path to base instances JSONL file (required)
    --output_dir DIR   Output directory for test files (default: same as input)
    --n_samples N      Number of test instances per test type (default: 500)
    --seed S           Random seed (default: 42)
    --skip_surface     Skip Surface Form test generation
    --skip_options     Skip Options Context test generation
    --skip_feature     Skip Feature Order test generation
    --prefix STR       Prefix for output filenames

Output:
    - surface_form_test.jsonl
    - options_context_test.jsonl
    - feature_order_test.jsonl

Examples:
    # Standard usage (after running generate_base_instances.py)
    python scripts/generate_validation_tests.py --input output/base_instances_20250110.jsonl
    
    # With more samples
    python scripts/generate_validation_tests.py --input output/base.jsonl --n_samples 1000
    
    # Only generate specific tests
    python scripts/generate_validation_tests.py --input output/base.jsonl --skip_feature
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from synthetic_sampling.approach_tests import (
    SurfaceFormTestGenerator,
    OptionsContextTestGenerator,
    FeatureOrderTestGenerator,
)


def find_latest_base_instances(output_dir: Path) -> Path | None:
    """Find the most recent base instances file in output directory."""
    candidates = list(output_dir.glob("**/base_instances*.jsonl")) + \
                 list(output_dir.glob("**/*multi_survey*.jsonl"))
    
    if not candidates:
        return None
    
    # Sort by modification time, most recent first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_base_instances(path: Path) -> list:
    """Load base instances from JSONL file."""
    instances = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    return instances


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation test instances from base predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all tests from specific input file
    python scripts/generate_validation_tests.py --input output/base_instances.jsonl
    
    # Generate with custom sample size
    python scripts/generate_validation_tests.py --input data.jsonl --n_samples 1000
    
    # Generate only Surface Form and Options Context tests
    python scripts/generate_validation_tests.py --input data.jsonl --skip_feature
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to base instances JSONL file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        help="Output directory for test files (default: same directory as input)"
    )
    parser.add_argument(
        "--n_samples", "-n",
        type=int,
        default=500,
        help="Number of test instances per test type (default: 500)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip_surface",
        action="store_true",
        help="Skip Surface Form test generation"
    )
    parser.add_argument(
        "--skip_options",
        action="store_true",
        help="Skip Options Context test generation"
    )
    parser.add_argument(
        "--skip_feature",
        action="store_true",
        help="Skip Feature Order test generation"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g., 'wvs_' -> 'wvs_surface_form_test.jsonl')"
    )
    
    args = parser.parse_args()
    
    # Find input file
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
    else:
        # Try to find in default locations
        repo_root = Path(__file__).parent.parent
        output_dir = repo_root / 'output'
        
        if output_dir.exists():
            input_path = find_latest_base_instances(output_dir)
        else:
            input_path = None
        
        if input_path is None:
            print("Error: No input file specified and couldn't find base instances in output/")
            print("Usage: python scripts/generate_validation_tests.py --input <path_to_base_instances.jsonl>")
            sys.exit(1)
        
        print(f"Found base instances: {input_path}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print("=" * 70)
    print("VALIDATION TEST INSTANCE GENERATION")
    print("=" * 70)
    print(f"Input:        {input_path}")
    print(f"Output dir:   {output_dir}")
    print(f"N samples:    {args.n_samples}")
    print(f"Seed:         {args.seed}")
    print(f"Prefix:       {args.prefix or '(none)'}")
    print()
    
    # Load base instances
    print("Loading base instances...")
    base_instances = load_base_instances(input_path)
    print(f"  Loaded {len(base_instances):,} instances")
    
    # Show survey distribution
    from collections import Counter
    survey_counts = Counter(inst.get('survey', 'unknown') for inst in base_instances)
    print(f"  Surveys: {dict(survey_counts)}")
    print()
    
    # Check if we have enough instances
    if len(base_instances) < args.n_samples:
        print(f"  Warning: Requested {args.n_samples} samples but only {len(base_instances)} available")
        print(f"  Will generate as many as possible")
    
    results = {}
    
    # Test 1: Surface Form Sensitivity
    if not args.skip_surface:
        print("-" * 70)
        print("Test 1: Surface Form Sensitivity")
        print("-" * 70)
        
        generator = SurfaceFormTestGenerator(seed=args.seed)
        instances = generator.generate(
            base_instances,
            n_samples=args.n_samples,
            require_variations=True,
            stratify_by_variation=True,
        )
        
        output_file = output_dir / f"{args.prefix}surface_form_test.jsonl"
        generator.save_jsonl(instances, output_file)
        
        print(f"Generated {len(instances):,} instances")
        generator.print_stats()
        print(f"Saved to: {output_file}")
        
        results['surface_form'] = {
            'n_instances': len(instances),
            'output_file': str(output_file),
        }
        print()
    
    # Test 2: Options Context
    if not args.skip_options:
        print("-" * 70)
        print("Test 2: Options Context Sensitivity")
        print("-" * 70)
        
        generator = OptionsContextTestGenerator(seed=args.seed)
        instances = generator.generate(
            base_instances,
            n_samples=args.n_samples,
            stratify_by_type=True,
        )
        
        output_file = output_dir / f"{args.prefix}options_context_test.jsonl"
        generator.save_jsonl(instances, output_file)
        
        print(f"Generated {len(instances):,} instances")
        generator.print_stats()
        print(f"Saved to: {output_file}")
        
        results['options_context'] = {
            'n_instances': len(instances),
            'output_file': str(output_file),
        }
        print()
    
    # Test 3: Feature Order Robustness
    if not args.skip_feature:
        print("-" * 70)
        print("Test 3: Feature Order Robustness")
        print("-" * 70)
        
        generator = FeatureOrderTestGenerator(
            seed=args.seed,
            n_random_orderings=2,  # 2-3 as per paper spec
        )
        instances = generator.generate(
            base_instances,
            n_samples=args.n_samples,
            stratify_by_n_features=True,
        )
        
        output_file = output_dir / f"{args.prefix}feature_order_test.jsonl"
        generator.save_jsonl(instances, output_file)
        
        print(f"Generated {len(instances):,} instances")
        generator.print_stats()
        print(f"Saved to: {output_file}")
        
        results['feature_order'] = {
            'n_instances': len(instances),
            'output_file': str(output_file),
        }
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Base instances:     {len(base_instances):,}")
    
    for test_name, info in results.items():
        print(f"{test_name:20s} {info['n_instances']:,} instances -> {Path(info['output_file']).name}")
    
    print()
    print(f"Output directory: {output_dir.absolute()}")
    
    # List output files with sizes
    print("\nGenerated files:")
    for test_name, info in results.items():
        p = Path(info['output_file'])
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()