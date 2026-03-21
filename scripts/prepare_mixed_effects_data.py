#!/usr/bin/env python3
"""
Prepare data for mixed effects model analysis.

Creates a DataFrame with all required fields for:
correct ~ n_options + modal_share + model + region + topic_section + (1|survey)

This specification uses fixed effects for model, region, and topic_section to enable
direct comparisons between levels, and a random effect for survey to account for
survey-level variation.

Uses country_canonical_mapping.json to convert raw country values to ISO-2 codes,
then country_to_region.json to map ISO-2 codes to regions. This ensures all
regions from the disaggregated analysis are available (not just 6 European regions).

Usage:
    python prepare_mixed_effects_data.py \\
        --results-dir results/ \\
        --inputs outputs/main_data \\
        --output analysis/mixed_effects_data.csv
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

import pandas as pd

# Add src and scripts to path
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_src_dir))

from shared_data_cache import get_all_models_enriched
# Reuse helpers from disaggregated analysis to identify instances where
# country/region questions are explicitly included in the profile features.
from analyze_disaggregated import (
    load_input_data_with_profile_features,
    filter_instances_by_country_in_profile,
)


def _normalize_raw(raw: Optional[str]) -> str:
    """Normalize raw country value: strip whitespace, remove .0 suffix."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _to_canonical_iso2(
    country_raw: Optional[str],
    survey: str,
    canonical: Dict[str, Any],
) -> Optional[str]:
    """
    Convert raw country value to ISO-2 using canonical mapping.
    
    Uses country_canonical_mapping.json: by_survey first, then iso_numeric for
    WVS/Latinobarometer, then pass-through if already ISO-2.
    Matches the logic in analyze_disaggregated.py.
    """
    s = _normalize_raw(country_raw)
    if not s:
        return None
    
    by_survey = canonical.get("by_survey") or {}
    iso_numeric = canonical.get("iso_numeric") or {}
    
    # Try by_survey first
    survey_map = by_survey.get(survey)
    if survey_map is not None and s in survey_map:
        return survey_map[s]
    
    # For WVS and Latinobarometer, use iso_numeric
    if survey in ("wvs", "latinobarometer") and s in iso_numeric:
        return iso_numeric[s]
    
    # For ESS, pass through if already 2-letter code
    if survey in ("ess_wave_10", "ess_wave_11") and len(s) == 2 and s.isalpha():
        return s.upper()
    
    # Try iso_numeric as fallback
    if s in iso_numeric:
        return iso_numeric[s]
    
    return None


def prepare_mixed_effects_data(
    results_dir: Path,
    input_paths: List[Path],
    cache_dir: Path,
    region_mapping_path: Path,
    canonical_mapping_path: Path,
    output_path: Path,
    profile_filter: str = "s6m4",
    model_whitelist: Optional[List[str]] = None,
    country_in_profile_only: bool = False,
) -> pd.DataFrame:
    """
    Prepare DataFrame for mixed effects model.
    
    Returns DataFrame with columns:
    - correct (int): 0/1 outcome
    - model (str): Model name (fixed effect for comparisons)
    - region (str): Geographic region (fixed effect for comparisons)
    - topic_section (str): Target section (fixed effect for comparisons)
    - survey (str): Survey name (random effect)
    - n_options (int): Number of answer options for the question (fixed effect)
    - modal_share (float): Share of respondents who chose the most common answer (fixed effect)
    - question (str): survey_target_code (for reference)
    - country (str): ISO-2 country code (for reference)
    - target_code (str): Question code (for reference)
    - respondent (str): Respondent ID (for reference)
    
    Default model specification:
    correct ~ n_options + modal_share + model + region + topic_section + (1|survey)
    """
    # ------------------------------------------------------------------
    # Optional: load profile features and filter to instances where
    # country/region variables are explicitly present in the profile.
    # This mirrors the "country_in_profile_only" logic used in
    # analyze_disaggregated.py, ensuring we analyze the same subset.
    # ------------------------------------------------------------------

    # Load enriched instances
    # Load ALL profiles from cache (profile_filter=None) to reuse shared cache
    # Then filter to requested profile in the DataFrame step
    print("Loading enriched instances (using shared cache)...")
    print("=" * 80)
    print(f"  (Will filter to profile={profile_filter} after loading)")
    instances_by_model = get_all_models_enriched(
        results_dir=results_dir,
        input_paths=input_paths,
        cache_dir=cache_dir,
        profile_filter=None,  # Load all profiles to reuse shared cache
        model_whitelist=model_whitelist,
        force_reload=False,
        verbose=True,
    )
    
    if not instances_by_model:
        print("\nError: No instances loaded.")
        sys.exit(1)
    
    # If requested, restrict to instances with country/region in profile features
    input_data_with_features: Dict[str, Dict[str, Any]] = {}
    if country_in_profile_only:
        print("\n[Filter] Restricting to instances with country/region in profile features...")

        # Collect input JSONL paths (main_data) – prefer *_instances.jsonl
        feature_input_paths: list[str] = []
        for p in input_paths:
            if p.is_file() and p.suffix == ".jsonl":
                feature_input_paths.append(str(p))
            elif p.is_dir():
                jsonl_files = sorted(p.rglob("*.jsonl"))
                instances_files = [f for f in jsonl_files if "_instances.jsonl" in f.name]
                use_files = instances_files if instances_files else jsonl_files
                feature_input_paths.extend(str(f) for f in use_files)

        if not feature_input_paths:
            print("  Warning: No input JSONL files found for profile features; skipping country_in_profile filter.")
        else:
            print(f"  Loading profile features from {len(feature_input_paths)} JSONL file(s)...")
            input_data_with_features = load_input_data_with_profile_features(feature_input_paths)
            print(f"  Loaded profile features for {len(input_data_with_features):,} examples")

            # Apply filtering per model
            # IMPORTANT: Filter to rich profiles FIRST, then apply country-in-profile filter
            from pathlib import Path as _P  # avoid confusion with type hints above

            metadata_dir = _script_dir.parent / "src" / "synthetic_sampling" / "profiles" / "metadata"
            total_before = 0
            total_after = 0
            for model_name, instances in list(instances_by_model.items()):
                # First filter to rich profiles only (profile_filter)
                rich_instances = [inst for inst in instances if inst.profile_type == profile_filter]
                total_before += len(rich_instances)
                
                if not rich_instances:
                    print(f"  [Filter] Model {model_name}: 0 rich profile instances; dropping model from dataset.")
                    del instances_by_model[model_name]
                    continue
                
                # Then filter to country/region in profile
                filtered = filter_instances_by_country_in_profile(
                    rich_instances,
                    input_data_with_features,
                    metadata_dir=metadata_dir,
                )
                if not filtered:
                    print(f"  [Filter] Model {model_name}: 0 instances with country/region in profile; dropping model from dataset.")
                    del instances_by_model[model_name]
                else:
                    print(f"  [Filter] Model {model_name}: {len(rich_instances):,} rich -> {len(filtered):,} instances (country/region in profile)")
                    instances_by_model[model_name] = filtered
                    total_after += len(filtered)

            if not instances_by_model:
                print("\nError: After filtering, no instances remain with country/region in profile features.")
                sys.exit(1)

            print(f"\n[Filter] Global: {total_before:,} -> {total_after:,} instances (country/region in profile)")

    # Load canonical mapping (to convert raw country values to ISO-2)
    print("\nLoading canonical country mapping...")
    if not canonical_mapping_path.exists():
        print(f"Warning: Canonical mapping file not found: {canonical_mapping_path}")
        print("  Proceeding without canonical mapping (may result in many 'Unknown' regions)")
        canonical_mapping = {}
    else:
        with open(canonical_mapping_path, "r") as f:
            canonical_mapping = json.load(f)
        # Remove metadata keys
        canonical_mapping = {k: v for k, v in canonical_mapping.items() if not k.startswith("_")}
        print(f"  Loaded canonical mapping")
    
    # Load region mapping (ISO-2 -> region)
    print("\nLoading region mapping...")
    if not region_mapping_path.exists():
        print(f"Warning: Region mapping file not found: {region_mapping_path}")
        print("  Proceeding without region mapping (region will be 'Unknown')")
        country_to_region = {}
    else:
        with open(region_mapping_path, "r") as f:
            country_to_region = json.load(f)
        # Remove metadata keys
        country_to_region = {k: v for k, v in country_to_region.items() if not k.startswith("_")}
        print(f"  Loaded {len(country_to_region)} country-to-region mappings")
    
    # First pass: collect all data and calculate modal_share per question
    print("\nPreparing DataFrame...")
    print(f"  Filtering to profile: {profile_filter}")
    print("  Step 1: Collecting instances and calculating question-level statistics...")
    
    rows = []
    skipped_missing = 0
    skipped_profile = 0
    
    # Collect ground_truth values per question-respondent pair for modal_share calculation
    # Use a set to track unique respondent-question pairs (avoid duplicates across models)
    question_respondent_answers = defaultdict(dict)  # question -> {respondent_id: ground_truth}
    
    for model_name, instances in instances_by_model.items():
        for inst in instances:
            # Filter by profile type (we loaded all profiles, now filter to requested one)
            if inst.profile_type != profile_filter:
                skipped_profile += 1
                continue
            
            # Skip if missing required fields
            if not all([inst.country, inst.target_section, inst.survey, inst.target_code, inst.respondent_id]):
                skipped_missing += 1
                continue
            
            # Construct question identifier
            question = f"{inst.survey}_{inst.target_code}"
            
            # Collect ground_truth for modal_share calculation
            # Use first model's data for each respondent-question pair (all models have same ground_truth)
            # We only need one entry per respondent-question pair (avoid duplicates across models)
            if inst.ground_truth and inst.respondent_id:
                # Only store if we haven't seen this respondent-question pair yet
                if inst.respondent_id not in question_respondent_answers[question]:
                    question_respondent_answers[question][inst.respondent_id] = inst.ground_truth
            
            # Convert country to canonical ISO-2, then to region
            iso2 = _to_canonical_iso2(inst.country, inst.survey, canonical_mapping)
            if iso2:
                region = country_to_region.get(iso2, "Unknown")
            else:
                region = "Unknown"
            
            # Get n_options from the instance
            n_options = len(inst.options) if inst.options else None
            
            rows.append({
                'example_id': inst.example_id,  # Unique identifier for each instance
                'model': model_name,
                'correct': int(inst.correct),
                'respondent': inst.respondent_id,
                'region': region,
                'topic_section': inst.target_section,
                'survey': inst.survey,
                'question': question,
                'country': inst.country,  # Keep for reference
                'target_code': inst.target_code,  # Keep for reference
                'n_options': n_options,
                'ground_truth': inst.ground_truth,  # Needed for modal_share calculation
            })
    
    df = pd.DataFrame(rows)
    
    # Calculate modal_share: for each question, what % of respondents chose the modal answer
    print("  Step 2: Calculating modal_share per question...")
    
    question_modal_share = {}
    for question, respondent_answers in question_respondent_answers.items():
        if not respondent_answers:
            question_modal_share[question] = None
            continue
        
        # Get all ground_truth values (one per respondent)
        ground_truths = list(respondent_answers.values())
        
        # Count occurrences of each ground_truth
        counts = Counter(ground_truths)
        # Find the modal (most common) answer
        modal_answer, modal_count = counts.most_common(1)[0]
        # Calculate share: what % of respondents chose the modal answer
        modal_share = modal_count / len(ground_truths)
        question_modal_share[question] = modal_share
    
    # Add modal_share to dataframe
    df['modal_share'] = df['question'].map(question_modal_share)
    
    # Drop ground_truth column (no longer needed)
    df = df.drop(columns=['ground_truth'])
    
    if skipped_profile > 0:
        print(f"  Skipped {skipped_profile:,} instances (not profile {profile_filter})")
    if skipped_missing > 0:
        print(f"  Skipped {skipped_missing:,} instances with missing required fields")
    
    # Remove Unknown regions (optional - comment out if you want to keep them)
    # df = df[df['region'] != 'Unknown'].copy()
    
    print(f"\nDataset prepared:")
    print(f"  Total instances: {len(df):,}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Regions: {df['region'].nunique()}")
    print(f"  Sections: {df['topic_section'].nunique()}")
    print(f"  Surveys: {df['survey'].nunique()}")
    print(f"  Questions: {df['question'].nunique()}")
    print(f"  Overall accuracy: {df['correct'].mean():.1%}")
    
    # Show statistics for new variables
    print(f"\nNew variables:")
    print(f"  n_options: {df['n_options'].describe()}")
    print(f"  modal_share: mean={df['modal_share'].mean():.3f}, min={df['modal_share'].min():.3f}, max={df['modal_share'].max():.3f}")
    print(f"  Questions with modal_share: {df['modal_share'].notna().sum():,} / {len(df):,}")
    
    # Show breakdown by fixed and random effects
    print(f"\nFixed effects (for model):")
    print(f"  Surveys: {sorted(df['survey'].unique())}")
    print(f"  n_options range: {df['n_options'].min():.0f} - {df['n_options'].max():.0f}")
    print(f"  modal_share range: {df['modal_share'].min():.3f} - {df['modal_share'].max():.3f}")
    
    print(f"\nRandom effects (for model):")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Regions: {sorted(df['region'].unique())}")
    print(f"  Sections: {sorted(df['topic_section'].unique())}")
    
    # Check for sufficient data per level
    print(f"\nData sufficiency check:")
    min_instances_per_level = 5
    for col in ['model', 'region', 'topic_section', 'survey']:
        counts = df[col].value_counts()
        insufficient = (counts < min_instances_per_level).sum()
        if insufficient > 0:
            print(f"  {col}: {insufficient} levels with <{min_instances_per_level} instances")
        else:
            print(f"  {col}: All levels have ≥{min_instances_per_level} instances")
    
    # Save to CSV/Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    # Also save as Parquet (more efficient for large datasets)
    try:
        parquet_path = output_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False)
        print(f"✓ Saved to {parquet_path}")
    except ImportError:
        print("  (Parquet export requires pyarrow - skipping)")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare data for mixed effects model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help='Results root directory (containing model folders)'
    )
    parser.add_argument(
        '--inputs',
        type=Path,
        required=True,
        help='Input data directory (main_data) for enrichment'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/mixed_effects/mixed_effects_data.csv'),
        help='Output CSV file path'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Cache directory (default: analysis/.cache/enriched - shared with other scripts)'
    )
    parser.add_argument(
        '--region-mapping',
        type=Path,
        default=Path('scripts/country_to_region.json'),
        help='Path to country_to_region.json mapping file'
    )
    parser.add_argument(
        '--canonical-mapping',
        type=Path,
        default=Path('scripts/country_canonical_mapping.json'),
        help='Path to country_canonical_mapping.json file (converts raw country values to ISO-2)'
    )
    parser.add_argument(
        '--profile',
        type=str,
        default='s6m4',
        help='Profile filter (default: s6m4 for rich profiles)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='*',
        help='Restrict to these model names (default: all)'
    )
    parser.add_argument(
        '--country-in-profile-only',
        action='store_true',
        help='Only include instances where country/region variables are in profile features'
    )
    
    args = parser.parse_args()
    
    # Use shared cache directory so all analysis scripts can reuse the same cache
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        # Default to shared cache location (analysis/.cache/enriched)
        # This allows profile_richness, profile_richness_by_topic, disaggregated, and mixed_effects to share cache
        cache_dir = Path('analysis/.cache/enriched')
    
    df = prepare_mixed_effects_data(
        results_dir=args.results_dir,
        input_paths=[args.inputs],
        cache_dir=cache_dir,
        region_mapping_path=args.region_mapping,
        canonical_mapping_path=args.canonical_mapping,
        output_path=args.output,
        profile_filter=args.profile,
        model_whitelist=args.models,
        country_in_profile_only=args.country_in_profile_only,
    )
    
    print("\nDone!")
    print(f"\nNext steps:")
    print(f"  1. Load the DataFrame: df = pd.read_csv('{args.output}')")
    print(f"  2. Fit your mixed effects model with the formula:")
    print(f"     correct ~ n_options + modal_share + model + region + topic_section + (1|survey)")
    print(f"     (This allows direct comparisons between models, regions, and topics as fixed effects)")


if __name__ == '__main__':
    main()
