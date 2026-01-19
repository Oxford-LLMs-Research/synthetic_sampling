#!/usr/bin/env python
"""
Main Dataset Generation Script for LLM Survey Prediction Experiments.

Uses existing SURVEY_REGISTRY configuration to generate instances.

Usage:
    # Generate for a single survey
    python scripts/generate_main_dataset.py --survey ess_wave_11
    
    # Generate for all surveys
    python scripts/generate_main_dataset.py --all
    
    # Generate for ESS surveys only
    python scripts/generate_main_dataset.py --ess-only
    
    # List available surveys
    python scripts/generate_main_dataset.py --list
    
    # Custom paths
    python scripts/generate_main_dataset.py --survey wvs \
        --raw-data-dir /path/to/data \
        --metadata-dir /path/to/metadata \
        --output-dir /path/to/output
"""

import sys
import argparse
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from synthetic_sampling.config.surveys import (
    SURVEY_REGISTRY,
    ESS_SURVEYS,
    ALL_SURVEYS,
    get_survey_config,
    list_surveys_detailed,
    SurveyConfig,
)
from synthetic_sampling.config import load_config

from synthetic_sampling.targets import (
    sample_targets_stratified,
    ESS_CONCEPT_CONFIGS,
)
from synthetic_sampling.profiles.generator import RespondentProfileGenerator

# =============================================================================
# Profile Richness Configuration  
# =============================================================================

RICHNESS_LEVELS = {
    'sparse': {'n_sections': 3, 'm_features': 2},   # 6 features
    'medium': {'n_sections': 4, 'm_features': 3},   # 12 features  
    'rich':   {'n_sections': 6, 'm_features': 4},   # 24 features
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_respondent_target_seed(base_seed: int, respondent_id: Any, target_code: str) -> int:
    """
    Generate a unique but reproducible seed for a respondent x target combination.
    
    This ensures:
    1. Same (respondent_id, target_code, base_seed) always produces same profile
    2. Different respondent x target combinations get different random sequences
    3. Changing base_seed changes all profiles
    4. Profiles for the same respondent x target are supersets (via expand_profile)
    """
    combined = f"{base_seed}_{respondent_id}_{target_code}"
    hash_val = int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)
    return hash_val


# =============================================================================
# Data Loading
# =============================================================================

def convert_to_native_types(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    # Handle arrays first to avoid ambiguous truth value errors
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    
    # Check for NaN/NA values (only for scalar values, not arrays)
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # pd.isna() can fail for some types, continue
        pass
    
    # Convert numpy scalar types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle dicts
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    
    return obj

def load_survey_data(
    survey_config: SurveyConfig,
    raw_data_dir: Path,
    metadata_dir: Path
) -> Tuple[Dict, pd.DataFrame]:
    """
    Load metadata and response data for a survey.
    
    Parameters
    ----------
    survey_config : SurveyConfig
        Survey configuration from SURVEY_REGISTRY
    raw_data_dir : Path
        Base directory containing survey data folders
    metadata_dir : Path
        Base directory containing metadata JSONs
        
    Returns
    -------
    tuple
        (metadata dict, response DataFrame)
    """
    # Load metadata
    metadata_path = metadata_dir / survey_config.metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Find data file
    data_dir = raw_data_dir / survey_config.folder_name
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Prioritize numeric formats (.dta, .sav) over CSV to preserve numeric codes
    # CSV files often have pre-converted text labels which cause mapping issues
    data_file = None
    preferred_formats = ['.dta', '.sav']  # Prefer these over CSV
    
    # First, try to find preferred numeric formats
    for preferred_ext in preferred_formats:
        matches = list(data_dir.glob(f'*{preferred_ext}'))
        if matches:
            data_file = matches[0]
            break
    
    # If no preferred format found, fall back to file patterns
    if data_file is None:
        for pattern in survey_config.get_file_patterns():
            matches = list(data_dir.glob(pattern))
            if matches:
                # Skip CSV if we haven't found preferred formats yet
                if matches[0].suffix == '.csv' and any(
                    (data_dir / f).suffix in preferred_formats 
                    for f in data_dir.iterdir() 
                    if f.is_file()
                ):
                    # There might be a .dta or .sav file, continue searching
                    continue
                data_file = matches[0]
                break
    
    if data_file is None:
        raise FileNotFoundError(
            f"No data file in {data_dir} matching: {survey_config.file_patterns}"
        )
    
    # Load based on extension
    print(f"  Loading: {data_file.name} ({data_file.suffix})")
    
    if data_file.suffix == '.csv':
        data = pd.read_csv(data_file, encoding=survey_config.encoding)
    elif data_file.suffix == '.dta':
        data = pd.read_stata(data_file)
    elif data_file.suffix == '.sav':
        import pyreadstat
        data, _ = pyreadstat.read_sav(str(data_file))
    else:
        raise ValueError(f"Unsupported format: {data_file.suffix}")
    
    # Construct respondent ID if needed
    if survey_config.id_columns_to_combine:
        cols = list(survey_config.id_columns_to_combine)
        data[survey_config.respondent_id_col] = data[cols].astype(str).agg(
            survey_config.id_separator.join, axis=1
        )
    
    return metadata, data


# =============================================================================
# Instance Generation (simplified - uses existing generator if available)
# =============================================================================

def process_survey(
    survey_id: str,
    raw_data_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    n_respondents: int = 1200,
    n_targets: int = 50,
    seed: int = 42,
    verbose: bool = True,
    similarity_model: str = 'all-MiniLM-L6-v2',
    similarity_threshold: float = 0.85
) -> Optional[Path]:
    """
    Process a single survey and generate instances.
    
    Returns output file path if successful.
    """
    survey_config = get_survey_config(survey_id)
    is_ess = survey_id in ESS_SURVEYS
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {survey_config.name} ({survey_id})")
        print(f"{'='*60}")
    
    # Load data
    try:
        metadata, data = load_survey_data(survey_config, raw_data_dir, metadata_dir)
        if verbose:
            print(f"  Loaded {len(data)} respondents")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None
    
    rng = np.random.RandomState(seed)
    
    # Get country codes for ESS concept handling
    country_codes = None
    concept_configs = None
    if is_ess:
        country_col = survey_config.country_col
        if country_col in data.columns:
            country_codes = data[country_col].unique().tolist()
        concept_configs = ESS_CONCEPT_CONFIGS
        if verbose:
            print(f"  ESS mode: {len(country_codes or [])} countries")
    
    # Sample targets
    if verbose:
        print(f"  Sampling {n_targets} targets...")
    
    targets, sampling_metadata = sample_targets_stratified(
        metadata=metadata,
        n_targets=n_targets,
        seed=seed,
        country_codes=country_codes,
        concept_configs=concept_configs
    )
    
    if verbose:
        sections = set(t.section for t in targets)
        print(f"    Got {len(targets)} targets across {len(sections)} sections")
    
    # Sample respondents
    n_sample = min(n_respondents, len(data))
    respondent_indices = rng.choice(len(data), size=n_sample, replace=False)
    
    if verbose:
        print(f"  Sampled {n_sample} respondents")
    
    # Output file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{survey_id}_instances.jsonl"
    
    # Extract target codes (skip ESS concepts for now)
    target_codes = [t.var_code for t in targets if not (t.is_concept and is_ess)]
    
    if not target_codes:
        if verbose:
            print(f"  No valid targets found (all are ESS concepts)")
        return None
    
    # Create generator with semantic filtering enabled
    if verbose:
        print(f"  Initializing RespondentProfileGenerator with semantic filtering...")
        print(f"    Model: {similarity_model}")
        print(f"    Threshold: {similarity_threshold}")
    
    # Verify sentence-transformers is available before creating generator
    if similarity_model is not None:
        try:
            import sentence_transformers
            if verbose:
                print(f"    [OK] sentence-transformers {sentence_transformers.__version__} available")
        except ImportError as e:
            print(f"\n  ERROR: sentence-transformers package not found in current Python environment.")
            print(f"  Install with: pip install sentence-transformers")
            print(f"  Or check that you're using the correct Python environment.")
            print(f"  Current Python: {sys.executable}")
            raise
    
    # Use comprehensive missing value patterns
    # NOTE: "Don't know" is a VALID response (respondent has no opinion) - we want to predict it
    # Only exclude survey/interview artifacts like "Not asked", "Refused", "Missing", etc.
    missing_value_labels = [
        'Missing', 'Refused', 'No answer', 'Not asked',
        'Not applicable', 'Decline to answer', "Can't choose", 
        'Do not understand', 'Not available', 'No response'
    ]
    missing_value_patterns = [
        'missing', 'refused', 'no answer', 'not asked',
        'not applicable', 'decline', "can't", 'do not understand',
        'not available', 'no response', 'nan', 'na', 'n/a'
        # NOTE: "don't know" is intentionally NOT in this list - it's a valid response
    ]
    
    generator = RespondentProfileGenerator(
        survey_data=data,
        metadata=metadata,
        respondent_id_col=survey_config.respondent_id_col,
        country_col=survey_config.country_col,
        survey=survey_id,
        missing_value_labels=missing_value_labels,
        missing_value_patterns=missing_value_patterns,
        similarity_model=similarity_model,
        similarity_threshold=similarity_threshold
    )
    
    # Set target questions (this triggers semantic similarity computation)
    if verbose:
        print(f"  Setting {len(target_codes)} target questions...")
    generator.set_target_questions(target_codes)
    
    # Generate instances
    if verbose:
        print(f"  Generating instances...")
    
    instance_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, idx in enumerate(respondent_indices):
            row = data.iloc[idx]
            resp_id = row[survey_config.respondent_id_col] if survey_config.respondent_id_col in row.index else idx
            resp_id = convert_to_native_types(resp_id)
            
            # Generate instances for each target at each richness level
            for target in targets:
                # Handle ESS concepts
                if target.is_concept and is_ess:
                    continue
                
                target_code = target.var_code
                
                # Generate unique seed for this respondent x target combination
                # This ensures different combinations get different features,
                # while maintaining superset relationships within the same combination
                respondent_target_seed = get_respondent_target_seed(seed, resp_id, target_code)
                
                # Generate profiles at each richness level with strict superset relationships
                # Use expand_profile to maintain superset relationships while applying
                # semantic filtering at each expansion step
                base_profile = None
                
                for level_name, level_config in RICHNESS_LEVELS.items():
                    n_sections = level_config['n_sections']
                    m_features = level_config['m_features']
                    
                    # Generate or expand profile
                    if base_profile is None:
                        # First level: generate base profile with semantic filtering
                        try:
                            base_profile = generator.generate_profile(
                                respondent_id=resp_id,
                                n_sections=n_sections,
                                m_features_per_section=m_features,
                                seed=respondent_target_seed,  # Use unique seed per respondent x target
                                shuffle_features=False,
                                target_code=target_code  # Apply per-target semantic exclusions
                            )
                        except (ValueError, KeyError) as e:
                            # Skip if profile generation fails (e.g., insufficient features)
                            if verbose and i == 0:
                                print(f"    Warning: Could not generate profile for {target_code} at {level_name}: {e}")
                            break
                    else:
                        # Subsequent levels: expand existing profile
                        # This maintains superset relationships while respecting semantic exclusions
                        add_sections = n_sections - base_profile.config.n_sections
                        add_features = m_features - base_profile.config.m_features_per_section
                        
                        if add_sections < 0 or add_features < 0:
                            # This shouldn't happen if RICHNESS_LEVELS is ordered correctly
                            continue
                        
                        try:
                            base_profile = generator.expand_profile(
                                profile=base_profile,
                                add_sections=add_sections,
                                add_features_per_section=add_features,
                                target_code=target_code  # Apply per-target semantic exclusions during expansion
                            )
                        except (ValueError, KeyError) as e:
                            # Skip if expansion fails
                            if verbose and i == 0:
                                print(f"    Warning: Could not expand profile for {target_code} at {level_name}: {e}")
                            break
                    
                    # Generate prediction instance from profile
                    instance = generator.generate_prediction_instance_from_profile(
                        profile=base_profile,
                        target_code=target_code,
                        skip_missing_targets=True
                    )
                    
                    if instance is None:
                        # Skip if target answer is missing
                        continue
                    
                    # Convert to dict format matching expected output
                    profile_type_code = f"s{n_sections}m{m_features}"
                    instance_dict = {
                        'example_id': f"{survey_id}_{resp_id}_{target_code}_{profile_type_code}",
                        'base_id': f"{survey_id}_{resp_id}_{target_code}",
                        'survey': survey_id,
                        'id': str(instance.id),
                        'country': str(instance.country) if instance.country is not None else None,
                        'questions': instance.features,
                        'target_question': instance.target_question,
                        'target_code': target_code,
                        'target_section': instance.target_section,
                        'target_topic_tag': target.topic_tag,
                        'target_response_format': target.response_format,
                        'answer': instance.answer,
                        'options': convert_to_native_types(instance.options),
                        'profile_type': profile_type_code,
                    }
                    
                    # Convert entire instance to ensure all values are JSON-serializable
                    instance_dict = convert_to_native_types(instance_dict)
                    f.write(json.dumps(instance_dict, ensure_ascii=False) + '\n')
                    instance_count += 1
            
            if verbose and (i + 1) % 200 == 0:
                print(f"    Processed {i + 1}/{n_sample} respondents...")
    
    if verbose:
        print(f"\n  [OK] Wrote {instance_count} instances to: {output_path}")
    
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset using SURVEY_REGISTRY configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Survey selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--survey', type=str, help='Single survey ID')
    group.add_argument('--surveys', type=str, nargs='+', help='Multiple survey IDs')
    group.add_argument('--all', action='store_true', help='All surveys')
    group.add_argument('--ess-only', action='store_true', help='ESS surveys only')
    group.add_argument('--list', action='store_true', help='List available surveys')
    
    # Config
    parser.add_argument('--config', type=str, default='configs/local.yaml',
                       help='Path to YAML config file (default: configs/local.yaml)')
    
    # Paths (can override config)
    parser.add_argument('--raw-data-dir', type=str, default=None,
                       help='Base directory for survey data (overrides config)')
    parser.add_argument('--metadata-dir', type=str, default=None,
                       help='Base directory for metadata (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (overrides config)')
    
    # Sampling
    parser.add_argument('--n-respondents', type=int, default=1200)
    parser.add_argument('--n-targets', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    
    # Semantic filtering
    parser.add_argument('--similarity-model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model for semantic filtering (default: all-MiniLM-L6-v2)')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                       help='Similarity threshold for semantic filtering (default: 0.85)')
    
    args = parser.parse_args()
    
    # List and exit
    if args.list:
        print(list_surveys_detailed())
        print(f"\nSurvey groups:")
        print(f"  ESS_SURVEYS: {ESS_SURVEYS}")
        print(f"  ALL_SURVEYS: {ALL_SURVEYS}")
        return
    
    # Determine surveys to process
    if args.survey:
        surveys = [args.survey]
    elif args.surveys:
        surveys = args.surveys
    elif args.all:
        surveys = ALL_SURVEYS
    elif args.ess_only:
        surveys = ESS_SURVEYS
    else:
        parser.print_help()
        print("\nSpecify --survey, --surveys, --all, --ess-only, or --list")
        return
    
    # Validate
    invalid = [s for s in surveys if s not in SURVEY_REGISTRY]
    if invalid:
        print(f"Unknown survey(s): {invalid}")
        print(f"Available: {list(SURVEY_REGISTRY.keys())}")
        return
    
    # Load config for paths (if not overridden)
    config_path = repo_root / args.config
    if config_path.exists():
        config = load_config(config_path)
        paths = config['paths']
        default_raw_data = paths.raw_data_dir
        default_metadata = paths.metadata_dir
        default_output = paths.output_dir
    else:
        # Fallback defaults if config not found
        default_raw_data = Path('data/raw')
        default_metadata = Path('data')
        default_output = Path('data/generated')
    
    # Use provided paths or config defaults
    raw_data_dir = Path(args.raw_data_dir) if args.raw_data_dir else default_raw_data
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else default_metadata
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    
    print(f"Configuration:")
    print(f"  Raw data:    {raw_data_dir}")
    print(f"  Metadata:    {metadata_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Surveys:     {surveys}")
    print(f"  Respondents: {args.n_respondents}")
    print(f"  Targets:     {args.n_targets}")
    print(f"  Semantic filtering: {args.similarity_model} (threshold={args.similarity_threshold})")
    
    # Process
    results = {}
    for survey_id in surveys:
        try:
            path = process_survey(
                survey_id=survey_id,
                raw_data_dir=raw_data_dir,
                metadata_dir=metadata_dir,
                output_dir=output_dir,
                n_respondents=args.n_respondents,
                n_targets=args.n_targets,
                seed=args.seed,
                verbose=not args.quiet,
                similarity_model=args.similarity_model,
                similarity_threshold=args.similarity_threshold
            )
            results[survey_id] = 'success' if path else 'skipped'
        except Exception as e:
            print(f"ERROR: {e}")
            results[survey_id] = f'error: {e}'
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        symbol = '[OK]' if status == 'success' else '[ERROR]'
        print(f"  {symbol} {name}: {status}")


if __name__ == '__main__':
    main()