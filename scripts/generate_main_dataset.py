#!/usr/bin/env python
"""
Main Dataset Generation Script for LLM Survey Prediction Experiments.

This script orchestrates the complete data generation pipeline:
1. Target sampling (adaptive stratified across sections)
2. Respondent sampling
3. Profile generation at 3 richness levels (sparse/medium/rich) with strict superset
4. Instance serialization to JSONL

For ESS surveys, handles country-specific variables via concept sampling and resolution.

Profile Richness Levels:
- Sparse: 3 sections × 2 features = 6 features
- Medium: 4 sections × 3 features = 12 features  
- Rich:   6 sections × 4 features = 24 features

Usage:
    python scripts/generate_main_dataset.py --config config.yaml
    python scripts/generate_main_dataset.py --survey ess_wave_11 --n_respondents 100 --output test.jsonl
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
import numpy as np
import pandas as pd

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# Import from synthetic_sampling package
from synthetic_sampling.targets import (
    sample_targets_stratified,
    SampledTarget,
    ESS_CONCEPT_CONFIGS,
)


# =============================================================================
# Profile Richness Configuration
# =============================================================================

@dataclass
class ProfileRichnessConfig:
    """Configuration for a profile richness level."""
    name: str           # e.g., 'sparse', 'medium', 'rich'
    n_sections: int     # Number of sections to sample from
    m_features: int     # Features per section
    
    @property
    def total_features(self) -> int:
        return self.n_sections * self.m_features
    
    @property
    def type_code(self) -> str:
        """Short code like 's3m2' for sparse (3 sections, 2 features each)."""
        return f"s{self.n_sections}m{self.m_features}"


# Default richness levels with strict subset relationships
RICHNESS_LEVELS = [
    ProfileRichnessConfig('sparse', n_sections=3, m_features=2),   # 6 features
    ProfileRichnessConfig('medium', n_sections=4, m_features=3),   # 12 features
    ProfileRichnessConfig('rich', n_sections=6, m_features=4),     # 24 features
]


# =============================================================================
# ESS Concept Resolution
# =============================================================================

class ConceptResolver:
    """
    Resolves ESS country-specific concepts to actual variable codes.
    
    Handles concepts like 'party_voted' which map to different variables
    per country (e.g., prtvtdat for Austria, prtvgde1 for Germany).
    """
    
    def __init__(self, metadata: Dict, concept_configs: Dict[str, Dict] = None):
        """
        Initialize resolver with survey metadata.
        
        Parameters
        ----------
        metadata : dict
            Full survey metadata with questions
        concept_configs : dict, optional
            Concept configurations (defaults to ESS_CONCEPT_CONFIGS)
        """
        self.metadata = metadata
        self.concept_configs = concept_configs or ESS_CONCEPT_CONFIGS
        
        # Build concept -> country -> variable mapping
        self._build_concept_maps()
    
    def _build_concept_maps(self):
        """Build mappings from concepts to country-specific variables."""
        self.concept_to_vars = defaultdict(dict)  # concept_id -> {country: var_code}
        self.var_to_concept = {}  # var_code -> concept_id
        
        all_var_codes = set(self.metadata.get('questions', {}).keys())
        
        for concept_id, config in self.concept_configs.items():
            prefix = config['prefix']
            
            for var_code in all_var_codes:
                if var_code.startswith(prefix):
                    # Extract country code from variable name
                    country = self._extract_country(var_code, prefix)
                    if country:
                        self.concept_to_vars[concept_id][country] = var_code
                        self.var_to_concept[var_code] = concept_id
    
    def _extract_country(self, var_code: str, prefix: str) -> Optional[str]:
        """Extract country code from variable name."""
        suffix = var_code[len(prefix):]
        
        # Handle patterns like 'prtvgde1' -> 'de', 'prtvtdat' -> 'at'
        # Country code is typically 2 chars before any trailing digits
        suffix_no_digits = suffix.rstrip('0123456789')
        
        if len(suffix_no_digits) >= 2:
            return suffix_no_digits[-2:].upper()
        return None
    
    def get_variable_for_country(
        self, 
        concept_id: str, 
        country: str
    ) -> Optional[str]:
        """
        Get the variable code for a concept in a specific country.
        
        Parameters
        ----------
        concept_id : str
            Concept identifier (e.g., 'party_voted')
        country : str
            Country code (e.g., 'DE', 'AT')
            
        Returns
        -------
        str or None
            Variable code for this concept in this country, or None if not available
        """
        country_upper = country.upper()
        return self.concept_to_vars.get(concept_id, {}).get(country_upper)
    
    def get_variable_metadata(self, var_code: str) -> Optional[Dict]:
        """Get metadata for a specific variable."""
        return self.metadata.get('questions', {}).get(var_code)
    
    def get_all_concept_variables(self, concept_id: str) -> Set[str]:
        """Get all variable codes associated with a concept."""
        return set(self.concept_to_vars.get(concept_id, {}).values())
    
    def get_all_concepts(self) -> List[str]:
        """Get list of all concept IDs."""
        return list(self.concept_to_vars.keys())
    
    def is_country_specific_var(self, var_code: str) -> bool:
        """Check if a variable is country-specific."""
        return var_code in self.var_to_concept


# =============================================================================
# Profile Generation with Superset Constraint
# =============================================================================

def expand_profile_superset(
    base_features: Dict[str, Any],
    base_sections: List[str],
    target_config: ProfileRichnessConfig,
    metadata: Dict,
    respondent_data: Dict,
    exclude_codes: Set[str],
    rng: np.random.RandomState,
    missing_patterns: Set[str] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Expand a profile to a richer level while preserving all base features.
    
    Guarantees strict superset: all features in base profile appear in expanded profile.
    
    Parameters
    ----------
    base_features : dict
        Features from the sparser profile (code -> answer)
    base_sections : list
        Sections used in the sparser profile
    target_config : ProfileRichnessConfig
        Target richness configuration
    metadata : dict
        Survey metadata
    respondent_data : dict
        Respondent's full data row
    exclude_codes : set
        Variable codes to exclude (target + related)
    rng : np.random.RandomState
        Random state for reproducibility
    missing_patterns : set, optional
        Patterns indicating missing values
        
    Returns
    -------
    tuple
        (expanded_features dict, sections_used list)
    """
    if missing_patterns is None:
        missing_patterns = {
            'missing', 'refused', "don't know", 'no answer',
            'not asked', 'not applicable', 'nan', 'n/a', ''
        }
    
    # Start with base features
    expanded = deepcopy(base_features)
    sections_used = list(base_sections)
    
    # Get all available sections
    all_sections = list(metadata.get('sections', {}).keys())
    
    # Need to add more sections?
    sections_to_add = target_config.n_sections - len(sections_used)
    if sections_to_add > 0:
        available_sections = [s for s in all_sections if s not in sections_used]
        if len(available_sections) >= sections_to_add:
            new_sections = rng.choice(
                available_sections, 
                size=sections_to_add, 
                replace=False
            ).tolist()
            sections_used.extend(new_sections)
    
    # For each section, ensure we have enough features
    questions = metadata.get('questions', {})
    
    for section in sections_used:
        # Get current features from this section
        section_questions = metadata.get('sections', {}).get(section, [])
        current_section_features = {
            code: val for code, val in expanded.items()
            if code in section_questions
        }
        
        # Need more features from this section?
        features_needed = target_config.m_features - len(current_section_features)
        
        if features_needed > 0:
            # Get available questions from this section
            available_qs = [
                q for q in section_questions
                if q not in expanded
                and q not in exclude_codes
                and q in questions
            ]
            
            # Oversample to handle missing values
            oversample_factor = 3
            sample_size = min(features_needed * oversample_factor, len(available_qs))
            
            if sample_size > 0:
                candidates = rng.choice(
                    available_qs,
                    size=sample_size,
                    replace=False
                ).tolist()
                
                added = 0
                for q_code in candidates:
                    if added >= features_needed:
                        break
                    
                    # Get respondent's answer
                    answer = respondent_data.get(q_code)
                    
                    # Skip missing values
                    if answer is None:
                        continue
                    answer_str = str(answer).lower().strip()
                    if answer_str in missing_patterns:
                        continue
                    
                    expanded[q_code] = answer
                    added += 1
    
    return expanded, sections_used


def generate_profile_hierarchy(
    respondent_data: Dict,
    target_code: str,
    metadata: Dict,
    richness_levels: List[ProfileRichnessConfig],
    exclude_codes: Set[str],
    seed: int,
    missing_patterns: Set[str] = None
) -> List[Tuple[ProfileRichnessConfig, Dict[str, Any], List[str]]]:
    """
    Generate a hierarchy of profiles with strict superset relationships.
    
    Parameters
    ----------
    respondent_data : dict
        Respondent's data
    target_code : str
        Target question code
    metadata : dict
        Survey metadata
    richness_levels : list
        List of ProfileRichnessConfig, ordered sparse to rich
    exclude_codes : set
        Codes to exclude from profiles
    seed : int
        Random seed
    missing_patterns : set, optional
        Missing value patterns
        
    Returns
    -------
    list of tuples
        [(config, features_dict, sections_used), ...] for each level
    """
    if missing_patterns is None:
        missing_patterns = {
            'missing', 'refused', "don't know", 'no answer',
            'not asked', 'not applicable', 'nan', 'n/a', ''
        }
    
    rng = np.random.RandomState(seed)
    profiles = []
    
    # Generate sparsest profile first
    base_config = richness_levels[0]
    base_features, base_sections = _generate_base_profile(
        respondent_data=respondent_data,
        config=base_config,
        metadata=metadata,
        exclude_codes=exclude_codes,
        rng=rng,
        missing_patterns=missing_patterns
    )
    profiles.append((base_config, base_features, base_sections))
    
    # Expand to richer levels
    current_features = base_features
    current_sections = base_sections
    
    for config in richness_levels[1:]:
        expanded_features, expanded_sections = expand_profile_superset(
            base_features=current_features,
            base_sections=current_sections,
            target_config=config,
            metadata=metadata,
            respondent_data=respondent_data,
            exclude_codes=exclude_codes,
            rng=rng,
            missing_patterns=missing_patterns
        )
        profiles.append((config, expanded_features, expanded_sections))
        current_features = expanded_features
        current_sections = expanded_sections
    
    return profiles


def _generate_base_profile(
    respondent_data: Dict,
    config: ProfileRichnessConfig,
    metadata: Dict,
    exclude_codes: Set[str],
    rng: np.random.RandomState,
    missing_patterns: Set[str]
) -> Tuple[Dict[str, Any], List[str]]:
    """Generate the base (sparsest) profile."""
    all_sections = list(metadata.get('sections', {}).keys())
    questions = metadata.get('questions', {})
    
    # Sample sections
    if len(all_sections) >= config.n_sections:
        sections = rng.choice(
            all_sections, 
            size=config.n_sections, 
            replace=False
        ).tolist()
    else:
        sections = all_sections
    
    features = {}
    
    for section in sections:
        section_qs = metadata.get('sections', {}).get(section, [])
        
        # Filter to usable questions
        available = [
            q for q in section_qs
            if q not in exclude_codes
            and q in questions
        ]
        
        # Oversample
        oversample_factor = 3
        sample_size = min(config.m_features * oversample_factor, len(available))
        
        if sample_size > 0:
            candidates = rng.choice(available, size=sample_size, replace=False).tolist()
            
            added = 0
            for q_code in candidates:
                if added >= config.m_features:
                    break
                
                answer = respondent_data.get(q_code)
                if answer is None:
                    continue
                answer_str = str(answer).lower().strip()
                if answer_str in missing_patterns:
                    continue
                
                features[q_code] = answer
                added += 1
    
    return features, sections


# =============================================================================
# Instance Generation
# =============================================================================

@dataclass
class GeneratedInstance:
    """A single generated instance for evaluation."""
    example_id: str
    base_id: str
    survey: str
    respondent_id: Any
    country: Optional[str]
    
    profile_type: str
    profile_name: str
    n_features: int
    feature_codes: List[str]
    sections_used: List[str]
    
    target_code: str
    target_section: str
    target_concept: Optional[str]  # For ESS concepts
    
    answer: str
    answer_raw: Any
    options: List[str]
    
    questions: Dict[str, str]  # code -> question text
    profile_answers: Dict[str, str]  # code -> answer text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'example_id': self.example_id,
            'base_id': self.base_id,
            'survey': self.survey,
            'respondent_id': self.respondent_id,
            'country': self.country,
            'profile_type': self.profile_type,
            'profile_name': self.profile_name,
            'n_features': self.n_features,
            'feature_codes': self.feature_codes,
            'sections_used': self.sections_used,
            'target_code': self.target_code,
            'target_section': self.target_section,
            'target_concept': self.target_concept,
            'answer': self.answer,
            'answer_raw': self.answer_raw,
            'options': self.options,
            'questions': self.questions,
            'profile_answers': self.profile_answers,
        }
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def generate_instances_for_respondent(
    respondent_data: Dict,
    respondent_id: Any,
    survey_name: str,
    targets: List[SampledTarget],
    metadata: Dict,
    richness_levels: List[ProfileRichnessConfig],
    concept_resolver: Optional[ConceptResolver] = None,
    country_col: str = 'cntry',
    seed: int = 42
) -> List[GeneratedInstance]:
    """
    Generate all instances for a single respondent.
    
    Parameters
    ----------
    respondent_data : dict
        Respondent's full data row
    respondent_id : any
        Respondent identifier
    survey_name : str
        Survey identifier
    targets : list
        Sampled target questions
    metadata : dict
        Survey metadata
    richness_levels : list
        Profile richness configurations
    concept_resolver : ConceptResolver, optional
        For ESS concept resolution
    country_col : str
        Column name for country
    seed : int
        Base random seed
        
    Returns
    -------
    list of GeneratedInstance
    """
    instances = []
    questions = metadata.get('questions', {})
    country = respondent_data.get(country_col)
    
    for target in targets:
        # Handle concept targets
        if target.is_concept and concept_resolver:
            resolved_code = concept_resolver.get_variable_for_country(
                target.concept_id, country
            )
            if resolved_code is None:
                continue  # No variable for this concept in this country
            
            actual_target_code = resolved_code
            target_concept = target.concept_id
            
            # Exclude all concept variables from features
            exclude_codes = concept_resolver.get_all_concept_variables(target.concept_id)
            exclude_codes.add(actual_target_code)
        else:
            actual_target_code = target.code
            target_concept = None
            exclude_codes = {actual_target_code}
        
        # Get target answer
        answer_raw = respondent_data.get(actual_target_code)
        if answer_raw is None:
            continue
        
        answer_str = str(answer_raw).lower().strip()
        if answer_str in {'missing', 'refused', "don't know", 'no answer', 'nan', ''}:
            continue
        
        # Get target metadata
        target_meta = questions.get(actual_target_code, {})
        options = target_meta.get('options', [])
        target_section = target.section
        
        # Generate profile hierarchy
        respondent_seed = hash((seed, respondent_id, actual_target_code)) % (2**31)
        
        profiles = generate_profile_hierarchy(
            respondent_data=respondent_data,
            target_code=actual_target_code,
            metadata=metadata,
            richness_levels=richness_levels,
            exclude_codes=exclude_codes,
            seed=respondent_seed
        )
        
        # Create instances for each profile level
        for config, features, sections in profiles:
            # Build questions dict (feature code -> question text)
            questions_dict = {}
            profile_answers = {}
            
            for code, answer in features.items():
                q_meta = questions.get(code, {})
                questions_dict[code] = q_meta.get('question_text', '')
                profile_answers[code] = str(answer)
            
            base_id = f"{survey_name}_{respondent_id}_{actual_target_code}"
            example_id = f"{base_id}_{config.type_code}"
            
            instance = GeneratedInstance(
                example_id=example_id,
                base_id=base_id,
                survey=survey_name,
                respondent_id=respondent_id,
                country=country,
                profile_type=config.type_code,
                profile_name=config.name,
                n_features=len(features),
                feature_codes=list(features.keys()),
                sections_used=sections,
                target_code=actual_target_code,
                target_section=target_section,
                target_concept=target_concept,
                answer=str(answer_raw),
                answer_raw=answer_raw,
                options=options,
                questions=questions_dict,
                profile_answers=profile_answers
            )
            
            instances.append(instance)
    
    return instances


# =============================================================================
# Main Processing
# =============================================================================

def load_metadata(path: str) -> Dict:
    """Load survey metadata from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data(path: str) -> pd.DataFrame:
    """Load survey data from CSV, DTA, or SAV."""
    path = Path(path)
    
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.dta':
        return pd.read_stata(path)
    elif path.suffix == '.sav':
        import pyreadstat
        df, _ = pyreadstat.read_sav(path)
        return df
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def process_survey(
    metadata: Dict,
    data: pd.DataFrame,
    survey_name: str,
    n_respondents: int,
    n_targets: int,
    output_path: Optional[str] = None,
    richness_levels: List[ProfileRichnessConfig] = None,
    is_ess: bool = False,
    respondent_id_col: str = 'respondent_id',
    country_col: str = 'cntry',
    seed: int = 42,
    verbose: bool = True
) -> List[GeneratedInstance]:
    """
    Process a survey and generate all instances.
    
    Parameters
    ----------
    metadata : dict
        Survey metadata
    data : pd.DataFrame
        Survey response data
    survey_name : str
        Survey identifier
    n_respondents : int
        Number of respondents to sample
    n_targets : int
        Number of targets to sample
    output_path : str, optional
        Path for JSONL output (streams if provided)
    richness_levels : list, optional
        Profile configurations
    is_ess : bool
        Whether this is an ESS survey (enables concept handling)
    respondent_id_col : str
        Column for respondent ID
    country_col : str
        Column for country
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    list of GeneratedInstance
        All generated instances (empty if streaming to file)
    """
    if richness_levels is None:
        richness_levels = RICHNESS_LEVELS
    
    rng = np.random.RandomState(seed)
    
    # Setup ESS concept handling
    concept_resolver = None
    concept_configs = None
    country_codes = None
    
    if is_ess:
        concept_resolver = ConceptResolver(metadata, ESS_CONCEPT_CONFIGS)
        concept_configs = ESS_CONCEPT_CONFIGS
        country_codes = data[country_col].unique().tolist() if country_col in data.columns else None
        
        if verbose:
            print(f"ESS mode: {len(concept_resolver.get_all_concepts())} concepts detected")
    
    # Sample targets
    if verbose:
        print(f"Sampling {n_targets} targets...")
    
    targets = sample_targets_stratified(
        metadata=metadata,
        n_targets=n_targets,
        seed=seed,
        country_codes=country_codes,
        concept_configs=concept_configs
    )
    
    if verbose:
        print(f"  Sampled {len(targets)} targets across {len(set(t.section for t in targets))} sections")
    
    # Sample respondents
    if verbose:
        print(f"Sampling {n_respondents} respondents...")
    
    respondent_indices = rng.choice(
        len(data),
        size=min(n_respondents, len(data)),
        replace=False
    )
    
    # Generate instances
    instances = []
    output_file = None
    
    if output_path:
        output_file = open(output_path, 'w', encoding='utf-8')
    
    try:
        for i, idx in enumerate(respondent_indices):
            row = data.iloc[idx].to_dict()
            resp_id = row.get(respondent_id_col, idx)
            
            resp_instances = generate_instances_for_respondent(
                respondent_data=row,
                respondent_id=resp_id,
                survey_name=survey_name,
                targets=targets,
                metadata=metadata,
                richness_levels=richness_levels,
                concept_resolver=concept_resolver,
                country_col=country_col,
                seed=seed
            )
            
            if output_file:
                for inst in resp_instances:
                    output_file.write(inst.to_jsonl() + '\n')
            else:
                instances.extend(resp_instances)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(respondent_indices)} respondents")
    
    finally:
        if output_file:
            output_file.close()
    
    if verbose:
        total = len(instances) if not output_path else (i + 1) * len(targets) * len(richness_levels)
        print(f"Generated ~{total} instances")
    
    return instances


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate main dataset for LLM survey prediction experiments'
    )
    
    parser.add_argument(
        '--metadata', type=str, required=True,
        help='Path to survey metadata JSON'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to survey data (CSV, DTA, or SAV)'
    )
    parser.add_argument(
        '--survey', type=str, required=True,
        help='Survey name/identifier'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output JSONL path'
    )
    parser.add_argument(
        '--n_respondents', type=int, default=1000,
        help='Number of respondents to sample'
    )
    parser.add_argument(
        '--n_targets', type=int, default=40,
        help='Number of target questions to sample'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--respondent_id_col', type=str, default='respondent_id',
        help='Column name for respondent ID'
    )
    parser.add_argument(
        '--country_col', type=str, default='cntry',
        help='Column name for country'
    )
    parser.add_argument(
        '--is_ess', action='store_true',
        help='Enable ESS country-specific variable handling'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading metadata from {args.metadata}...")
    metadata = load_metadata(args.metadata)
    
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"  Loaded {len(data)} respondents")
    
    # Process
    process_survey(
        metadata=metadata,
        data=data,
        survey_name=args.survey,
        n_respondents=args.n_respondents,
        n_targets=args.n_targets,
        output_path=args.output,
        is_ess=args.is_ess,
        respondent_id_col=args.respondent_id_col,
        country_col=args.country_col,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    print(f"Output written to {args.output}")


if __name__ == '__main__':
    main()