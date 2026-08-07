"""
Adaptive Target Question Sampling for LLM Survey Prediction Experiments.

This module provides stratified target sampling that:
- Allocates targets proportionally to section size
- Preserves minimum features per section for profile generation
- Tracks response format diversity (binary, Likert, categorical)
- Records topic tags for post-hoc analysis
- Excludes Demographics section from target pool
- Handles ESS-style country-specific variables via concept grouping

For ESS surveys, country-specific variables (party vote, education level, etc.)
are grouped into "concepts". When sampling targets, the concept is sampled;
at evaluation time, it resolves to the respondent's country-specific variable.

Designed to integrate with the existing DatasetBuilder and RespondentProfileGenerator.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set


# =============================================================================
# ESS Country-Specific Concept Configuration
# =============================================================================

# These define groups of variables that represent the same concept across countries
# e.g., prtvtdat (Austria), prtvtebe (Belgium) all represent "party voted for"
ESS_CONCEPT_CONFIGS = [
    {
        'concept_id': 'education_level',
        'concept_name': 'Highest level of education',
        'prefixes': ['edlv'],
    },
    {
        'concept_id': 'religion_denomination', 
        'concept_name': 'Religious denomination',
        'prefixes': ['rlgdn'],
    },
    {
        'concept_id': 'religion_raised',
        'concept_name': 'Religion raised in',
        'prefixes': ['rlgde'],
    },
    {
        'concept_id': 'party_voted',
        'concept_name': 'Party voted for in last election',
        'prefixes': ['prtvt', 'prtvg', 'prtvc'],
    },
    {
        'concept_id': 'party_close',
        'concept_name': 'Party feels closest to',
        'prefixes': ['prtcl'],
    },
]

CONCEPT_MARKER = "__concept__"


def detect_country_specific_variables(
    metadata: Dict[str, Dict[str, Any]],
    country_codes: List[str],
    concept_configs: List[Dict] = None,
    min_countries: int = 3
) -> Dict[str, Dict]:
    """
    Detect country-specific variable groups in survey metadata.
    
    Returns information about which variables are country-specific and
    how they should be grouped into concepts.
    
    Parameters
    ----------
    metadata : dict
        Survey metadata (section -> var_code -> info)
    country_codes : list[str]
        Valid country codes in this survey (e.g., ['AT', 'BE', 'DE', ...])
    concept_configs : list[dict], optional
        Concept configurations. If None, uses ESS_CONCEPT_CONFIGS.
    min_countries : int
        Minimum countries for a concept to be valid
        
    Returns
    -------
    dict with keys:
        - 'concepts': dict mapping concept_id -> {vars, section, question, ...}
        - 'vars_to_exclude': set of individual variable codes to exclude from pool
        - 'concept_representatives': list of concept entries to add to pool
    """
    if concept_configs is None:
        concept_configs = ESS_CONCEPT_CONFIGS
    
    country_codes_upper = set(c.upper() for c in country_codes)
    
    # Collect all variables
    all_vars = {}
    for section, variables in metadata.items():
        if isinstance(variables, dict):
            for var_code, var_info in variables.items():
                all_vars[var_code] = {'section': section, 'info': var_info}
    
    concepts = {}
    vars_to_exclude = set()
    
    for config in concept_configs:
        concept_id = config['concept_id']
        prefixes = config['prefixes']
        
        country_vars = defaultdict(list)
        
        for var_code, var_data in all_vars.items():
            for prefix in prefixes:
                if var_code.startswith(prefix):
                    # Extract potential country code
                    country = _extract_country_code(var_code, prefix, country_codes_upper)
                    if country:
                        country_vars[country].append(var_code)
                        vars_to_exclude.add(var_code)
                    break
        
        if len(country_vars) >= min_countries:
            # Get representative info from first variable
            rep_var = None
            rep_info = None
            section = None
            
            for country, var_list in country_vars.items():
                if var_list:
                    rep_var = var_list[0]
                    rep_info = all_vars[rep_var]['info']
                    section = all_vars[rep_var]['section']
                    break
            
            concepts[concept_id] = {
                'concept_id': concept_id,
                'concept_name': config['concept_name'],
                'country_variables': dict(country_vars),
                'n_countries': len(country_vars),
                'n_total_vars': sum(len(v) for v in country_vars.values()),
                'section': section,
                'representative_question': rep_info.get('question') if rep_info else config['concept_name'],
                'representative_description': rep_info.get('description') if rep_info else config['concept_name'],
                'prefixes': prefixes,
            }
    
    # Build concept representatives for the pool
    concept_representatives = []
    for concept_id, concept_data in concepts.items():
        concept_representatives.append({
            'var_code': f"{CONCEPT_MARKER}{concept_id}",
            'section': concept_data['section'],
            'question': concept_data['representative_question'],
            'description': concept_data['representative_description'],
            'topic_tag': None,  # Concepts don't have tags
            'is_concept': True,
            'concept_id': concept_id,
            'values': {},  # Will be resolved per-country at evaluation time
        })
    
    return {
        'concepts': concepts,
        'vars_to_exclude': vars_to_exclude,
        'concept_representatives': concept_representatives,
    }


def _extract_country_code(
    var_code: str, 
    prefix: str, 
    country_codes: Set[str]
) -> Optional[str]:
    """
    Extract country code from a variable name.
    
    Handles patterns like:
    - edlveat -> AT (last 2 chars)
    - prtvgde1 -> DE (last 2 chars before digit)
    """
    if not var_code.startswith(prefix):
        return None
    
    remainder = var_code[len(prefix):]
    if len(remainder) < 2:
        return None
    
    # Strip trailing digits
    remainder_no_digits = remainder.rstrip('0123456789')
    
    if len(remainder_no_digits) < 2:
        return None
    
    # Take last 2 characters
    potential_cc = remainder_no_digits[-2:].upper()
    
    if potential_cc in country_codes:
        return potential_cc
    
    return None


def is_concept_code(code: str) -> bool:
    """Check if a code is a concept marker."""
    return code.startswith(CONCEPT_MARKER)


def get_concept_id_from_code(code: str) -> Optional[str]:
    """Extract concept_id from a concept code."""
    if is_concept_code(code):
        return code[len(CONCEPT_MARKER):]
    return None


# =============================================================================
# Response Format Detection
# =============================================================================

def detect_response_format(values_map: Dict[str, str]) -> str:
    """
    Detect the response format from a values mapping.
    
    Categories:
    - 'binary': 2 substantive options (Yes/No, Agree/Disagree, etc.)
    - 'likert_4': 4-point scale
    - 'likert_5': 5-point scale
    - 'likert_7': 7-point scale (or similar)
    - 'likert_10': 10/11-point scale (0-10)
    - 'categorical': Unordered categories (3+ options, not a scale)
    - 'other': Doesn't fit standard categories
    
    Parameters
    ----------
    values_map : dict
        Mapping of raw values to labels (e.g., {'1': 'Strongly agree', ...})
        
    Returns
    -------
    str
        Response format category
    """
    if not values_map or not isinstance(values_map, dict):
        return 'other'
    
    # Filter out missing/artifact values
    missing_patterns = [
        'missing', 'refused', "don't know", 'no answer', 'not asked',
        'not applicable', 'nan', 'n/a', 'dk', 'ra'
    ]
    
    substantive_labels = []
    for raw_val, label in values_map.items():
        label_lower = str(label).lower()
        is_missing = any(pattern in label_lower for pattern in missing_patterns)
        if not is_missing:
            substantive_labels.append(label)
    
    n_options = len(substantive_labels)
    
    if n_options == 0:
        return 'other'
    elif n_options == 2:
        return 'binary'
    elif n_options == 3:
        # Could be categorical or short Likert - check for scale patterns
        if _looks_like_scale(substantive_labels):
            return 'likert_3'
        return 'categorical'
    elif n_options == 4:
        if _looks_like_scale(substantive_labels):
            return 'likert_4'
        return 'categorical'
    elif n_options == 5:
        if _looks_like_scale(substantive_labels):
            return 'likert_5'
        return 'categorical'
    elif n_options in [6, 7]:
        if _looks_like_scale(substantive_labels):
            return 'likert_7'
        return 'categorical'
    elif n_options >= 10:
        # Check for 0-10 or 1-10 numeric scales
        if _looks_like_numeric_scale(values_map):
            return 'likert_10'
        return 'categorical'
    else:  # 8-9 options
        if _looks_like_scale(substantive_labels):
            return 'likert_7'  # Group with 7-point
        return 'categorical'


def _looks_like_scale(labels: List[str]) -> bool:
    """
    Heuristic: does this look like an ordinal scale?
    
    Checks for common scale patterns in labels.
    """
    scale_indicators = [
        'strongly', 'somewhat', 'slightly', 'very', 'fairly', 'quite',
        'agree', 'disagree', 'approve', 'disapprove',
        'satisfied', 'dissatisfied', 'trust', 'distrust',
        'likely', 'unlikely', 'often', 'never', 'always', 'sometimes',
        'good', 'bad', 'excellent', 'poor', 'fair',
        'important', 'unimportant',
        'confident', 'not confident',
        'more', 'less', 'same'
    ]
    
    labels_lower = ' '.join(str(l).lower() for l in labels)
    matches = sum(1 for indicator in scale_indicators if indicator in labels_lower)
    
    # If we find 2+ scale indicators, likely a scale
    return matches >= 2


def _looks_like_numeric_scale(values_map: Dict[str, str]) -> bool:
    """
    Check if values form a numeric scale (0-10, 1-10, etc.)
    """
    try:
        numeric_values = [int(v) for v in values_map.keys() if v.lstrip('-').isdigit()]
        if len(numeric_values) >= 10:
            # Check if consecutive or near-consecutive
            sorted_vals = sorted(numeric_values)
            range_span = sorted_vals[-1] - sorted_vals[0]
            # Allow for some missing values in sequence
            return range_span <= len(numeric_values) + 2
    except (ValueError, TypeError):
        pass
    return False


def has_valid_fixed_options(values_map: Dict[str, str]) -> bool:
    """
    Check if a question has valid fixed options suitable for prediction.
    
    Excludes:
    - Questions with no values mapping
    - Questions with only missing values
    - Questions that appear to be numeric/open-ended (many consecutive numeric values)
    
    Note: Does NOT exclude questions with many categorical options (e.g., party choice,
    religion) as these are valid fixed-option questions, just with many categories.
    
    Parameters
    ----------
    values_map : dict
        Mapping of raw values to labels
        
    Returns
    -------
    bool
        True if question has valid fixed options for prediction
    """
    if not values_map or not isinstance(values_map, dict):
        return False
    
    # Filter out missing/artifact values
    missing_patterns = [
        'missing', 'refused', "don't know", 'no answer', 'not asked',
        'not applicable', 'nan', 'n/a', 'dk', 'ra', 'decline',
        'do not understand', "can't choose"
    ]
    
    substantive_labels = []
    substantive_raw_values = []
    for raw_val, label in values_map.items():
        label_lower = str(label).lower()
        is_missing = any(pattern in label_lower for pattern in missing_patterns)
        if not is_missing:
            substantive_labels.append(label)
            substantive_raw_values.append(raw_val)
    
    # Must have at least 2 substantive options (binary minimum)
    if len(substantive_labels) < 2:
        return False
    
    # Check if this looks like a numeric/open-ended question
    # (many consecutive numeric values, like age 18-99)
    # vs. categorical with many options (like party/religion with text labels)
    # Key difference: numeric questions have numeric LABELS, categorical have text labels
    if len(substantive_raw_values) > 20:
        # Check if LABELS are mostly numeric (indicates numeric question like age)
        # vs. text labels (indicates categorical like party/religion)
        numeric_labels = 0
        for raw_val, label in values_map.items():
            label_lower = str(label).lower()
            is_missing = any(pattern in label_lower for pattern in missing_patterns)
            if is_missing:
                continue
            
            # Check if label is just a number (like "18", "19", "30", "60")
            # vs. text (like "Democratic Party", "Catholic")
            label_stripped = str(label).strip()
            try:
                # If label can be parsed as a number and is just the number, it's numeric
                float(label_stripped)
                # Also check if it's a simple integer representation
                if label_stripped == str(int(float(label_stripped))):
                    numeric_labels += 1
            except (ValueError, TypeError):
                # Label is text - this is categorical
                pass
        
        # If most labels are numeric, check if they form a consecutive sequence
        if numeric_labels > len(substantive_labels) * 0.7:
            # Most labels are numeric - check if they form consecutive sequence
            try:
                numeric_label_values = []
                for raw_val, label in values_map.items():
                    label_lower = str(label).lower()
                    is_missing = any(pattern in label_lower for pattern in missing_patterns)
                    if is_missing:
                        continue
                    try:
                        num = float(str(label).strip())
                        numeric_label_values.append(num)
                    except (ValueError, TypeError):
                        pass
                
                if len(numeric_label_values) > 20:
                    sorted_nums = sorted(numeric_label_values)
                    range_span = sorted_nums[-1] - sorted_nums[0]
                    # If the range is close to the count, it's likely consecutive
                    # (e.g., 18-99 = 81 values for 82 options = dense sequence)
                    if range_span <= len(sorted_nums) * 1.5:
                        # This looks like a numeric scale (age, hours, etc.) - exclude
                        return False
            except Exception:
                # If we can't analyze, err on the side of inclusion
                pass
    
    return True


# =============================================================================
# Target Question Dataclass
# =============================================================================

@dataclass
class SampledTarget:
    """
    A target question selected for prediction evaluation.
    
    Contains all metadata needed for:
    - Generator configuration (var_code, section)
    - Analysis (topic_tag, response_format)
    - Documentation (question text)
    - Country-specific resolution (is_concept, concept_id)
    
    For concept targets (is_concept=True), var_code is a marker like 
    '__concept__party_voted' that gets resolved to the respondent's 
    country-specific variable at evaluation time.
    """
    var_code: str
    section: str
    question: str
    values: Dict[str, str]
    topic_tag: Optional[str] = None
    response_format: str = field(default='other')
    is_concept: bool = False
    concept_id: Optional[str] = None
    
    def __post_init__(self):
        if self.response_format == 'other' and not self.is_concept:
            # Only detect format if values is a dict (not a string like "String variable")
            if isinstance(self.values, dict):
                self.response_format = detect_response_format(self.values)
        elif self.is_concept:
            # Concepts have dynamic response formats per country
            self.response_format = 'country_specific'
    
    def to_dict(self) -> Dict[str, Any]:
        # Safely compute n_options only if values is a dict
        n_options = 0
        if self.values and isinstance(self.values, dict):
            n_options = len([v for v in self.values.values() 
                            if not _is_missing_label(v)])
        
        return {
            'var_code': self.var_code,
            'section': self.section,
            'question': self.question,
            'topic_tag': self.topic_tag,
            'response_format': self.response_format,
            'n_options': n_options,
            'is_concept': self.is_concept,
            'concept_id': self.concept_id,
        }


def _is_missing_label(label: str) -> bool:
    """Check if a label represents missing/artifact value."""
    missing_patterns = [
        'missing', 'refused', "don't know", 'no answer', 'not asked',
        'not applicable', 'nan', 'n/a'
    ]
    label_lower = str(label).lower()
    return any(pattern in label_lower for pattern in missing_patterns)


# =============================================================================
# Section Statistics
# =============================================================================

@dataclass
class SectionStats:
    """Statistics for a single survey section."""
    section: str
    total_questions: int
    questions_with_values: int  # Questions that have defined response options
    topic_tags: Dict[str, int]  # tag -> count
    response_formats: Dict[str, int]  # format -> count
    
    @property
    def usable_questions(self) -> int:
        """Questions that can be used as targets."""
        return self.questions_with_values


def compute_section_stats(
    metadata: Dict[str, Dict[str, Any]],
    exclude_sections: Optional[List[str]] = None,
    exclude_vars: Optional[Set[str]] = None
) -> Dict[str, SectionStats]:
    """
    Compute statistics for each section in survey metadata.
    
    Parameters
    ----------
    metadata : dict
        Survey metadata: section -> var_code -> {question, values, topic_tag, ...}
    exclude_sections : list[str], optional
        Sections to exclude (e.g., ['demographics'])
    exclude_vars : set[str], optional
        Individual variable codes to exclude (e.g., country-specific variants)
        
    Returns
    -------
    dict[str, SectionStats]
        Section name -> statistics
    """
    exclude_sections = set(s.lower() for s in (exclude_sections or []))
    exclude_vars = exclude_vars or set()
    stats = {}
    
    for section, questions in metadata.items():
        if section.lower() in exclude_sections:
            continue
        if not isinstance(questions, dict):
            continue
            
        topic_counts = defaultdict(int)
        format_counts = defaultdict(int)
        n_with_values = 0
        
        for var_code, var_info in questions.items():
            if not isinstance(var_info, dict):
                continue
            
            # Skip excluded variables (e.g., country-specific variants)
            if var_code in exclude_vars:
                continue
                
            values = var_info.get('values', {})
            # Only count questions with valid fixed options (not numeric/open-ended)
            if isinstance(values, dict) and values and has_valid_fixed_options(values):
                n_with_values += 1
                
                # Track topic tag
                tag = var_info.get('topic_tag') or var_info.get('harmonized_tag')
                if tag:
                    topic_counts[tag] += 1
                else:
                    topic_counts['_untagged'] += 1
                
                # Track response format
                fmt = detect_response_format(values)
                format_counts[fmt] += 1
        
        stats[section] = SectionStats(
            section=section,
            total_questions=len(questions),
            questions_with_values=n_with_values,
            topic_tags=dict(topic_counts),
            response_formats=dict(format_counts)
        )
    
    return stats


def print_section_stats(stats: Dict[str, SectionStats]) -> None:
    """Pretty-print section statistics."""
    print("\nSection Statistics:")
    print("=" * 70)
    
    total_usable = 0
    for section, s in sorted(stats.items()):
        print(f"\n{section}:")
        print(f"  Total questions: {s.total_questions}")
        print(f"  Usable (with values): {s.questions_with_values}")
        total_usable += s.questions_with_values
        
        if s.response_formats:
            formats_str = ', '.join(f"{k}={v}" for k, v in sorted(s.response_formats.items()))
            print(f"  Response formats: {formats_str}")
        
        if s.topic_tags and len(s.topic_tags) > 1:
            print(f"  Topic tags: {len(s.topic_tags)} unique")
            for tag, count in sorted(s.topic_tags.items(), key=lambda x: -x[1])[:5]:
                print(f"    - {tag}: {count}")
    
    print(f"\n{'=' * 70}")
    print(f"Total usable questions: {total_usable}")


# =============================================================================
# Adaptive Target Allocation
# =============================================================================

def compute_target_allocation(
    section_stats: Dict[str, SectionStats],
    n_targets: int = 40,
    min_features_per_section: int = 3,
    min_targets_per_section: int = 0,
    max_targets_per_section: Optional[int] = None
) -> Dict[str, int]:
    """
    Compute how many targets to sample from each section.
    
    Allocation is proportional to section size, with constraints:
    - Preserve at least min_features_per_section for profile generation
    - At least min_targets_per_section if section is large enough
    - At most max_targets_per_section
    
    Parameters
    ----------
    section_stats : dict
        Section statistics from compute_section_stats
    n_targets : int
        Total number of targets to sample
    min_features_per_section : int
        Minimum features to preserve per section for profiles
    min_targets_per_section : int
        Minimum targets from each section (if feasible)
    max_targets_per_section : int, optional
        Maximum targets from any single section
        
    Returns
    -------
    dict[str, int]
        Section -> number of targets to sample
    """
    # Compute available targets per section (preserving min features)
    available = {}
    for section, s in section_stats.items():
        max_possible = max(0, s.usable_questions - min_features_per_section)
        available[section] = max_possible
    
    total_available = sum(available.values())
    
    if total_available < n_targets:
        print(f"Warning: Only {total_available} targets available "
              f"(requested {n_targets}). Adjusting.")
        n_targets = total_available
    
    if total_available == 0:
        return {section: 0 for section in section_stats}
    
    # Initial proportional allocation
    allocation = {}
    for section, avail in available.items():
        if avail == 0:
            allocation[section] = 0
        else:
            # Proportional to usable questions
            weight = section_stats[section].usable_questions / sum(
                s.usable_questions for s in section_stats.values() if s.usable_questions > 0
            )
            raw_alloc = weight * n_targets
            allocation[section] = int(raw_alloc)
    
    # Apply constraints
    for section in allocation:
        if max_targets_per_section:
            allocation[section] = min(allocation[section], max_targets_per_section)
        allocation[section] = min(allocation[section], available[section])
        if available[section] >= min_targets_per_section:
            allocation[section] = max(allocation[section], min_targets_per_section)
    
    # Distribute remaining targets
    allocated = sum(allocation.values())
    remaining = n_targets - allocated
    
    if remaining > 0:
        # Sort by available - allocated (most room first)
        sections_by_room = sorted(
            allocation.keys(),
            key=lambda s: available[s] - allocation[s],
            reverse=True
        )
        
        for section in sections_by_room:
            if remaining <= 0:
                break
            room = available[section] - allocation[section]
            if max_targets_per_section:
                room = min(room, max_targets_per_section - allocation[section])
            add = min(remaining, room)
            allocation[section] += add
            remaining -= add
    
    return allocation


# =============================================================================
# Target Sampling
# =============================================================================

def sample_targets_stratified(
    metadata: Dict[str, Dict[str, Any]],
    n_targets: int = 40,
    min_features_per_section: int = 3,
    exclude_sections: Optional[List[str]] = None,
    seed: Optional[int] = None,
    diversity_weight: float = 0.3,
    verbose: bool = True,
    country_codes: Optional[List[str]] = None,
    concept_configs: Optional[List[Dict]] = None,
) -> Tuple[List[SampledTarget], Dict[str, Any]]:
    """
    Sample target questions with stratified allocation across sections.
    
    This is the main function for target sampling. It:
    1. Computes section sizes (excluding specified sections)
    2. Handles country-specific variables for ESS surveys (groups into concepts)
    3. Allocates targets proportionally while preserving feature pool
    4. Samples within each section, preferring response format diversity
    5. Returns targets with full metadata for analysis
    
    Parameters
    ----------
    metadata : dict
        Survey metadata: section -> var_code -> {question, values, topic_tag, ...}
    n_targets : int
        Total number of targets to sample
    min_features_per_section : int
        Minimum features to preserve per section
    exclude_sections : list[str], optional
        Sections to exclude from target pool (default: ['demographics', 'EXCLUDED'])
    seed : int, optional
        Random seed for reproducibility
    diversity_weight : float
        Weight for response format diversity in sampling (0-1).
        0 = pure random, 1 = maximize diversity
    verbose : bool
        Print sampling details
    country_codes : list[str], optional
        Country codes for ESS concept detection (e.g., ['AT', 'BE', 'DE']).
        If provided, enables country-specific variable grouping.
    concept_configs : list[dict], optional
        Custom concept configurations. If None, uses ESS_CONCEPT_CONFIGS.
        
    Returns
    -------
    tuple[list[SampledTarget], dict]
        - List of sampled targets (may include concept targets for ESS)
        - Sampling metadata (allocation, stats, concept info, etc.)
    """
    if seed is not None:
        random.seed(seed)
    
    # Default: exclude demographics and any explicit EXCLUDED section
    if exclude_sections is None:
        exclude_sections = ['demographics', 'EXCLUDED', 'excluded']
    
    # Detect country-specific variables if country codes provided
    concept_info = None
    vars_to_exclude = set()
    concept_representatives = []
    
    if country_codes:
        concept_info = detect_country_specific_variables(
            metadata,
            country_codes,
            concept_configs
        )
        vars_to_exclude = concept_info['vars_to_exclude']
        concept_representatives = concept_info['concept_representatives']
        
        if verbose and vars_to_exclude:
            n_concepts = len(concept_info['concepts'])
            print(f"\nCountry-specific variable handling:")
            print(f"  Detected {n_concepts} concepts covering {len(vars_to_exclude)} individual variables")
            for concept_id, data in concept_info['concepts'].items():
                print(f"    - {concept_id}: {data['n_countries']} countries, {data['n_total_vars']} vars")
    
    # Compute section statistics (excluding country-specific individual vars)
    section_stats = compute_section_stats(
        metadata, 
        exclude_sections,
        exclude_vars=vars_to_exclude
    )
    
    if verbose:
        print_section_stats(section_stats)
    
    # Compute target allocation
    allocation = compute_target_allocation(
        section_stats,
        n_targets=n_targets,
        min_features_per_section=min_features_per_section
    )
    
    if verbose:
        print(f"\nTarget Allocation:")
        for section, n in sorted(allocation.items()):
            available = max(0, section_stats[section].usable_questions - min_features_per_section)
            print(f"  {section}: {n} targets (from {available} available)")
    
    # Sample from each section
    sampled_targets = []
    
    for section, n_targets in allocation.items():
        if n_targets == 0:
            continue
            
        section_questions = metadata.get(section, {})
        
        # Build candidate list (excluding country-specific individual vars)
        candidates = []
        for var_code, var_info in section_questions.items():
            if not isinstance(var_info, dict):
                continue
            
            # Skip excluded variables
            if var_code in vars_to_exclude:
                continue
                
            values = var_info.get('values', {})
            if not isinstance(values, dict) or not values:
                continue
            
            # Skip questions without valid fixed options (numeric/open-ended)
            if not has_valid_fixed_options(values):
                continue
            
            target = SampledTarget(
                var_code=var_code,
                section=section,
                question=var_info.get('question', var_info.get('description', var_code)),
                values=values,
                topic_tag=var_info.get('topic_tag') or var_info.get('harmonized_tag'),
                is_concept=False,
            )
            candidates.append(target)
        
        # Add concept representatives for this section
        for rep in concept_representatives:
            if rep['section'] == section:
                # Concept representatives should have valid options too
                rep_values = rep.get('values', {})
                if rep_values and not has_valid_fixed_options(rep_values):
                    continue  # Skip concepts without valid fixed options
                
                target = SampledTarget(
                    var_code=rep['var_code'],
                    section=section,
                    question=rep['question'],
                    values=rep_values,
                    topic_tag=rep.get('topic_tag'),
                    is_concept=True,
                    concept_id=rep['concept_id'],
                )
                candidates.append(target)
        
        # Sample with optional diversity weighting
        if diversity_weight > 0 and n_targets < len(candidates):
            section_sampled = _sample_with_diversity(
                candidates, n_targets, diversity_weight
            )
        else:
            section_sampled = random.sample(candidates, min(n_targets, len(candidates)))
        
        sampled_targets.extend(section_sampled)
    
    # Compile sampling metadata
    sampling_metadata = {
        'total_requested': n_targets,
        'total_sampled': len(sampled_targets),
        'allocation': allocation,
        'section_stats': {s: {'usable': stats.usable_questions}
                         for s, stats in section_stats.items()},
        'excluded_sections': exclude_sections,
        'min_features_preserved': min_features_per_section,
        'seed': seed,
        'response_format_distribution': _count_formats(sampled_targets),
        'topic_tag_distribution': _count_tags(sampled_targets),
        'n_concept_targets': sum(1 for t in sampled_targets if t.is_concept),
        'country_specific_handling': concept_info is not None,
    }
    
    if concept_info:
        sampling_metadata['concepts'] = {
            cid: {'n_countries': data['n_countries'], 'n_vars': data['n_total_vars']}
            for cid, data in concept_info['concepts'].items()
        }
    
    if verbose:
        n_concepts = sum(1 for t in sampled_targets if t.is_concept)
        print(f"\nSampled {len(sampled_targets)} targets:")
        print(f"  Response formats: {sampling_metadata['response_format_distribution']}")
        print(f"  Topic tags: {len(sampling_metadata['topic_tag_distribution'])} unique")
        if n_concepts > 0:
            print(f"  Concept targets: {n_concepts} (will resolve to country-specific vars)")
    
    return sampled_targets, sampling_metadata


def _sample_with_diversity(
    candidates: List[SampledTarget],
    n: int,
    diversity_weight: float
) -> List[SampledTarget]:
    """
    Sample n targets from candidates, preferring response format diversity.
    
    Uses a greedy algorithm that balances random selection with 
    format diversity.
    """
    if n >= len(candidates):
        return candidates
    
    sampled = []
    remaining = candidates.copy()
    format_counts = defaultdict(int)
    
    for _ in range(n):
        # Score each candidate
        scores = []
        for c in remaining:
            # Random component
            random_score = random.random()
            
            # Diversity component: prefer underrepresented formats
            fmt = c.response_format
            current_count = format_counts[fmt]
            max_count = max(format_counts.values()) if format_counts else 0
            diversity_score = 1.0 / (1 + current_count) if max_count > 0 else 1.0
            
            # Combined score
            combined = (1 - diversity_weight) * random_score + diversity_weight * diversity_score
            scores.append((combined, c))
        
        # Select best scoring candidate
        scores.sort(key=lambda x: -x[0])
        selected = scores[0][1]
        
        sampled.append(selected)
        remaining.remove(selected)
        format_counts[selected.response_format] += 1
    
    return sampled


def _count_formats(targets: List[SampledTarget]) -> Dict[str, int]:
    """Count response formats in sampled targets."""
    counts = defaultdict(int)
    for t in targets:
        counts[t.response_format] += 1
    return dict(counts)


def _count_tags(targets: List[SampledTarget]) -> Dict[str, int]:
    """Count topic tags in sampled targets."""
    counts = defaultdict(int)
    for t in targets:
        tag = t.topic_tag or '_untagged'
        counts[tag] += 1
    return dict(counts)


# =============================================================================
# Integration Helpers
# =============================================================================

def targets_to_codes(targets: List[SampledTarget]) -> List[str]:
    """Extract just the variable codes for use with RespondentProfileGenerator."""
    return [t.var_code for t in targets]


def targets_to_dataframe(targets: List[SampledTarget]):
    """Convert targets to pandas DataFrame for inspection/saving."""
    import pandas as pd
    return pd.DataFrame([t.to_dict() for t in targets])


def save_targets(
    targets: List[SampledTarget],
    sampling_metadata: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save sampled targets and metadata to JSON.
    
    Useful for reproducibility and documentation.
    """
    import json
    
    output = {
        'targets': [t.to_dict() for t in targets],
        'metadata': sampling_metadata
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_targets(filepath: str) -> Tuple[List[SampledTarget], Dict[str, Any]]:
    """Load previously sampled targets from JSON."""
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    targets = []
    for t_dict in data['targets']:
        # Need to reconstruct with values for response format detection
        # This is lossy - we only have n_options, not full values
        target = SampledTarget(
            var_code=t_dict['var_code'],
            section=t_dict['section'],
            question=t_dict['question'],
            values={},  # Lost in serialization
            topic_tag=t_dict.get('topic_tag'),
            response_format=t_dict.get('response_format', 'other')
        )
        targets.append(target)
    
    return targets, data['metadata']


# =============================================================================
# Main for Testing
# =============================================================================

if __name__ == '__main__':
    import json
    import sys
    
    # Test with a metadata file if provided
    if len(sys.argv) > 1:
        metadata_path = sys.argv[1]
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        targets, meta = sample_targets_stratified(
            metadata,
            n_targets=40,
            min_features_per_section=3,
            seed=42,
            verbose=True
        )
        
        print(f"\n{'='*70}")
        print("Sampled Targets:")
        for t in targets[:10]:
            print(f"  {t.var_code} ({t.section}) - {t.response_format}")
            print(f"    Q: {t.question[:60]}...")
        if len(targets) > 10:
            print(f"  ... and {len(targets) - 10} more")