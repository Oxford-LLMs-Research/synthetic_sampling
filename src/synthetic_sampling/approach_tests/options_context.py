"""
Options Context Test Generator (Test 2).

This module generates test instances for validating whether showing answer 
options in the prompt affects perplexity-based predictions.

The test compares perplexity scores under different prompt conditions:
- Hidden: Options NOT shown in prompt (our primary approach)
- Shown (natural order): Options shown in their original survey order
- Shown (reversed): Options shown in reversed order (for scale questions only)

Design rationale:
- Natural order preserves ecological validity (how humans see surveys)
- Reversed order tests ordering sensitivity for scale-type questions
- Categorical questions (no inherent order) only get hidden vs. shown_natural

Usage:
    from synthetic_sampling.approach_tests import OptionsContextTestGenerator
    
    # Load base instances from DatasetBuilder output
    with open('base_instances.jsonl') as f:
        base_instances = [json.loads(line) for line in f]
    
    # Generate options context test instances
    generator = OptionsContextTestGenerator(seed=42)
    test_instances = generator.generate(base_instances, n_samples=500)
    
    # Save for inference
    generator.save_jsonl(test_instances, 'options_context_test.jsonl')
"""

import json
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from copy import deepcopy

from .variations import SPECIAL_VALUES


# =============================================================================
# OPTION TYPE DETECTION
# =============================================================================

# Patterns indicating a Likert-type scale (has inherent order)
SCALE_PATTERNS = [
    # Agreement scales
    r'(?i)^strongly\s+(agree|disagree)',
    r'(?i)^(agree|disagree)\s+strongly',
    r'(?i)^(completely|totally)\s+(agree|disagree)',
    r'(?i)^(somewhat|slightly)\s+(agree|disagree)',
    r'(?i)^neither\s+agree\s+nor\s+disagree',
    r'(?i)^(agree|disagree)$',
    
    # Frequency scales
    r'(?i)^(very|quite|fairly|rather)\s+(often|frequently|rarely|seldom)',
    r'(?i)^(always|usually|sometimes|rarely|never)$',
    r'(?i)^(every\s+day|several\s+times|once\s+a\s+week)',
    
    # Intensity/amount scales  
    r'(?i)^(very|quite|fairly|somewhat|a\s+little|not\s+at\s+all)',
    r'(?i)^(a\s+great\s+deal|a\s+lot|some|a\s+little|none)',
    r'(?i)^(extremely|very|moderately|slightly|not\s+at\s+all)',
    
    # Satisfaction scales
    r'(?i)^(very|fairly|not\s+very|not\s+at\s+all)\s+(satisfied|dissatisfied)',
    r'(?i)^(completely|mostly|somewhat)\s+(satisfied|dissatisfied)',
    
    # Importance scales
    r'(?i)^(very|fairly|not\s+very|not\s+at\s+all)\s+important',
    r'(?i)^(essential|very\s+important|fairly\s+important|not\s+important)',
    
    # Trust/confidence scales
    r'(?i)^(a\s+great\s+deal|quite\s+a\s+lot|not\s+very\s+much|none\s+at\s+all)',
    r'(?i)^(complete|a\s+lot\s+of|some|little|no)\s+(trust|confidence)',
    
    # Likelihood scales
    r'(?i)^(very|fairly|not\s+very|not\s+at\s+all)\s+likely',
    r'(?i)^(definitely|probably|probably\s+not|definitely\s+not)',
    
    # Quality/extent scales
    r'(?i)^(excellent|good|fair|poor|very\s+poor)$',
    r'(?i)^(much\s+better|somewhat\s+better|about\s+the\s+same|somewhat\s+worse|much\s+worse)',
    
    # Numeric-style ordinal
    r'(?i)^(first|second|third|fourth|fifth)\s+',
    r'(?i)^(1st|2nd|3rd|4th|5th)\s+',
]

# Compile patterns for efficiency
_SCALE_PATTERN_COMPILED = [re.compile(p) for p in SCALE_PATTERNS]


def is_scale_option(option: str) -> bool:
    """Check if an option looks like part of an ordered scale."""
    option_clean = option.strip()
    return any(p.search(option_clean) for p in _SCALE_PATTERN_COMPILED)


def detect_option_type(options: List[str]) -> str:
    """
    Detect whether options form an ordered scale or unordered categories.
    
    Args:
        options: List of answer option strings
        
    Returns:
        "scale" if options appear to be an ordered Likert-type scale
        "categorical" if options appear to be unordered categories
    """
    if len(options) < 2:
        return "categorical"
    
    # Count how many options match scale patterns
    scale_matches = sum(1 for opt in options if is_scale_option(opt))
    
    # If majority of options match scale patterns, treat as scale
    # Threshold: at least 50% of options should look like scale points
    if scale_matches >= len(options) / 2:
        return "scale"
    
    return "categorical"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptionsContextTestInstance:
    """
    A single test instance for options context sensitivity evaluation.
    
    This extends a base prediction instance with different prompt conditions
    for evaluating whether showing options affects perplexity predictions.
    
    Key design: `conditions` maps condition names to option orderings.
    - "hidden": None (no options shown)
    - "shown_natural": original order
    - "shown_reversed": reversed order (scale questions only)
    
    The `ground_truth_positions` dict tracks where the correct answer appears
    in each condition's ordering, enabling position bias analysis.
    
    Attributes:
        example_id: Unique identifier (prefixed with 'oc_')
        base_id: Original base_id from the prediction instance
        survey: Survey identifier (e.g., 'wvs', 'ess')
        respondent_id: Survey respondent identifier
        country: Country code or name
        
        profile_type: Profile format code (e.g., 's2m2')
        questions: Dict of profile questions and answers
        target_question: The question being predicted
        target_code: Variable code for the target question
        
        ground_truth: The respondent's actual answer
        ground_truth_index: Index of ground truth in natural order
        
        options: The answer options in natural (original) order
        option_type: "scale" or "categorical"
        n_options: Number of options
        
        conditions: Dict mapping condition name to option ordering
            - "hidden": None
            - "shown_natural": List[str] in natural order
            - "shown_reversed": List[str] in reversed order (if scale)
        
        ground_truth_positions: Dict mapping condition -> position of ground truth
            - "hidden": None (N/A)
            - "shown_natural": int
            - "shown_reversed": int (if scale)
        
        eligible: Whether this instance is eligible for testing
        ineligibility_reason: If not eligible, why
    """
    # Identifiers
    example_id: str
    base_id: str
    survey: str
    respondent_id: str
    country: str
    
    # Profile data
    profile_type: str
    questions: Dict[str, str]
    
    # Target
    target_question: str
    target_code: str
    
    # Ground truth
    ground_truth: str
    ground_truth_index: int  # Index in natural order
    
    # Options
    options: List[str]  # Natural order (substantive only, DK/Refused filtered)
    option_type: str    # "scale" or "categorical"
    n_options: int
    
    # Conditions - THE KEY STRUCTURE
    conditions: Dict[str, Optional[List[str]]]
    # {
    #   "hidden": None,
    #   "shown_natural": ["Strongly agree", "Agree", ...],
    #   "shown_reversed": ["Strongly disagree", ..., "Strongly agree"],  # if scale
    # }
    
    # Position tracking for bias analysis
    ground_truth_positions: Dict[str, Optional[int]]
    # {
    #   "hidden": None,
    #   "shown_natural": 1,
    #   "shown_reversed": 3,  # if scale
    # }
    
    # Eligibility
    eligible: bool = True
    ineligibility_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionsContextTestInstance':
        """Create instance from dictionary."""
        return cls(**data)
    
    def get_condition_names(self) -> List[str]:
        """Get list of active condition names (excluding None-valued)."""
        return [k for k, v in self.conditions.items() if k == "hidden" or v is not None]
    
    def is_scale(self) -> bool:
        """Check if this instance has scale-type options."""
        return self.option_type == "scale"
    
    def has_reversed_condition(self) -> bool:
        """Check if this instance has the reversed condition."""
        return "shown_reversed" in self.conditions and self.conditions["shown_reversed"] is not None


# =============================================================================
# TEST GENERATOR
# =============================================================================

class OptionsContextTestGenerator:
    """
    Generates test instances for options context sensitivity evaluation.
    
    Takes base prediction instances and produces test instances with
    multiple prompt conditions (hidden, shown_natural, shown_reversed).
    
    The generator:
    1. Filters instances to those eligible for testing
    2. Detects option type (scale vs categorical)
    3. Creates condition orderings
    4. Tracks statistics for analysis
    
    Example:
        generator = OptionsContextTestGenerator(seed=42)
        
        # From JSONL file
        test_instances = generator.generate_from_jsonl(
            'base_instances.jsonl',
            n_samples=500
        )
        
        # Save
        generator.save_jsonl(test_instances, 'options_context_test.jsonl')
    """
    
    def __init__(
        self,
        seed: int = 42,
        include_reversed: bool = True,
    ):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
            include_reversed: Whether to include reversed condition for scale questions
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.include_reversed = include_reversed
        
        # Statistics tracking
        self._stats = {
            'total_processed': 0,
            'eligible': 0,
            'ineligible': 0,
            'ineligibility_reasons': {},
            'option_types': {'scale': 0, 'categorical': 0},
            'by_n_options': {},
        }
    
    def generate(
        self,
        base_instances: List[Dict[str, Any]],
        n_samples: Optional[int] = None,
        stratify_by_type: bool = True,
    ) -> List[OptionsContextTestInstance]:
        """
        Generate options context test instances from base prediction instances.
        
        Args:
            base_instances: List of base prediction instance dicts
            n_samples: Number of samples to generate (None = all eligible)
            stratify_by_type: If True, ensure representation of both option types
            
        Returns:
            List of OptionsContextTestInstance objects
        """
        # Reset stats
        self._reset_stats()
        
        # Process all instances
        processed = []
        for inst in base_instances:
            test_inst = self._process_instance(inst)
            if test_inst is not None:
                processed.append(test_inst)
        
        # Filter to eligible instances
        eligible = [inst for inst in processed if inst.eligible]
        
        # Sample if needed
        if n_samples is not None and n_samples < len(eligible):
            if stratify_by_type:
                eligible = self._stratified_sample(eligible, n_samples)
            else:
                self.rng.shuffle(eligible)
                eligible = eligible[:n_samples]
        
        return eligible
    
    def generate_from_jsonl(
        self,
        jsonl_path: Path | str,
        n_samples: Optional[int] = None,
        stratify_by_type: bool = True,
    ) -> List[OptionsContextTestInstance]:
        """
        Generate test instances from a JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file with base instances
            n_samples: Number of samples to generate
            stratify_by_type: If True, ensure option type representation
            
        Returns:
            List of OptionsContextTestInstance objects
        """
        base_instances = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    base_instances.append(json.loads(line))
        
        return self.generate(
            base_instances,
            n_samples=n_samples,
            stratify_by_type=stratify_by_type,
        )
    
    def _process_instance(
        self,
        base_instance: Dict[str, Any]
    ) -> Optional[OptionsContextTestInstance]:
        """
        Process a single base instance into a test instance.
        
        Returns None if the instance is missing required fields.
        Returns an ineligible instance if it fails eligibility checks.
        """
        self._stats['total_processed'] += 1
        
        # Check required fields
        required = ['survey', 'target_question', 'options', 'answer']
        missing = [f for f in required if f not in base_instance]
        if missing:
            return None
        
        # Extract fields with fallbacks
        example_id = base_instance.get('example_id', '')
        base_id = base_instance.get('base_id', '')
        survey = base_instance['survey']
        respondent_id = str(base_instance.get('id', base_instance.get('respondent_id', '')))
        country = str(base_instance.get('country', ''))
        profile_type = base_instance.get('profile_type', '')
        questions = base_instance.get('questions', {})
        target_question = base_instance['target_question']
        target_code = base_instance.get('target_code', base_instance.get('target_question_id', ''))
        original_options_raw = base_instance['options']
        ground_truth = str(base_instance['answer'])
        
        # Generate IDs if not provided
        if not example_id:
            example_id = f"{survey}_{respondent_id}_{target_code}_{profile_type}"
        if not base_id:
            base_id = f"{survey}_{respondent_id}_{target_code}"
        
        # Helper to create ineligible instance
        def make_ineligible(reason: str) -> OptionsContextTestInstance:
            self._stats['ineligible'] += 1
            self._stats['ineligibility_reasons'][reason] = \
                self._stats['ineligibility_reasons'].get(reason, 0) + 1
            return OptionsContextTestInstance(
                example_id=f"oc_{example_id}",
                base_id=base_id,
                survey=survey,
                respondent_id=respondent_id,
                country=country,
                profile_type=profile_type,
                questions=questions,
                target_question=target_question,
                target_code=target_code,
                ground_truth=ground_truth,
                ground_truth_index=-1,
                options=[],
                option_type="unknown",
                n_options=0,
                conditions={},
                ground_truth_positions={},
                eligible=False,
                ineligibility_reason=reason,
            )
        
        # Validate options
        if not isinstance(original_options_raw, list) or len(original_options_raw) < 2:
            return make_ineligible('insufficient_options')
        
        # Filter substantive options (remove DK, Refused, etc.)
        substantive_options = self._filter_substantive_options(original_options_raw)
        
        if len(substantive_options) < 2:
            return make_ineligible('all_special_values')
        
        # Detect option type
        option_type = detect_option_type(substantive_options)
        
        # Find ground truth index
        ground_truth_index = -1
        for i, opt in enumerate(substantive_options):
            if opt == ground_truth:
                ground_truth_index = i
                break
        
        # Check if ground truth is in options (might have been filtered)
        if ground_truth_index == -1:
            # Ground truth might be a special value that was filtered
            return make_ineligible('ground_truth_filtered')
        
        # Build conditions
        conditions: Dict[str, Optional[List[str]]] = {
            "hidden": None,  # No options shown
            "shown_natural": substantive_options.copy(),
        }
        
        ground_truth_positions: Dict[str, Optional[int]] = {
            "hidden": None,
            "shown_natural": ground_truth_index,
        }
        
        # Add reversed condition for scale questions
        if self.include_reversed and option_type == "scale":
            reversed_options = list(reversed(substantive_options))
            conditions["shown_reversed"] = reversed_options
            
            # Find ground truth position in reversed order
            reversed_gt_index = len(substantive_options) - 1 - ground_truth_index
            ground_truth_positions["shown_reversed"] = reversed_gt_index
        
        # Update stats
        self._stats['eligible'] += 1
        self._stats['option_types'][option_type] += 1
        n_opts = len(substantive_options)
        self._stats['by_n_options'][n_opts] = self._stats['by_n_options'].get(n_opts, 0) + 1
        
        return OptionsContextTestInstance(
            example_id=f"oc_{example_id}",
            base_id=base_id,
            survey=survey,
            respondent_id=respondent_id,
            country=country,
            profile_type=profile_type,
            questions=questions,
            target_question=target_question,
            target_code=target_code,
            ground_truth=ground_truth,
            ground_truth_index=ground_truth_index,
            options=substantive_options,
            option_type=option_type,
            n_options=len(substantive_options),
            conditions=conditions,
            ground_truth_positions=ground_truth_positions,
            eligible=True,
            ineligibility_reason=None,
        )
    
    def _filter_substantive_options(self, options: List[Any]) -> List[str]:
        """
        Filter out special values (DK, Refused, etc.) from options.
        
        Args:
            options: Raw list of answer options
            
        Returns:
            List of substantive (non-special) options as strings
        """
        substantive = []
        for opt in options:
            opt_str = str(opt).strip()
            opt_lower = opt_str.lower()
            
            # Check against special values
            is_special = False
            for sv in SPECIAL_VALUES:
                if sv.lower() in opt_lower or opt_lower in sv.lower():
                    is_special = True
                    break
            
            if not is_special and opt_str:
                substantive.append(opt_str)
        
        return substantive
    
    def _stratified_sample(
        self,
        instances: List[OptionsContextTestInstance],
        n_samples: int,
    ) -> List[OptionsContextTestInstance]:
        """
        Sample instances with stratification by option type.
        
        Ensures balanced representation of scale vs categorical questions.
        """
        # Group by option type
        by_type = {'scale': [], 'categorical': []}
        
        for inst in instances:
            if inst.option_type in by_type:
                by_type[inst.option_type].append(inst)
        
        # Shuffle each group
        for group in by_type.values():
            self.rng.shuffle(group)
        
        # Allocate samples proportionally with minimum representation
        selected = []
        
        # Calculate proportional allocation
        total = len(instances)
        if total == 0:
            return []
        
        # Ensure at least 30% representation for minority type
        min_per_type = max(50, n_samples // 3)
        
        for opt_type in ['scale', 'categorical']:
            available = by_type[opt_type]
            to_take = min(min_per_type, len(available))
            selected.extend(available[:to_take])
        
        # Fill remaining slots proportionally
        remaining = n_samples - len(selected)
        if remaining > 0:
            selected_ids = {id(inst) for inst in selected}
            pool = [i for i in instances if id(i) not in selected_ids]
            self.rng.shuffle(pool)
            selected.extend(pool[:remaining])
        
        return selected[:n_samples]
    
    def _reset_stats(self) -> None:
        """Reset statistics tracking."""
        self._stats = {
            'total_processed': 0,
            'eligible': 0,
            'ineligible': 0,
            'ineligibility_reasons': {},
            'option_types': {'scale': 0, 'categorical': 0},
            'by_n_options': {},
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics from the last generate() call."""
        return deepcopy(self._stats)
    
    def print_stats(self) -> None:
        """Print processing statistics."""
        stats = self._stats
        print("\n" + "=" * 60)
        print("Options Context Test Generator Statistics")
        print("=" * 60)
        print(f"Total processed:  {stats['total_processed']}")
        print(f"Eligible:         {stats['eligible']}")
        print(f"Ineligible:       {stats['ineligible']}")
        
        if stats['ineligibility_reasons']:
            print("\nIneligibility reasons:")
            for reason, count in sorted(stats['ineligibility_reasons'].items()):
                print(f"  {reason}: {count}")
        
        print("\nOption types:")
        for opt_type, count in stats['option_types'].items():
            pct = 100 * count / stats['eligible'] if stats['eligible'] > 0 else 0
            print(f"  {opt_type}: {count} ({pct:.1f}%)")
        
        print("\nBy number of options:")
        for n_opts, count in sorted(stats['by_n_options'].items()):
            print(f"  {n_opts} options: {count}")
    
    @staticmethod
    def save_jsonl(
        instances: List[OptionsContextTestInstance],
        output_path: Path | str,
    ) -> None:
        """
        Save test instances to a JSONL file.
        
        Args:
            instances: List of test instances to save
            output_path: Path for output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for inst in instances:
                f.write(json.dumps(inst.to_dict(), ensure_ascii=False) + '\n')
    
    @staticmethod
    def load_jsonl(input_path: Path | str) -> List[OptionsContextTestInstance]:
        """
        Load test instances from a JSONL file.
        
        Args:
            input_path: Path to JSONL file
            
        Returns:
            List of OptionsContextTestInstance objects
        """
        instances = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instances.append(OptionsContextTestInstance.from_dict(data))
        return instances


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_options_context_stats(instances: List[OptionsContextTestInstance]) -> Dict[str, Any]:
    """
    Get aggregate statistics for a list of test instances.
    
    Args:
        instances: List of test instances
        
    Returns:
        Dict with counts and breakdowns
    """
    stats = {
        'total_instances': len(instances),
        'by_option_type': {'scale': 0, 'categorical': 0},
        'with_reversed_condition': 0,
        'by_n_options': {},
        'by_survey': {},
        'ground_truth_positions': {
            'shown_natural': {},
            'shown_reversed': {},
        },
    }
    
    for inst in instances:
        # By option type
        if inst.option_type in stats['by_option_type']:
            stats['by_option_type'][inst.option_type] += 1
        
        # Has reversed condition
        if inst.has_reversed_condition():
            stats['with_reversed_condition'] += 1
        
        # By number of options
        n = inst.n_options
        stats['by_n_options'][n] = stats['by_n_options'].get(n, 0) + 1
        
        # By survey
        if inst.survey not in stats['by_survey']:
            stats['by_survey'][inst.survey] = 0
        stats['by_survey'][inst.survey] += 1
        
        # Ground truth positions (for position bias analysis)
        for cond in ['shown_natural', 'shown_reversed']:
            pos = inst.ground_truth_positions.get(cond)
            if pos is not None:
                if pos not in stats['ground_truth_positions'][cond]:
                    stats['ground_truth_positions'][cond][pos] = 0
                stats['ground_truth_positions'][cond][pos] += 1
    
    return stats


def print_options_context_stats(instances: List[OptionsContextTestInstance]) -> None:
    """Print formatted statistics for test instances."""
    stats = get_options_context_stats(instances)
    
    print("\n" + "=" * 60)
    print("Options Context Test Instance Statistics")
    print("=" * 60)
    print(f"Total instances:           {stats['total_instances']}")
    print(f"With reversed condition:   {stats['with_reversed_condition']}")
    
    print("\nBy option type:")
    for opt_type, count in stats['by_option_type'].items():
        pct = 100 * count / stats['total_instances'] if stats['total_instances'] > 0 else 0
        print(f"  {opt_type}: {count} ({pct:.1f}%)")
    
    print("\nBy number of options:")
    for n_opts, count in sorted(stats['by_n_options'].items()):
        print(f"  {n_opts} options: {count}")
    
    print("\nBy survey:")
    for survey, count in sorted(stats['by_survey'].items()):
        print(f"  {survey}: {count}")
    
    # Position distribution (useful for bias analysis)
    print("\nGround truth position distribution (shown_natural):")
    pos_natural = stats['ground_truth_positions']['shown_natural']
    for pos in sorted(pos_natural.keys()):
        count = pos_natural[pos]
        pct = 100 * count / stats['total_instances'] if stats['total_instances'] > 0 else 0
        print(f"  Position {pos}: {count} ({pct:.1f}%)")