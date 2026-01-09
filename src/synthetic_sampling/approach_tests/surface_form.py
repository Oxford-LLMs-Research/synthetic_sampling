"""
Surface Form Sensitivity Test Generator (Test 1).

This module generates test instances for validating that perplexity-based 
evaluation is robust to surface form variations in answer options.

The test checks whether semantically equivalent answer phrasings produce
consistent predictions. A well-calibrated model should give similar perplexity
to "Strongly agree" and "Completely agree" (synonyms), "Agree" and "I agree"
(pronoun variants), and "Strongly agree" and "Agree strongly" (reorderings).

Usage:
    from synthetic_sampling.approach_tests import SurfaceFormTestGenerator
    
    # Load base instances from DatasetBuilder output
    with open('base_instances.jsonl') as f:
        base_instances = [json.loads(line) for line in f]
    
    # Generate surface form test instances
    generator = SurfaceFormTestGenerator(seed=42)
    test_instances = generator.generate(base_instances, n_samples=500)
    
    # Save for inference
    generator.save_jsonl(test_instances, 'surface_form_test.jsonl')
"""

import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
from copy import deepcopy

from .variations import (
    AnswerVariation,
    AnswerVariationGenerator,
    SPECIAL_VALUES,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SurfaceFormTestInstance:
    """
    A single test instance for surface form sensitivity evaluation.
    
    This extends a base prediction instance with variation metadata,
    allowing comparison of perplexity across semantically equivalent
    answer phrasings.
    
    Key design: `option_sets` contains parallel lists where index alignment
    enables direct comparison. For each position i:
        option_sets["original"][i] is the original option
        option_sets["synonym"][i] is its synonym variant (or null)
        option_sets["reorder"][i] is its reordered variant (or null)
        option_sets["pronoun"][i] is its pronoun variant (or null)
    
    This makes analysis trivial: compare perplexity at same index across sets.
    
    Attributes:
        example_id: Unique identifier (prefixed with 'sf_')
        base_id: Original base_id from the prediction instance
        survey: Survey identifier (e.g., 'wvs', 'ess')
        respondent_id: Survey respondent identifier  
        country: Country code or name
        
        profile_type: Profile format code (e.g., 's2m2')
        questions: Dict of profile questions and answers
        target_question: The question being predicted
        target_code: Variable code for the target question
        
        ground_truth: The respondent's actual answer
        ground_truth_index: Index of ground truth in option_sets lists (-1 if filtered)
        
        option_sets: Dict of parallel option lists
            - "original": substantive options (DK/Refused filtered out)
            - "synonym": synonym variations (null where no variation exists)
            - "reorder": reordered variations (null where none)
            - "pronoun": pronoun variations (null where none)
        
        variation_rules: Dict mapping (variation_type, index) -> rule applied
        has_variations: Dict of bool per variation type
        
        eligible: Whether this instance is eligible for surface form testing
        ineligibility_reason: If not eligible, why
    """
    # Identifiers (matching base instance format)
    example_id: str
    base_id: str
    survey: str
    respondent_id: str
    country: str
    
    # Profile data
    profile_type: str
    questions: Dict[str, str]  # The profile Q&A pairs
    
    # Target
    target_question: str
    target_code: str
    
    # Ground truth
    ground_truth: str
    ground_truth_index: int  # Index in option_sets lists (-1 if filtered out)
    
    # Parallel option sets - THE KEY STRUCTURE
    # Each list has same length; null where no variation exists
    option_sets: Dict[str, List[Optional[str]]]
    
    # Metadata about variations
    variation_rules: Dict[str, Optional[str]]  # "synonym_0" -> "Strongly -> Completely"
    has_variations: Dict[str, bool]  # "synonym" -> True/False
    
    # Eligibility
    eligible: bool = True
    ineligibility_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SurfaceFormTestInstance':
        """Create instance from dictionary."""
        return cls(**data)
    
    def has_variation_type(self, variation_type: str) -> bool:
        """Check if this instance has any variations of the given type."""
        return self.has_variations.get(variation_type, False)
    
    def get_n_options(self) -> int:
        """Get number of substantive options."""
        return len(self.option_sets.get("original", []))
    
    def get_variations_at_index(self, idx: int) -> Dict[str, Optional[str]]:
        """Get all option variants at a given index."""
        return {
            var_type: options[idx] if idx < len(options) else None
            for var_type, options in self.option_sets.items()
        }
    
    def count_variations(self) -> Dict[str, int]:
        """Count non-null variations per type."""
        counts = {}
        for var_type, options in self.option_sets.items():
            counts[var_type] = sum(1 for opt in options if opt is not None)
        return counts


# =============================================================================
# TEST GENERATOR
# =============================================================================

class SurfaceFormTestGenerator:
    """
    Generates test instances for surface form sensitivity evaluation.
    
    Takes base prediction instances (from DatasetBuilder or JSONL) and produces
    test instances with answer variations attached.
    
    The generator:
    1. Filters instances to those eligible for surface form testing
    2. Generates synonym, pronoun, and reorder variations for answer options
    3. Tracks coverage statistics for stratified sampling
    4. Outputs instances ready for inference pipeline
    
    Example:
        generator = SurfaceFormTestGenerator(seed=42)
        
        # From JSONL file
        test_instances = generator.generate_from_jsonl(
            'base_instances.jsonl', 
            n_samples=500
        )
        
        # Or from list of dicts
        test_instances = generator.generate(base_instances, n_samples=500)
        
        # Save
        generator.save_jsonl(test_instances, 'surface_form_test.jsonl')
    """
    
    # Required fields in base instances
    REQUIRED_FIELDS = [
        'respondent_id',
        'survey', 
        'target_question',
        'options',
        'answer',
    ]
    
    # Fields that map to our output (with possible aliases)
    FIELD_ALIASES = {
        'profile': ['profile', 'profile_text', 'features'],
        'target_question_id': ['target_question_id', 'target_code', 'target_var'],
        'country': ['country', 'country_code', 'country_name'],
        'instance_id': ['instance_id', 'example_id', 'id'],
    }
    
    def __init__(
        self,
        seed: int = 42,
        variation_generator: Optional[AnswerVariationGenerator] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
            variation_generator: Custom variation generator (uses default if None)
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.variation_generator = variation_generator or AnswerVariationGenerator()
        
        # Statistics tracking
        self._stats = {
            'total_processed': 0,
            'eligible': 0,
            'ineligible': 0,
            'ineligibility_reasons': {},
            'variation_coverage': {'synonym': 0, 'reorder': 0, 'pronoun': 0},
        }
    
    def generate(
        self,
        base_instances: List[Dict[str, Any]],
        n_samples: Optional[int] = None,
        require_variations: bool = True,
        stratify_by_variation: bool = True,
    ) -> List[SurfaceFormTestInstance]:
        """
        Generate surface form test instances from base prediction instances.
        
        Args:
            base_instances: List of base prediction instance dicts
            n_samples: Number of samples to generate (None = all eligible)
            require_variations: If True, only include instances with ≥1 variation
            stratify_by_variation: If True, ensure representation of all variation types
            
        Returns:
            List of SurfaceFormTestInstance objects
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
        
        # Optionally filter to those with variations
        if require_variations:
            eligible = [
                inst for inst in eligible 
                if any(inst.has_variations.get(vt, False) for vt in ['synonym', 'reorder', 'pronoun'])
            ]
        
        # Sample if needed
        if n_samples is not None and n_samples < len(eligible):
            if stratify_by_variation:
                eligible = self._stratified_sample(eligible, n_samples)
            else:
                self.rng.shuffle(eligible)
                eligible = eligible[:n_samples]
        
        return eligible
    
    def generate_from_jsonl(
        self,
        jsonl_path: Path | str,
        n_samples: Optional[int] = None,
        require_variations: bool = True,
        stratify_by_variation: bool = True,
    ) -> List[SurfaceFormTestInstance]:
        """
        Generate test instances from a JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file with base instances
            n_samples: Number of samples to generate
            require_variations: If True, only include instances with variations
            stratify_by_variation: If True, ensure variation type representation
            
        Returns:
            List of SurfaceFormTestInstance objects
        """
        base_instances = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    base_instances.append(json.loads(line))
        
        return self.generate(
            base_instances,
            n_samples=n_samples,
            require_variations=require_variations,
            stratify_by_variation=stratify_by_variation,
        )
    
    def _process_instance(
        self, 
        base_instance: Dict[str, Any]
    ) -> Optional[SurfaceFormTestInstance]:
        """
        Process a single base instance into a test instance.
        
        Expected base instance format (from DatasetBuilder):
        {
            "example_id": "latinobarometer_340_794_S2_s2m2",
            "base_id": "latinobarometer_340_794_S2",
            "survey": "latinobarometer",
            "id": "340_794",  # respondent_id
            "country": 340.0,
            "questions": {"Q1": "A1", "Q2": "A2", ...},  # profile
            "target_question": "...",
            "target_code": "S2",
            "answer": "Lower class",
            "options": ["Upper class", ..., "Do not know"],
            "profile_type": "s2m2"
        }
        
        Returns None if the instance is missing required fields.
        Returns an ineligible instance if it fails eligibility checks.
        """
        self._stats['total_processed'] += 1
        
        # Check required fields
        required = ['survey', 'target_question', 'options', 'answer']
        missing = [f for f in required if f not in base_instance]
        if missing:
            return None
        
        # Extract fields with fallbacks for different naming conventions
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
        
        # Generate example_id if not provided
        if not example_id:
            example_id = f"{survey}_{respondent_id}_{target_code}_{profile_type}"
        if not base_id:
            base_id = f"{survey}_{respondent_id}_{target_code}"
        
        # Helper to create ineligible instance
        def make_ineligible(reason: str) -> SurfaceFormTestInstance:
            self._stats['ineligible'] += 1
            self._stats['ineligibility_reasons'][reason] = \
                self._stats['ineligibility_reasons'].get(reason, 0) + 1
            return SurfaceFormTestInstance(
                example_id=f"sf_{example_id}",
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
                option_sets={},
                variation_rules={},
                has_variations={},
                eligible=False,
                ineligibility_reason=reason,
            )
        
        # Validate options
        if not isinstance(original_options_raw, list) or len(original_options_raw) < 2:
            return make_ineligible('insufficient_options')
        
        # Filter substantive options (remove DK, Refused, etc.)
        substantive_options = self.variation_generator.filter_substantive_options(original_options_raw)
        
        if len(substantive_options) < 2:
            return make_ineligible('all_special_values')
        
        # Check question eligibility (party voting, ethnicity, etc.)
        if self.variation_generator.is_question_ineligible(target_question):
            return make_ineligible('ineligible_question_pattern')
        
        # Check option eligibility (party names, ethnicities, etc.)
        for opt in substantive_options:
            if self.variation_generator.is_option_ineligible(opt):
                return make_ineligible('ineligible_option_content')
        
        # Generate variations for each option
        variations_raw = self.variation_generator.generate_all_variations(substantive_options)
        
        # Build parallel option_sets structure
        # Each list has same length as substantive_options
        n_options = len(substantive_options)
        option_sets = {
            "original": substantive_options.copy(),
            "synonym": [None] * n_options,
            "reorder": [None] * n_options,
            "pronoun": [None] * n_options,
        }
        
        # Track which rules were applied
        variation_rules = {}
        
        # Map variations back to their original option index
        # Create lookup from original text to index
        orig_to_idx = {opt: i for i, opt in enumerate(substantive_options)}
        
        # Fill in variation slots
        for var_type in ['synonym', 'reorder', 'pronoun']:
            for var in variations_raw.get(var_type, []):
                idx = orig_to_idx.get(var.original)
                if idx is not None:
                    # Only take first variation per slot (avoid duplicates)
                    if option_sets[var_type][idx] is None:
                        option_sets[var_type][idx] = var.variation
                        variation_rules[f"{var_type}_{idx}"] = var.rule_applied
        
        # Determine which variation types have any non-null values
        has_variations = {
            var_type: any(opt is not None for opt in options)
            for var_type, options in option_sets.items()
            if var_type != "original"
        }
        
        # Find ground truth index
        ground_truth_index = -1
        for i, opt in enumerate(substantive_options):
            if opt == ground_truth:
                ground_truth_index = i
                break
        
        # Update stats
        self._stats['eligible'] += 1
        for var_type, has_var in has_variations.items():
            if has_var:
                self._stats['variation_coverage'][var_type] += 1
        
        return SurfaceFormTestInstance(
            example_id=f"sf_{example_id}",
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
            option_sets=option_sets,
            variation_rules=variation_rules,
            has_variations=has_variations,
            eligible=True,
            ineligibility_reason=None,
        )
    
    def _stratified_sample(
        self,
        instances: List[SurfaceFormTestInstance],
        n_samples: int,
    ) -> List[SurfaceFormTestInstance]:
        """
        Sample instances with stratification by variation type.
        
        Ensures balanced representation across synonym, pronoun, and reorder
        variations, prioritizing instances with rarer variation types.
        """
        # Group by variation types present
        by_type = {
            'pronoun': [],
            'reorder': [],
            'synonym': [],
            'none': [],
        }
        
        for inst in instances:
            has_any = False
            for var_type in ['pronoun', 'reorder', 'synonym']:
                if inst.has_variation_type(var_type):
                    by_type[var_type].append(inst)
                    has_any = True
            if not has_any:
                by_type['none'].append(inst)
        
        # Shuffle each group
        for group in by_type.values():
            self.rng.shuffle(group)
        
        # Allocate samples proportionally but with minimum representation
        # Prioritize rarer types (pronoun > reorder > synonym)
        selected = set()
        result = []
        
        # First, ensure minimum representation for each type
        min_per_type = min(50, n_samples // 6)  # At least 50 or 1/6 of total
        
        for var_type in ['pronoun', 'reorder', 'synonym']:
            available = [i for i in by_type[var_type] if id(i) not in selected]
            to_take = min(min_per_type, len(available))
            for inst in available[:to_take]:
                selected.add(id(inst))
                result.append(inst)
        
        # Fill remaining slots from all eligible instances
        remaining = n_samples - len(result)
        if remaining > 0:
            pool = [i for i in instances if id(i) not in selected]
            self.rng.shuffle(pool)
            result.extend(pool[:remaining])
        
        return result
    
    def _reset_stats(self) -> None:
        """Reset statistics tracking."""
        self._stats = {
            'total_processed': 0,
            'eligible': 0,
            'ineligible': 0,
            'ineligibility_reasons': {},
            'variation_coverage': {'synonym': 0, 'reorder': 0, 'pronoun': 0},
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics from the last generate() call."""
        return deepcopy(self._stats)
    
    def print_stats(self) -> None:
        """Print processing statistics."""
        stats = self._stats
        print("\n" + "=" * 60)
        print("Surface Form Test Generator Statistics")
        print("=" * 60)
        print(f"Total processed:  {stats['total_processed']}")
        print(f"Eligible:         {stats['eligible']}")
        print(f"Ineligible:       {stats['ineligible']}")
        
        if stats['ineligibility_reasons']:
            print("\nIneligibility reasons:")
            for reason, count in sorted(stats['ineligibility_reasons'].items()):
                print(f"  {reason}: {count}")
        
        print("\nVariation coverage (instances with ≥1 variation):")
        for var_type, count in stats['variation_coverage'].items():
            pct = 100 * count / stats['eligible'] if stats['eligible'] > 0 else 0
            print(f"  {var_type}: {count} ({pct:.1f}%)")
    
    @staticmethod
    def save_jsonl(
        instances: List[SurfaceFormTestInstance],
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
    def load_jsonl(input_path: Path | str) -> List[SurfaceFormTestInstance]:
        """
        Load test instances from a JSONL file.
        
        Args:
            input_path: Path to JSONL file
            
        Returns:
            List of SurfaceFormTestInstance objects
        """
        instances = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instances.append(SurfaceFormTestInstance.from_dict(data))
        return instances


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_instance_stats(instances: List[SurfaceFormTestInstance]) -> Dict[str, Any]:
    """
    Get aggregate statistics for a list of test instances.
    
    Args:
        instances: List of test instances
        
    Returns:
        Dict with counts and breakdowns
    """
    stats = {
        'total_instances': len(instances),
        'total_option_slots': 0,  # Total original options across all instances
        'variation_slots_filled': {'synonym': 0, 'reorder': 0, 'pronoun': 0},
        'instances_with_variation': {'synonym': 0, 'reorder': 0, 'pronoun': 0},
        'by_survey': {},
    }
    
    for inst in instances:
        # Count option slots
        n_options = inst.get_n_options()
        stats['total_option_slots'] += n_options
        
        # Count filled variation slots
        for vt in ['synonym', 'reorder', 'pronoun']:
            options = inst.option_sets.get(vt, [])
            filled = sum(1 for opt in options if opt is not None)
            stats['variation_slots_filled'][vt] += filled
        
        # Count instances with each variation type
        for vt in ['synonym', 'reorder', 'pronoun']:
            if inst.has_variations.get(vt, False):
                stats['instances_with_variation'][vt] += 1
        
        # By survey
        if inst.survey not in stats['by_survey']:
            stats['by_survey'][inst.survey] = 0
        stats['by_survey'][inst.survey] += 1
    
    return stats


def print_instance_stats(instances: List[SurfaceFormTestInstance]) -> None:
    """Print formatted statistics for test instances."""
    stats = get_instance_stats(instances)
    
    print("\n" + "=" * 60)
    print("Surface Form Test Instance Statistics")
    print("=" * 60)
    print(f"Total instances:       {stats['total_instances']}")
    print(f"Total option slots:    {stats['total_option_slots']}")
    
    print("\nVariation slots filled (out of total option slots):")
    for vt, count in stats['variation_slots_filled'].items():
        pct = 100 * count / stats['total_option_slots'] if stats['total_option_slots'] > 0 else 0
        print(f"  {vt}: {count} ({pct:.1f}%)")
    
    print("\nInstances with ≥1 variation of type:")
    for vt, count in stats['instances_with_variation'].items():
        pct = 100 * count / stats['total_instances'] if stats['total_instances'] > 0 else 0
        print(f"  {vt}: {count} ({pct:.1f}%)")
    
    print("\nBy survey:")
    for survey, count in sorted(stats['by_survey'].items()):
        print(f"  {survey}: {count}")