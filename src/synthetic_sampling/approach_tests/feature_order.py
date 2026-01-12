"""
Feature Order Robustness Test Generator (Test 3).

This module generates test instances for validating that perplexity-based
predictions are robust to the ordering of features in respondent profiles.

The test checks whether models perform genuine conditional reasoning versus
pattern-matching on feature order. A well-calibrated model should give
identical predictions regardless of how profile features are arranged.

Design rationale:
- Same features, different orderings → should produce same prediction
- Multiple random orderings per instance enable consistency measurement
- Original order preserved as baseline for comparison

Usage:
    from synthetic_sampling.approach_tests import FeatureOrderTestGenerator
    
    # Load base instances from DatasetBuilder output
    with open('base_instances.jsonl') as f:
        base_instances = [json.loads(line) for line in f]
    
    # Generate feature order test instances
    generator = FeatureOrderTestGenerator(seed=42)
    test_instances = generator.generate(base_instances, n_samples=500)
    
    # Save for inference
    generator.save_jsonl(test_instances, 'feature_order_test.jsonl')
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from copy import deepcopy


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FeatureOrderTestInstance:
    """
    A single test instance for feature order robustness evaluation.
    
    This extends a base prediction instance with multiple orderings of the
    same profile features, enabling assessment of prediction consistency.
    
    Key design: `orderings` contains different arrangements of the same
    features. Each ordering is a list of (question, answer) tuples that
    can be formatted into a profile prompt.
    
    Attributes:
        example_id: Unique identifier (prefixed with 'fo_')
        base_id: Original base_id from the prediction instance
        survey: Survey identifier (e.g., 'wvs', 'ess')
        respondent_id: Survey respondent identifier
        country: Country code or name
        
        profile_type: Profile format code (e.g., 's2m2')
        n_features: Number of features in the profile
        
        target_question: The question being predicted
        target_code: Variable code for the target question
        
        ground_truth: The respondent's actual answer
        ground_truth_index: Index of ground truth in options
        options: List of answer options
        
        orderings: Dict mapping ordering name to list of (question, answer) tuples
            - "original": features in original order
            - "random_1": first random permutation
            - "random_2": second random permutation
            - "random_3": third random permutation (if n_features >= 5)
            - "reversed": features in reversed order
        
        ordering_seeds: Dict mapping ordering name to random seed used (for reproducibility)
        
        eligible: Whether this instance is eligible for testing
        ineligibility_reason: If not eligible, why
    """
    # Identifiers
    example_id: str
    base_id: str
    survey: str
    respondent_id: str
    country: str
    
    # Profile metadata
    profile_type: str
    n_features: int
    
    # Target
    target_question: str
    target_code: str
    
    # Ground truth and options
    ground_truth: str
    ground_truth_index: int
    options: List[str]
    
    # Orderings - THE KEY STRUCTURE
    # Each ordering is a list of (question, answer) tuples
    orderings: Dict[str, List[Tuple[str, str]]]
    
    # Seeds for reproducibility
    ordering_seeds: Dict[str, Optional[int]]
    
    # Eligibility
    eligible: bool = True
    ineligibility_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Convert tuples to lists for JSON compatibility
        d = asdict(self)
        d['orderings'] = {
            k: [list(pair) for pair in v] 
            for k, v in self.orderings.items()
        }
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureOrderTestInstance':
        """Create instance from dictionary."""
        # Convert lists back to tuples
        if 'orderings' in data:
            data['orderings'] = {
                k: [tuple(pair) for pair in v]
                for k, v in data['orderings'].items()
            }
        return cls(**data)
    
    def get_ordering_names(self) -> List[str]:
        """Get list of ordering names."""
        return list(self.orderings.keys())
    
    def get_n_orderings(self) -> int:
        """Get number of orderings."""
        return len(self.orderings)
    
    def get_ordering_as_dict(self, ordering_name: str) -> Dict[str, str]:
        """Get a specific ordering as a question->answer dict."""
        if ordering_name not in self.orderings:
            raise KeyError(f"Unknown ordering: {ordering_name}")
        return {q: a for q, a in self.orderings[ordering_name]}
    
    def format_ordering(
        self, 
        ordering_name: str, 
        template: str = "Q: {question}\nA: {answer}"
    ) -> str:
        """
        Format an ordering as a string using a template.
        
        Args:
            ordering_name: Name of the ordering to format
            template: Template string with {question} and {answer} placeholders
            
        Returns:
            Formatted profile string
        """
        if ordering_name not in self.orderings:
            raise KeyError(f"Unknown ordering: {ordering_name}")
        
        lines = []
        for question, answer in self.orderings[ordering_name]:
            lines.append(template.format(question=question, answer=answer))
        return "\n".join(lines)


# =============================================================================
# TEST GENERATOR
# =============================================================================

class FeatureOrderTestGenerator:
    """
    Generates test instances for feature order robustness evaluation.
    
    Takes base prediction instances and produces test instances with
    multiple orderings of the same profile features.
    
    The generator:
    1. Filters instances to those eligible for testing (sufficient features)
    2. Creates multiple random orderings of profile features
    3. Tracks statistics for analysis
    
    Example:
        generator = FeatureOrderTestGenerator(seed=42, n_random_orderings=3)
        
        # From JSONL file
        test_instances = generator.generate_from_jsonl(
            'base_instances.jsonl',
            n_samples=500
        )
        
        # Save
        generator.save_jsonl(test_instances, 'feature_order_test.jsonl')
    """
    
    # Minimum features required for meaningful order testing
    MIN_FEATURES = 3
    
    def __init__(
        self,
        seed: int = 42,
        n_random_orderings: int = 2,
        include_reversed: bool = True,
        min_features_for_extra_ordering: int = 5,
    ):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
            n_random_orderings: Number of random orderings to generate (2-3 recommended)
            include_reversed: Whether to include reversed order as a condition
            min_features_for_extra_ordering: Add extra random ordering if profile has >= this many features
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_random_orderings = n_random_orderings
        self.include_reversed = include_reversed
        self.min_features_for_extra_ordering = min_features_for_extra_ordering
        
        # Statistics tracking
        self._reset_stats()
    
    def generate(
        self,
        base_instances: List[Dict[str, Any]],
        n_samples: Optional[int] = None,
        stratify_by_n_features: bool = True,
    ) -> List[FeatureOrderTestInstance]:
        """
        Generate feature order test instances from base prediction instances.
        
        Args:
            base_instances: List of base prediction instance dicts
            n_samples: Number of samples to generate (None = all eligible)
            stratify_by_n_features: If True, ensure representation across feature counts
            
        Returns:
            List of FeatureOrderTestInstance objects
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
            if stratify_by_n_features:
                eligible = self._stratified_sample(eligible, n_samples)
            else:
                self.rng.shuffle(eligible)
                eligible = eligible[:n_samples]
        
        return eligible
    
    def generate_from_jsonl(
        self,
        jsonl_path: Path | str,
        n_samples: Optional[int] = None,
        stratify_by_n_features: bool = True,
    ) -> List[FeatureOrderTestInstance]:
        """
        Generate test instances from a JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file with base instances
            n_samples: Number of samples to generate
            stratify_by_n_features: If True, ensure feature count representation
            
        Returns:
            List of FeatureOrderTestInstance objects
        """
        base_instances = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    base_instances.append(json.loads(line))
        
        return self.generate(
            base_instances,
            n_samples=n_samples,
            stratify_by_n_features=stratify_by_n_features,
        )
    
    def _process_instance(
        self,
        base_instance: Dict[str, Any]
    ) -> Optional[FeatureOrderTestInstance]:
        """
        Process a single base instance into a test instance.
        
        Returns None if the instance is missing required fields.
        Returns an ineligible instance if it fails eligibility checks.
        """
        self._stats['total_processed'] += 1
        
        # Check required fields
        required = ['survey', 'target_question', 'options', 'answer', 'questions']
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
        questions = base_instance['questions']  # Dict[str, str]
        target_question = base_instance['target_question']
        target_code = base_instance.get('target_code', base_instance.get('target_question_id', ''))
        options = base_instance['options']
        ground_truth = str(base_instance['answer'])
        
        # Generate IDs if not provided
        if not example_id:
            example_id = f"{survey}_{respondent_id}_{target_code}_{profile_type}"
        if not base_id:
            base_id = f"{survey}_{respondent_id}_{target_code}"
        
        # Helper to create ineligible instance
        def make_ineligible(reason: str) -> FeatureOrderTestInstance:
            self._stats['ineligible'] += 1
            self._stats['ineligibility_reasons'][reason] = \
                self._stats['ineligibility_reasons'].get(reason, 0) + 1
            return FeatureOrderTestInstance(
                example_id=f"fo_{example_id}",
                base_id=base_id,
                survey=survey,
                respondent_id=respondent_id,
                country=country,
                profile_type=profile_type,
                n_features=0,
                target_question=target_question,
                target_code=target_code,
                ground_truth=ground_truth,
                ground_truth_index=-1,
                options=[],
                orderings={},
                ordering_seeds={},
                eligible=False,
                ineligibility_reason=reason,
            )
        
        # Validate questions dict
        if not isinstance(questions, dict):
            return make_ineligible('questions_not_dict')
        
        n_features = len(questions)
        
        # Check minimum features
        if n_features < self.MIN_FEATURES:
            return make_ineligible(f'insufficient_features_{n_features}')
        
        # Validate options
        if not isinstance(options, list) or len(options) < 2:
            return make_ineligible('insufficient_options')
        
        # Convert options to strings
        options_str = [str(opt) for opt in options]
        
        # Find ground truth index
        ground_truth_index = -1
        for i, opt in enumerate(options_str):
            if opt == ground_truth:
                ground_truth_index = i
                break
        
        if ground_truth_index == -1:
            return make_ineligible('ground_truth_not_in_options')
        
        # Create original ordering as list of tuples
        original_ordering = list(questions.items())
        
        # Build orderings dict
        orderings: Dict[str, List[Tuple[str, str]]] = {
            "original": original_ordering,
        }
        ordering_seeds: Dict[str, Optional[int]] = {
            "original": None,
        }
        
        # Add reversed ordering
        if self.include_reversed:
            orderings["reversed"] = list(reversed(original_ordering))
            ordering_seeds["reversed"] = None
        
        # Add random orderings
        n_random = self.n_random_orderings
        if n_features >= self.min_features_for_extra_ordering:
            n_random += 1  # Extra ordering for larger profiles
        
        for i in range(n_random):
            ordering_name = f"random_{i + 1}"
            # Generate a deterministic seed for this ordering
            seed_for_ordering = self.seed + hash((example_id, i)) % (2**31)
            
            # Shuffle with this seed
            rng_local = random.Random(seed_for_ordering)
            shuffled = original_ordering.copy()
            rng_local.shuffle(shuffled)
            
            # Ensure it's actually different from original and reversed
            # (for small feature sets, shuffling might produce same order)
            attempts = 0
            while attempts < 10:
                if shuffled != original_ordering and shuffled != list(reversed(original_ordering)):
                    break
                seed_for_ordering += 1
                rng_local = random.Random(seed_for_ordering)
                shuffled = original_ordering.copy()
                rng_local.shuffle(shuffled)
                attempts += 1
            
            orderings[ordering_name] = shuffled
            ordering_seeds[ordering_name] = seed_for_ordering
        
        # Update stats
        self._stats['eligible'] += 1
        self._stats['by_n_features'][n_features] = self._stats['by_n_features'].get(n_features, 0) + 1
        self._stats['total_orderings'] += len(orderings)
        
        return FeatureOrderTestInstance(
            example_id=f"fo_{example_id}",
            base_id=base_id,
            survey=survey,
            respondent_id=respondent_id,
            country=country,
            profile_type=profile_type,
            n_features=n_features,
            target_question=target_question,
            target_code=target_code,
            ground_truth=ground_truth,
            ground_truth_index=ground_truth_index,
            options=options_str,
            orderings=orderings,
            ordering_seeds=ordering_seeds,
            eligible=True,
            ineligibility_reason=None,
        )
    
    def _stratified_sample(
        self,
        instances: List[FeatureOrderTestInstance],
        n_samples: int,
    ) -> List[FeatureOrderTestInstance]:
        """
        Sample instances with stratification by number of features.
        
        Ensures representation across different profile sizes.
        """
        # Group by n_features
        by_n_features: Dict[int, List[FeatureOrderTestInstance]] = {}
        
        for inst in instances:
            n = inst.n_features
            if n not in by_n_features:
                by_n_features[n] = []
            by_n_features[n].append(inst)
        
        # Shuffle each group
        for group in by_n_features.values():
            self.rng.shuffle(group)
        
        # Calculate proportional allocation
        selected = []
        total = len(instances)
        
        if total == 0:
            return []
        
        # Allocate proportionally
        for n_features, group in sorted(by_n_features.items()):
            proportion = len(group) / total
            allocation = max(1, int(n_samples * proportion))
            selected.extend(group[:allocation])
        
        # Fill remaining slots if needed
        remaining = n_samples - len(selected)
        if remaining > 0:
            selected_ids = {id(inst) for inst in selected}
            pool = [i for i in instances if id(i) not in selected_ids]
            self.rng.shuffle(pool)
            selected.extend(pool[:remaining])
        
        # Trim if over
        return selected[:n_samples]
    
    def _reset_stats(self) -> None:
        """Reset statistics tracking."""
        self._stats = {
            'total_processed': 0,
            'eligible': 0,
            'ineligible': 0,
            'ineligibility_reasons': {},
            'by_n_features': {},
            'total_orderings': 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics from the last generate() call."""
        return deepcopy(self._stats)
    
    def print_stats(self) -> None:
        """Print processing statistics."""
        stats = self._stats
        print("\n" + "=" * 60)
        print("Feature Order Test Generator Statistics")
        print("=" * 60)
        print(f"Total processed:   {stats['total_processed']}")
        print(f"Eligible:          {stats['eligible']}")
        print(f"Ineligible:        {stats['ineligible']}")
        print(f"Total orderings:   {stats['total_orderings']}")
        
        if stats['eligible'] > 0:
            avg_orderings = stats['total_orderings'] / stats['eligible']
            print(f"Avg orderings/instance: {avg_orderings:.1f}")
        
        if stats['ineligibility_reasons']:
            print("\nIneligibility reasons:")
            for reason, count in sorted(stats['ineligibility_reasons'].items()):
                print(f"  {reason}: {count}")
        
        print("\nBy number of features:")
        for n_features, count in sorted(stats['by_n_features'].items()):
            print(f"  {n_features} features: {count}")
    
    @staticmethod
    def save_jsonl(
        instances: List[FeatureOrderTestInstance],
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
    def load_jsonl(input_path: Path | str) -> List[FeatureOrderTestInstance]:
        """
        Load test instances from a JSONL file.
        
        Args:
            input_path: Path to JSONL file
            
        Returns:
            List of FeatureOrderTestInstance objects
        """
        instances = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instances.append(FeatureOrderTestInstance.from_dict(data))
        return instances


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_order_stats(instances: List[FeatureOrderTestInstance]) -> Dict[str, Any]:
    """
    Get aggregate statistics for a list of test instances.
    
    Args:
        instances: List of test instances
        
    Returns:
        Dict with counts and breakdowns
    """
    stats = {
        'total_instances': len(instances),
        'total_orderings': 0,
        'by_n_features': {},
        'by_n_orderings': {},
        'by_survey': {},
    }
    
    for inst in instances:
        # Total orderings
        n_orderings = inst.get_n_orderings()
        stats['total_orderings'] += n_orderings
        
        # By n_features
        n = inst.n_features
        stats['by_n_features'][n] = stats['by_n_features'].get(n, 0) + 1
        
        # By n_orderings
        stats['by_n_orderings'][n_orderings] = stats['by_n_orderings'].get(n_orderings, 0) + 1
        
        # By survey
        if inst.survey not in stats['by_survey']:
            stats['by_survey'][inst.survey] = 0
        stats['by_survey'][inst.survey] += 1
    
    return stats


def print_feature_order_stats(instances: List[FeatureOrderTestInstance]) -> None:
    """Print formatted statistics for test instances."""
    stats = get_feature_order_stats(instances)
    
    print("\n" + "=" * 60)
    print("Feature Order Test Instance Statistics")
    print("=" * 60)
    print(f"Total instances:    {stats['total_instances']}")
    print(f"Total orderings:    {stats['total_orderings']}")
    
    if stats['total_instances'] > 0:
        avg = stats['total_orderings'] / stats['total_instances']
        print(f"Avg orderings/inst: {avg:.1f}")
    
    print("\nBy number of features:")
    for n_features, count in sorted(stats['by_n_features'].items()):
        pct = 100 * count / stats['total_instances'] if stats['total_instances'] > 0 else 0
        print(f"  {n_features} features: {count} ({pct:.1f}%)")
    
    print("\nBy number of orderings:")
    for n_orderings, count in sorted(stats['by_n_orderings'].items()):
        pct = 100 * count / stats['total_instances'] if stats['total_instances'] > 0 else 0
        print(f"  {n_orderings} orderings: {count} ({pct:.1f}%)")
    
    print("\nBy survey:")
    for survey, count in sorted(stats['by_survey'].items()):
        print(f"  {survey}: {count}")


def compute_ordering_consistency(
    predictions: Dict[str, str],
    orderings: List[str],
) -> Dict[str, Any]:
    """
    Compute consistency metrics across orderings for a single instance.
    
    Args:
        predictions: Dict mapping ordering name to predicted answer
        orderings: List of ordering names to compare
        
    Returns:
        Dict with consistency metrics:
            - all_agree: bool, whether all orderings gave same prediction
            - n_unique: int, number of unique predictions
            - majority_prediction: str, most common prediction
            - majority_count: int, how many orderings gave majority prediction
    """
    if not orderings:
        return {
            'all_agree': True,
            'n_unique': 0,
            'majority_prediction': None,
            'majority_count': 0,
        }
    
    preds = [predictions.get(o) for o in orderings if o in predictions]
    
    if not preds:
        return {
            'all_agree': True,
            'n_unique': 0,
            'majority_prediction': None,
            'majority_count': 0,
        }
    
    unique_preds = set(preds)
    
    # Count occurrences
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    
    majority_pred = max(counts.keys(), key=lambda x: counts[x])
    
    return {
        'all_agree': len(unique_preds) == 1,
        'n_unique': len(unique_preds),
        'majority_prediction': majority_pred,
        'majority_count': counts[majority_pred],
    }