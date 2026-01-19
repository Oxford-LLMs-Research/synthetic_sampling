"""
Validation Test Data Quality Diagnostics
=========================================

Comprehensive checks for:
1. Data integrity and completeness
2. Diversity and coverage
3. Test-specific quality issues
4. Potential bugs and edge cases

Author: Research Team
Date: January 2026
"""

import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import glob

# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(path):
    """Load JSONL file."""
    instances = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    return instances

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)

def print_subsection(title):
    print(f"\n{'-'*50}")
    print(f" {title}")
    print('-'*50)

def print_distribution(counter, title, top_n=10):
    """Print distribution from counter."""
    print(f"\n{title}:")
    total = sum(counter.values())
    for item, count in counter.most_common(top_n):
        pct = count / total * 100
        print(f"  {item}: {count} ({pct:.1f}%)")
    if len(counter) > top_n:
        print(f"  ... and {len(counter) - top_n} more")

def check_condition(condition, message, critical=False):
    """Check a condition and print result."""
    status = "✓" if condition else ("✗ CRITICAL" if critical else "⚠ WARNING")
    print(f"  {status}: {message}")
    return condition

# =============================================================================
# BASE INSTANCES DIAGNOSTICS
# =============================================================================

def diagnose_base_instances(instances):
    print_section("BASE INSTANCES DIAGNOSTICS")
    
    print(f"\nTotal instances: {len(instances)}")
    
    # Basic field checks
    print_subsection("Field Completeness")
    required_fields = ['example_id', 'base_id', 'survey', 'id', 'country', 
                       'questions', 'target_question', 'target_code', 'answer', 'options']
    
    for field in required_fields:
        present = sum(1 for inst in instances if field in inst and inst[field])
        check_condition(present == len(instances), 
                       f"'{field}' present in all instances ({present}/{len(instances)})",
                       critical=True)
    
    # Survey distribution
    print_subsection("Survey Distribution")
    surveys = Counter(inst['survey'] for inst in instances)
    print_distribution(surveys, "Surveys")
    check_condition(len(surveys) >= 4, f"At least 4 surveys represented ({len(surveys)} found)")
    
    # Country distribution
    print_subsection("Country Distribution")
    countries = Counter(inst['country'] for inst in instances)
    print_distribution(countries, "Countries", top_n=15)
    print(f"\n  Total unique countries: {len(countries)}")
    check_condition(len(countries) >= 15, f"At least 15 countries ({len(countries)} found)")
    
    # Target question distribution
    print_subsection("Target Question Distribution")
    target_codes = Counter(inst['target_code'] for inst in instances)
    print(f"\n  Unique target questions: {len(target_codes)}")
    respondents_per_q = [count for count in target_codes.values()]
    print(f"  Respondents per question: min={min(respondents_per_q)}, max={max(respondents_per_q)}, "
          f"mean={statistics.mean(respondents_per_q):.1f}, median={statistics.median(respondents_per_q):.0f}")
    
    # Questions per survey
    questions_by_survey = defaultdict(set)
    for inst in instances:
        questions_by_survey[inst['survey']].add(inst['target_code'])
    print("\n  Questions per survey:")
    for survey, codes in sorted(questions_by_survey.items()):
        print(f"    {survey}: {len(codes)} unique questions")
    
    # Profile statistics
    print_subsection("Profile Statistics")
    profile_sizes = [len(inst['questions']) for inst in instances]
    print(f"  Features per profile: min={min(profile_sizes)}, max={max(profile_sizes)}, "
          f"mean={statistics.mean(profile_sizes):.1f}")
    
    profile_types = Counter(inst.get('profile_type', 'unknown') for inst in instances)
    print_distribution(profile_types, "Profile types")
    
    # Answer options
    print_subsection("Answer Options")
    n_options = [len(inst['options']) for inst in instances]
    print(f"  Options per question: min={min(n_options)}, max={max(n_options)}, "
          f"mean={statistics.mean(n_options):.1f}")
    
    # Check ground truth in options
    gt_in_options = sum(1 for inst in instances if inst['answer'] in inst['options'])
    check_condition(gt_in_options == len(instances),
                   f"Ground truth in options for all instances ({gt_in_options}/{len(instances)})",
                   critical=True)
    
    # Check for duplicates
    print_subsection("Duplicate Checks")
    example_ids = [inst['example_id'] for inst in instances]
    check_condition(len(example_ids) == len(set(example_ids)),
                   f"No duplicate example_ids ({len(example_ids)} total, {len(set(example_ids))} unique)",
                   critical=True)
    
    base_ids = [inst['base_id'] for inst in instances]
    print(f"  Unique base_ids: {len(set(base_ids))} (total: {len(base_ids)})")
    
    return {
        'n_instances': len(instances),
        'n_surveys': len(surveys),
        'n_countries': len(countries),
        'n_questions': len(target_codes),
        'surveys': surveys,
        'countries': countries,
    }

# =============================================================================
# SURFACE FORM TEST DIAGNOSTICS
# =============================================================================

def diagnose_surface_form(instances):
    print_section("SURFACE FORM TEST DIAGNOSTICS")
    
    print(f"\nTotal instances: {len(instances)}")
    
    # Eligibility
    print_subsection("Eligibility")
    eligible = [inst for inst in instances if inst.get('eligible', True)]
    ineligible = [inst for inst in instances if not inst.get('eligible', True)]
    print(f"  Eligible: {len(eligible)} ({len(eligible)/len(instances)*100:.1f}%)")
    print(f"  Ineligible: {len(ineligible)} ({len(ineligible)/len(instances)*100:.1f}%)")
    
    if ineligible:
        reasons = Counter(inst.get('ineligibility_reason', 'unknown') for inst in ineligible)
        print_distribution(reasons, "Ineligibility reasons")
    
    # Variation coverage
    print_subsection("Variation Coverage")
    
    has_synonym = sum(1 for inst in instances if inst.get('has_variations', {}).get('synonym', False))
    has_reorder = sum(1 for inst in instances if inst.get('has_variations', {}).get('reorder', False))
    has_pronoun = sum(1 for inst in instances if inst.get('has_variations', {}).get('pronoun', False))
    has_any = sum(1 for inst in instances if any(inst.get('has_variations', {}).values()))
    
    print(f"  Instances with synonym variations: {has_synonym} ({has_synonym/len(instances)*100:.1f}%)")
    print(f"  Instances with reorder variations: {has_reorder} ({has_reorder/len(instances)*100:.1f}%)")
    print(f"  Instances with pronoun variations: {has_pronoun} ({has_pronoun/len(instances)*100:.1f}%)")
    print(f"  Instances with ANY variation: {has_any} ({has_any/len(instances)*100:.1f}%)")
    
    check_condition(has_any / len(instances) > 0.5, 
                   f"At least 50% of instances have variations ({has_any/len(instances)*100:.1f}%)")
    
    # Option-level variation coverage
    print_subsection("Option-Level Variation Coverage")
    
    total_options = 0
    synonym_options = 0
    reorder_options = 0
    pronoun_options = 0
    
    for inst in instances:
        option_sets = inst.get('option_sets', {})
        original = option_sets.get('original', [])
        synonym = option_sets.get('synonym', [])
        reorder = option_sets.get('reorder', [])
        pronoun = option_sets.get('pronoun', [])
        
        total_options += len(original)
        synonym_options += sum(1 for s in synonym if s is not None)
        reorder_options += sum(1 for r in reorder if r is not None)
        pronoun_options += sum(1 for p in pronoun if p is not None)
    
    print(f"  Total options across all instances: {total_options}")
    print(f"  Options with synonym: {synonym_options} ({synonym_options/total_options*100:.1f}%)")
    print(f"  Options with reorder: {reorder_options} ({reorder_options/total_options*100:.1f}%)")
    print(f"  Options with pronoun: {pronoun_options} ({pronoun_options/total_options*100:.1f}%)")
    
    # Check for problematic patterns
    print_subsection("Potential Issues")
    
    # All-null variations
    all_null_synonym = sum(1 for inst in instances 
                          if all(s is None for s in inst.get('option_sets', {}).get('synonym', [])))
    all_null_reorder = sum(1 for inst in instances 
                          if all(r is None for r in inst.get('option_sets', {}).get('reorder', [])))
    all_null_pronoun = sum(1 for inst in instances 
                          if all(p is None for p in inst.get('option_sets', {}).get('pronoun', [])))
    
    print(f"  Instances with ALL null synonyms: {all_null_synonym}")
    print(f"  Instances with ALL null reorders: {all_null_reorder}")
    print(f"  Instances with ALL null pronouns: {all_null_pronoun}")
    
    # Check option_sets alignment
    misaligned = 0
    for inst in instances:
        option_sets = inst.get('option_sets', {})
        original_len = len(option_sets.get('original', []))
        for var_type in ['synonym', 'reorder', 'pronoun']:
            var_len = len(option_sets.get(var_type, []))
            if var_len != original_len:
                misaligned += 1
                break
    
    check_condition(misaligned == 0,
                   f"Option sets aligned (same length) for all instances ({misaligned} misaligned)",
                   critical=True)
    
    # Diversity of scales/question types
    print_subsection("Scale Diversity")
    
    # Analyze original option patterns
    option_patterns = Counter()
    for inst in instances:
        options = inst.get('option_sets', {}).get('original', [])
        # Create pattern signature
        pattern = tuple(opt.lower() for opt in options[:4])  # First 4 options
        option_patterns[pattern] += 1
    
    print(f"  Unique option patterns (first 4 options): {len(option_patterns)}")
    print("\n  Most common patterns:")
    for pattern, count in option_patterns.most_common(5):
        pct = count / len(instances) * 100
        pattern_str = ' | '.join(pattern[:3]) + ('...' if len(pattern) > 3 else '')
        print(f"    [{count:3d}] ({pct:5.1f}%) {pattern_str}")
    
    # Check for dominating scale
    top_pattern_count = option_patterns.most_common(1)[0][1] if option_patterns else 0
    check_condition(top_pattern_count / len(instances) < 0.3,
                   f"No single scale dominates >30% ({top_pattern_count/len(instances)*100:.1f}%)")
    
    # Survey distribution
    print_subsection("Survey Distribution")
    surveys = Counter(inst['survey'] for inst in instances)
    print_distribution(surveys, "Surveys")
    
    return {
        'n_instances': len(instances),
        'n_eligible': len(eligible),
        'has_synonym': has_synonym,
        'has_reorder': has_reorder,
        'has_pronoun': has_pronoun,
        'synonym_option_coverage': synonym_options / total_options if total_options > 0 else 0,
        'n_option_patterns': len(option_patterns),
    }

# =============================================================================
# OPTIONS CONTEXT TEST DIAGNOSTICS
# =============================================================================

def diagnose_options_context(instances):
    print_section("OPTIONS CONTEXT TEST DIAGNOSTICS")
    
    print(f"\nTotal instances: {len(instances)}")
    
    # Eligibility
    print_subsection("Eligibility")
    eligible = [inst for inst in instances if inst.get('eligible', True)]
    print(f"  Eligible: {len(eligible)} ({len(eligible)/len(instances)*100:.1f}%)")
    
    # Option type distribution
    print_subsection("Option Type Distribution")
    option_types = Counter(inst.get('option_type', 'unknown') for inst in instances)
    print_distribution(option_types, "Option types")
    
    n_scale = option_types.get('scale', 0)
    n_categorical = option_types.get('categorical', 0)
    check_condition(n_scale > 0 and n_categorical > 0,
                   f"Both scale and categorical questions present (scale={n_scale}, categorical={n_categorical})")
    
    # Conditions check
    print_subsection("Conditions Verification")
    
    has_hidden = sum(1 for inst in instances if 'hidden' in inst.get('conditions', {}))
    has_shown_natural = sum(1 for inst in instances if 'shown_natural' in inst.get('conditions', {}))
    has_shown_reversed = sum(1 for inst in instances 
                            if inst.get('conditions', {}).get('shown_reversed') is not None)
    
    print(f"  Instances with 'hidden' condition: {has_hidden}")
    print(f"  Instances with 'shown_natural' condition: {has_shown_natural}")
    print(f"  Instances with 'shown_reversed' condition: {has_shown_reversed}")
    
    # Verify scale questions have reversed, categorical don't
    scale_with_reversed = sum(1 for inst in instances 
                              if inst.get('option_type') == 'scale' 
                              and inst.get('conditions', {}).get('shown_reversed') is not None)
    cat_without_reversed = sum(1 for inst in instances 
                               if inst.get('option_type') == 'categorical' 
                               and inst.get('conditions', {}).get('shown_reversed') is None)
    
    check_condition(scale_with_reversed == n_scale,
                   f"All scale questions have shown_reversed ({scale_with_reversed}/{n_scale})")
    check_condition(cat_without_reversed == n_categorical,
                   f"No categorical questions have shown_reversed ({cat_without_reversed}/{n_categorical})")
    
    # Ground truth positions
    print_subsection("Ground Truth Position Verification")
    
    position_errors = 0
    for inst in instances:
        options = inst.get('options', [])
        gt = inst.get('ground_truth')
        gt_idx = inst.get('ground_truth_index')
        
        # Check index matches
        if gt_idx is not None and gt_idx < len(options):
            if options[gt_idx] != gt:
                position_errors += 1
        
        # Check positions in conditions
        conditions = inst.get('conditions', {})
        positions = inst.get('ground_truth_positions', {})
        
        if conditions.get('shown_natural'):
            natural_pos = positions.get('shown_natural')
            if natural_pos is not None:
                if conditions['shown_natural'][natural_pos] != gt:
                    position_errors += 1
        
        if conditions.get('shown_reversed'):
            reversed_pos = positions.get('shown_reversed')
            if reversed_pos is not None:
                if conditions['shown_reversed'][reversed_pos] != gt:
                    position_errors += 1
    
    check_condition(position_errors == 0,
                   f"Ground truth positions correct ({position_errors} errors found)",
                   critical=True)
    
    # Number of options distribution
    print_subsection("Options Count Distribution")
    n_options = Counter(inst.get('n_options', len(inst.get('options', []))) for inst in instances)
    print_distribution(n_options, "Number of options")
    
    # Survey distribution
    print_subsection("Survey Distribution")
    surveys = Counter(inst['survey'] for inst in instances)
    print_distribution(surveys, "Surveys")
    
    return {
        'n_instances': len(instances),
        'n_scale': n_scale,
        'n_categorical': n_categorical,
        'position_errors': position_errors,
    }

# =============================================================================
# FEATURE ORDER TEST DIAGNOSTICS
# =============================================================================

def diagnose_feature_order(instances):
    print_section("FEATURE ORDER TEST DIAGNOSTICS")
    
    print(f"\nTotal instances: {len(instances)}")
    
    # Eligibility
    print_subsection("Eligibility")
    eligible = [inst for inst in instances if inst.get('eligible', True)]
    print(f"  Eligible: {len(eligible)} ({len(eligible)/len(instances)*100:.1f}%)")
    
    # Feature count distribution
    print_subsection("Feature Count Distribution")
    n_features = Counter(inst.get('n_features', 0) for inst in instances)
    print_distribution(n_features, "Number of features")
    
    feature_counts = [inst.get('n_features', 0) for inst in instances]
    print(f"\n  Min features: {min(feature_counts)}")
    print(f"  Max features: {max(feature_counts)}")
    print(f"  Mean features: {statistics.mean(feature_counts):.1f}")
    
    check_condition(min(feature_counts) >= 3,
                   f"All instances have at least 3 features (min={min(feature_counts)})")
    
    # Orderings check
    print_subsection("Orderings Verification")
    
    ordering_counts = Counter()
    for inst in instances:
        orderings = inst.get('orderings', {})
        ordering_counts[len(orderings)] += 1
    
    print_distribution(ordering_counts, "Number of orderings per instance")
    
    # Verify ordering contents
    print_subsection("Ordering Content Verification")
    
    ordering_errors = 0
    content_mismatch = 0
    
    for inst in instances:
        orderings = inst.get('orderings', {})
        n_feat = inst.get('n_features', 0)
        
        # Check all orderings have same length (number of questions)
        lengths = [len(ord_dict) for ord_dict in orderings.values() if isinstance(ord_dict, dict)]
        if len(set(lengths)) > 1:
            ordering_errors += 1
        
        # Check all orderings have same content (same questions and answers, just reordered)
        if 'original' in orderings and 'reversed' in orderings:
            original = orderings['original']
            reversed_ord = orderings['reversed']
            
            # Both should be dictionaries
            if isinstance(original, dict) and isinstance(reversed_ord, dict):
                # Check they have the same questions
                if set(original.keys()) != set(reversed_ord.keys()):
                    content_mismatch += 1
                # Check they have the same question->answer mappings
                elif original != reversed_ord:
                    # They should have same mappings, just different order
                    # (order is preserved in dict insertion order in Python 3.7+)
                    for q, a in original.items():
                        if reversed_ord.get(q) != a:
                            content_mismatch += 1
                            break
            else:
                content_mismatch += 1
    
    check_condition(ordering_errors == 0,
                   f"All orderings have consistent length ({ordering_errors} errors)")
    check_condition(content_mismatch == 0,
                   f"All orderings have same content ({content_mismatch} mismatches)",
                   critical=True)
    
    # Check reversed is actually reversed
    print_subsection("Reversal Verification")
    
    reversal_correct = 0
    reversal_wrong = 0
    
    for inst in instances:
        orderings = inst.get('orderings', {})
        if 'original' in orderings and 'reversed' in orderings:
            original = orderings['original']
            reversed_ord = orderings['reversed']
            
            if isinstance(original, dict) and isinstance(reversed_ord, dict):
                # Check if reversed is the reverse order of original
                # In Python 3.7+, dict preserves insertion order
                original_items = list(original.items())
                reversed_items = list(reversed_ord.items())
                
                if original_items == list(reversed(reversed_items)):
                    reversal_correct += 1
                else:
                    reversal_wrong += 1
            else:
                reversal_wrong += 1
    
    check_condition(reversal_wrong == 0,
                   f"Reversed orderings are correct reversals ({reversal_wrong} wrong)")
    
    # Random orderings uniqueness
    print_subsection("Random Orderings Uniqueness")
    
    duplicate_randoms = 0
    for inst in instances:
        orderings = inst.get('orderings', {})
        random_orderings = [v for k, v in orderings.items() if k.startswith('random_')]
        
        # Check if any random orderings are identical
        for i, ord1 in enumerate(random_orderings):
            for ord2 in random_orderings[i+1:]:
                if ord1 == ord2:
                    duplicate_randoms += 1
    
    check_condition(duplicate_randoms == 0,
                   f"No duplicate random orderings ({duplicate_randoms} duplicates)")
    
    # Survey distribution
    print_subsection("Survey Distribution")
    surveys = Counter(inst['survey'] for inst in instances)
    print_distribution(surveys, "Surveys")
    
    return {
        'n_instances': len(instances),
        'min_features': min(feature_counts),
        'max_features': max(feature_counts),
        'ordering_errors': ordering_errors,
        'content_mismatch': content_mismatch,
        'reversal_wrong': reversal_wrong,
    }

# =============================================================================
# CROSS-TEST CONSISTENCY
# =============================================================================

def diagnose_cross_test_consistency(base, surface, options, feature):
    print_section("CROSS-TEST CONSISTENCY")
    
    # Check base_id coverage
    print_subsection("Base ID Coverage")
    
    base_ids = set(inst['base_id'] for inst in base)
    surface_base_ids = set(inst['base_id'] for inst in surface)
    options_base_ids = set(inst['base_id'] for inst in options)
    feature_base_ids = set(inst['base_id'] for inst in feature)
    
    print(f"  Base instances: {len(base_ids)} unique base_ids")
    print(f"  Surface form: {len(surface_base_ids)} base_ids ({len(surface_base_ids & base_ids)} overlap with base)")
    print(f"  Options context: {len(options_base_ids)} base_ids ({len(options_base_ids & base_ids)} overlap with base)")
    print(f"  Feature order: {len(feature_base_ids)} base_ids ({len(feature_base_ids & base_ids)} overlap with base)")
    
    # Check if test files are subsets of base
    surface_missing = surface_base_ids - base_ids
    options_missing = options_base_ids - base_ids
    feature_missing = feature_base_ids - base_ids
    
    check_condition(len(surface_missing) == 0,
                   f"All surface form base_ids exist in base ({len(surface_missing)} missing)")
    check_condition(len(options_missing) == 0,
                   f"All options context base_ids exist in base ({len(options_missing)} missing)")
    check_condition(len(feature_missing) == 0,
                   f"All feature order base_ids exist in base ({len(feature_missing)} missing)")
    
    # Check for consistent data across tests
    print_subsection("Data Consistency Across Tests")
    
    # Build lookup by base_id
    base_lookup = {inst['base_id']: inst for inst in base}
    
    inconsistencies = 0
    checked = 0
    
    for inst in surface[:100]:  # Check first 100
        base_id = inst['base_id']
        if base_id in base_lookup:
            base_inst = base_lookup[base_id]
            checked += 1
            
            # Check target question matches
            if inst['target_question'] != base_inst['target_question']:
                inconsistencies += 1
            
            # Check ground truth matches
            if inst['ground_truth'] != base_inst['answer']:
                inconsistencies += 1
            
            # Check options match
            if inst['option_sets']['original'] != base_inst['options']:
                inconsistencies += 1
    
    check_condition(inconsistencies == 0,
                   f"Data consistent across base and surface form ({inconsistencies} inconsistencies in {checked} checked)")

# =============================================================================
# EDGE CASES AND ANOMALIES
# =============================================================================

def diagnose_edge_cases(surface, options, feature):
    print_section("EDGE CASES AND ANOMALIES")
    
    print_subsection("Surface Form Edge Cases")
    
    # Check for weird variation patterns
    weird_synonyms = []
    for inst in surface:
        option_sets = inst.get('option_sets', {})
        original = option_sets.get('original', [])
        synonym = option_sets.get('synonym', [])
        
        for i, (orig, syn) in enumerate(zip(original, synonym)):
            if syn is not None:
                # Check if synonym is suspiciously similar or different
                if orig.lower() == syn.lower():
                    weird_synonyms.append((inst['example_id'], orig, syn, "identical"))
                elif len(syn) > 3 * len(orig) or len(orig) > 3 * len(syn):
                    weird_synonyms.append((inst['example_id'], orig, syn, "length_mismatch"))
    
    if weird_synonyms:
        print(f"\n  Suspicious synonyms found: {len(weird_synonyms)}")
        for ex_id, orig, syn, issue in weird_synonyms[:5]:
            print(f"    [{issue}] '{orig}' -> '{syn}'")
    else:
        print("  No suspicious synonyms found ✓")
    
    print_subsection("Options Context Edge Cases")
    
    # Check for very long or very short option lists
    option_lengths = [inst.get('n_options', 0) for inst in options]
    very_short = sum(1 for n in option_lengths if n < 3)
    very_long = sum(1 for n in option_lengths if n > 10)
    
    print(f"  Very short option lists (<3 options): {very_short}")
    print(f"  Very long option lists (>10 options): {very_long}")
    
    # Check for unusual scale detection
    scale_with_few = sum(1 for inst in options 
                        if inst.get('option_type') == 'scale' and inst.get('n_options', 0) < 4)
    cat_with_many = sum(1 for inst in options 
                       if inst.get('option_type') == 'categorical' and inst.get('n_options', 0) > 7)
    
    print(f"  'scale' type with <4 options: {scale_with_few}")
    print(f"  'categorical' type with >7 options: {cat_with_many}")
    
    print_subsection("Feature Order Edge Cases")
    
    # Very few features
    few_features = sum(1 for inst in feature if inst.get('n_features', 0) <= 3)
    print(f"  Instances with ≤3 features: {few_features}")
    
    # Check for identical orderings (should not happen with random)
    identical_orderings = 0
    for inst in feature:
        orderings = inst.get('orderings', {})
        ordering_dicts = [v for v in orderings.values() if isinstance(v, dict)]
        for i, ord1 in enumerate(ordering_dicts):
            for ord2 in ordering_dicts[i+1:]:
                # Compare both keys and values (order matters for dicts in Python 3.7+)
                if ord1 == ord2:
                    identical_orderings += 1
    
    print(f"  Instances with identical orderings: {identical_orderings}")
    check_condition(identical_orderings == 0, "No identical orderings within instances")

# =============================================================================
# SAMPLE INSPECTION
# =============================================================================

def print_sample_instances(surface, options, feature, n=3):
    print_section("SAMPLE INSTANCES FOR MANUAL INSPECTION")
    
    print_subsection(f"Surface Form Test (first {n})")
    for inst in surface[:n]:
        print(f"\n  ID: {inst['example_id']}")
        print(f"  Survey: {inst['survey']}, Country: {inst['country']}")
        print(f"  Target: {inst['target_question'][:80]}...")
        print(f"  Ground truth: {inst['ground_truth']}")
        print(f"  Original options: {inst['option_sets']['original']}")
        print(f"  Synonym options:  {inst['option_sets']['synonym']}")
        print(f"  Has variations: {inst['has_variations']}")
    
    print_subsection(f"Options Context Test (first {n})")
    for inst in options[:n]:
        print(f"\n  ID: {inst['example_id']}")
        print(f"  Survey: {inst['survey']}, Option type: {inst['option_type']}")
        print(f"  Target: {inst['target_question'][:80]}...")
        print(f"  Ground truth: {inst['ground_truth']} (index: {inst['ground_truth_index']})")
        print(f"  Conditions: {list(inst['conditions'].keys())}")
        print(f"  GT positions: {inst['ground_truth_positions']}")
    
    print_subsection(f"Feature Order Test (first {n})")
    for inst in feature[:n]:
        print(f"\n  ID: {inst['example_id']}")
        print(f"  Survey: {inst['survey']}, Features: {inst['n_features']}")
        print(f"  Target: {inst['target_question'][:80]}...")
        print(f"  Orderings: {list(inst['orderings'].keys())}")
        if inst['orderings'].get('original'):
            original = inst['orderings']['original']
            if isinstance(original, dict):
                first_q = list(original.keys())[0]
                first_a = original[first_q]
                print(f"  First feature (original): {first_q[:60]}... -> {first_a}")
            else:
                print(f"  First feature (original): {original[0] if original else 'N/A'}")

# =============================================================================
# PATH RESOLUTION
# =============================================================================

def get_outputs_dir():
    """Get the outputs directory relative to the script location."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    outputs_dir = project_root / "outputs"
    return outputs_dir

def find_latest_base_instances(outputs_dir):
    """Find the most recent base_instances file."""
    pattern = str(outputs_dir / "base_instances_*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, most recent first
    files.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
    return files[0]

def resolve_path(path_str, outputs_dir):
    """Resolve a path string to an absolute Path.
    
    If path_str is None or empty, returns None.
    If path_str is relative, resolves relative to outputs_dir.
    If path_str is absolute, uses as-is.
    """
    if not path_str:
        return None
    
    path = Path(path_str)
    if path.is_absolute():
        return path
    else:
        return outputs_dir / path

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validation Test Data Quality Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (auto-detects latest base_instances file)
  python validation_data_diagnostics.py
  
  # Specify all paths explicitly
  python validation_data_diagnostics.py \\
      --base outputs/base_instances_wvs_ess_wave_10_ess_wave_11_plus4_20260113_164142.jsonl \\
      --surface outputs/surface_form_test.jsonl \\
      --options outputs/options_context_test.jsonl \\
      --feature outputs/feature_order_test.jsonl
  
  # Specify only base instances, others use defaults
  python validation_data_diagnostics.py --base outputs/my_base_instances.jsonl
        """
    )
    
    outputs_dir = get_outputs_dir()
    
    parser.add_argument(
        '--base',
        type=str,
        default=None,
        help=f'Path to base instances JSONL file (default: auto-detect latest in {outputs_dir})'
    )
    parser.add_argument(
        '--surface',
        type=str,
        default='surface_form_test.jsonl',
        help=f'Path to surface form test JSONL file (default: {outputs_dir}/surface_form_test.jsonl)'
    )
    parser.add_argument(
        '--options',
        type=str,
        default='options_context_test.jsonl',
        help=f'Path to options context test JSONL file (default: {outputs_dir}/options_context_test.jsonl)'
    )
    parser.add_argument(
        '--feature',
        type=str,
        default='feature_order_test.jsonl',
        help=f'Path to feature order test JSONL file (default: {outputs_dir}/feature_order_test.jsonl)'
    )
    parser.add_argument(
        '--outputs-dir',
        type=str,
        default=None,
        help=f'Base directory for output files (default: {outputs_dir})'
    )
    
    args = parser.parse_args()
    
    # Override outputs_dir if specified
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    
    # Resolve paths
    if args.base:
        base_path = resolve_path(args.base, outputs_dir)
    else:
        # Auto-detect latest base instances file
        base_path = find_latest_base_instances(outputs_dir)
        if base_path is None:
            print(f"ERROR: Could not find base_instances_*.jsonl file in {outputs_dir}")
            print("Please specify --base explicitly.")
            return
        base_path = Path(base_path)
    
    surface_path = resolve_path(args.surface, outputs_dir)
    options_path = resolve_path(args.options, outputs_dir)
    feature_path = resolve_path(args.feature, outputs_dir)
    
    # Validate all paths exist
    missing = []
    if not base_path.exists():
        missing.append(('base', base_path))
    if not surface_path.exists():
        missing.append(('surface', surface_path))
    if not options_path.exists():
        missing.append(('options', options_path))
    if not feature_path.exists():
        missing.append(('feature', feature_path))
    
    if missing:
        print("ERROR: The following files were not found:")
        for name, path in missing:
            print(f"  {name}: {path}")
        return
    
    print("\n" + "="*70)
    print(" VALIDATION TEST DATA QUALITY DIAGNOSTICS")
    print("="*70)
    
    print("\nUsing data files:")
    print(f"  Base instances: {base_path}")
    print(f"  Surface form: {surface_path}")
    print(f"  Options context: {options_path}")
    print(f"  Feature order: {feature_path}")
    
    print("\nLoading data files...")
    base = load_jsonl(base_path)
    surface = load_jsonl(surface_path)
    options = load_jsonl(options_path)
    feature = load_jsonl(feature_path)
    
    print(f"  Base instances: {len(base)}")
    print(f"  Surface form: {len(surface)}")
    print(f"  Options context: {len(options)}")
    print(f"  Feature order: {len(feature)}")
    
    # Run diagnostics
    base_stats = diagnose_base_instances(base)
    surface_stats = diagnose_surface_form(surface)
    options_stats = diagnose_options_context(options)
    feature_stats = diagnose_feature_order(feature)
    
    diagnose_cross_test_consistency(base, surface, options, feature)
    diagnose_edge_cases(surface, options, feature)
    print_sample_instances(surface, options, feature)
    
    # Final summary
    print_section("FINAL SUMMARY")
    
    print(f"""
  BASE INSTANCES
    Total: {base_stats['n_instances']}
    Surveys: {base_stats['n_surveys']}
    Countries: {base_stats['n_countries']}
    Unique questions: {base_stats['n_questions']}

  SURFACE FORM TEST
    Total: {surface_stats['n_instances']}
    With synonym variations: {surface_stats['has_synonym']} ({surface_stats['has_synonym']/surface_stats['n_instances']*100:.1f}%)
    With reorder variations: {surface_stats['has_reorder']} ({surface_stats['has_reorder']/surface_stats['n_instances']*100:.1f}%)
    Option-level synonym coverage: {surface_stats['synonym_option_coverage']*100:.1f}%
    Unique scale patterns: {surface_stats['n_option_patterns']}

  OPTIONS CONTEXT TEST
    Total: {options_stats['n_instances']}
    Scale questions: {options_stats['n_scale']} ({options_stats['n_scale']/options_stats['n_instances']*100:.1f}%)
    Categorical questions: {options_stats['n_categorical']} ({options_stats['n_categorical']/options_stats['n_instances']*100:.1f}%)
    Position errors: {options_stats['position_errors']}

  FEATURE ORDER TEST
    Total: {feature_stats['n_instances']}
    Feature range: {feature_stats['min_features']}-{feature_stats['max_features']}
    Ordering errors: {feature_stats['ordering_errors']}
    Content mismatches: {feature_stats['content_mismatch']}
""")

if __name__ == "__main__":
    main()