#!/usr/bin/env python
"""
Analyze and visualize disaggregated results by region and topic section.
Creates publication-ready figures for ICML paper.

Usage:
    python plot_disaggregated.py                    # All surveys
    python plot_disaggregated.py --survey-filter wvs  # WVS survey only
    python plot_disaggregated.py --survey-filter afrobarometer  # Afrobarometer only
"""

import argparse
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import sys

# Add scripts directory to path to import from analyze_disaggregated
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_script_dir))

from synthetic_sampling.evaluation import compute_instance_metrics, ParsedInstance, ResultsAnalyzer

# Shared cache location - all analysis scripts use this same cache
SHARED_CACHE_DIR = Path(__file__).parent.parent / 'analysis' / '.cache' / 'enriched'

# Helper functions for survey filtering (copied from analyze_disaggregated.py)
def _load_canonical_mapping(path: Path):
    """Load country_canonical_mapping.json. Exclude keys starting with _."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}

def _load_region_mapping(path: Path):
    """Load country_to_region.json (ISO-2 -> region). Exclude keys starting with _."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: str(v) for k, v in data.items() if not k.startswith("_")}

def _normalize_raw(raw):
    """Strip and remove trailing .0 from numeric strings."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _to_canonical_iso2(raw, survey, canonical):
    """Map survey-specific raw country value to ISO 3166-1 alpha-2."""
    s = _normalize_raw(raw)
    if not s:
        return None
    by_survey = canonical.get("by_survey") or {}
    iso_numeric = canonical.get("iso_numeric") or {}
    survey_map = by_survey.get(survey)
    if survey_map is not None and s in survey_map:
        return survey_map[s]
    if survey in ("wvs", "latinobarometer") and s in iso_numeric:
        return iso_numeric[s]
    if survey in ("ess_wave_10", "ess_wave_11") and len(s) == 2 and s.isalpha():
        return s.upper()
    if s in iso_numeric:
        return iso_numeric[s]
    return None

def _country_iso2(inst, canonical):
    """Resolve ISO-2 for an instance. Fallback: ESS-style respondent_id prefix or UNKNOWN."""
    iso2 = _to_canonical_iso2(inst.country, inst.survey, canonical)
    if iso2:
        return iso2
    rid = str(inst.respondent_id)
    if len(rid) >= 2 and rid[:2].isalpha():
        if len(rid) == 2 or (len(rid) > 2 and (rid[2] in "_" or rid[2].isdigit())):
            return rid[:2].upper()
    return "UNKNOWN"

def _metrics_by_region(instances, canonical, region_mapping, min_n):
    """Group instances by region (ISO-2 -> region via mapping) and compute metrics."""
    by_region = defaultdict(list)
    for inst in instances:
        iso2 = _country_iso2(inst, canonical)
        region = region_mapping.get(iso2) or "Unknown"
        by_region[region].append(inst)
    return {
        k: compute_instance_metrics(v).to_dict()
        for k, v in sorted(by_region.items())
        if len(v) >= min_n
    }

def load_enriched_instances_from_cache(cache_dir, model_name=None, max_instances_per_file=None):
    """
    Load enriched instances from cache directory.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing cache files
    model_name : str, optional
        Filter to specific model
    max_instances_per_file : int, optional
        If provided, only load first N instances from each file (for faster baseline calculation)
    """
    if not cache_dir.exists():
        return {}
    
    enriched_data = {}
    pattern = f"{model_name}_*.json" if model_name else "*.json"
    cache_files = list(cache_dir.glob(pattern))
    
    if not cache_files:
        return {}
    
    print(f"  Found {len(cache_files)} cache file(s)")
    
    for i, cache_file in enumerate(cache_files, 1):
        # Extract model name from filename (format: {model_name}_{cache_key}.json)
        filename = cache_file.stem
        if "_" in filename:
            parts = filename.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 32:  # MD5 hash is 32 chars
                model = parts[0]
            else:
                model = parts[0]
        else:
            continue
            
        # Load cache file
        try:
            print(f"    [{i}/{len(cache_files)}] Loading {cache_file.name}...", end='\r')
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Limit instances if requested (for faster baseline calculation)
            if max_instances_per_file and len(data) > max_instances_per_file:
                data = data[:max_instances_per_file]
            
            instances = [ParsedInstance(**inst_dict) for inst_dict in data]
            if model not in enriched_data:
                enriched_data[model] = []
            enriched_data[model].extend(instances)
        except Exception as e:
            print(f"\n    Warning: Could not load cache {cache_file.name}: {e}")
    
    print()  # New line after progress indicators
    return enriched_data

CAN_USE_SURVEY_FILTER = True

# =============================================================================
# BASELINE CALCULATIONS
# =============================================================================

def calculate_majority_baseline(instances):
    """
    Calculate majority class baseline accuracy.
    Returns the proportion of instances that belong to the most common class.
    """
    if not instances:
        return 0.0
    from collections import Counter
    class_counts = Counter(inst.ground_truth for inst in instances)
    if not class_counts:
        return 0.0
    majority_count = max(class_counts.values())
    return majority_count / len(instances)

def calculate_baselines_by_dimension(instances_dict, dimension_key='target_section'):
    """
    Calculate majority baselines for each dimension (section/topic/region).
    
    Parameters
    ----------
    instances_dict : dict
        {model: [ParsedInstance, ...]} or aggregated data structure
    dimension_key : str
        'target_section', 'target_topic_tag', or 'region' (requires region mapping)
    
    Returns
    -------
    dict : {dimension: baseline_accuracy}
    """
    baselines = {}
    
    # If we have instances, calculate from them
    if instances_dict and isinstance(next(iter(instances_dict.values())), list):
        # Group by dimension
        by_dim = defaultdict(list)
        for model_instances in instances_dict.values():
            for inst in model_instances:
                if dimension_key == 'target_section':
                    dim = inst.target_section
                elif dimension_key == 'target_topic_tag':
                    dim = inst.target_topic_tag
                else:
                    continue
                if dim:
                    by_dim[dim].append(inst)
        
        # Calculate baseline for each dimension
        for dim, dim_instances in by_dim.items():
            baselines[dim] = calculate_majority_baseline(dim_instances)
    
    return baselines

def get_baselines_from_cache(survey_filter=None, max_instances_per_group=10000):
    """
    Load instances from cache and calculate majority baselines per section/topic.
    Uses sampling to avoid loading millions of instances.
    
    Parameters
    ----------
    survey_filter : str, optional
        Filter to specific survey
    max_instances_per_group : int
        Maximum instances to sample per section/topic for baseline calculation (default: 10000)
        This is sufficient for accurate baseline estimates while keeping performance reasonable.
    
    Returns
    -------
    tuple : (section_baselines dict, topic_baselines dict)
    """
    cache_dir = SHARED_CACHE_DIR
    
    if not cache_dir.exists():
        return None, None
    
    print("  Loading cache files (sampling for performance)...")
    # Sample up to 50k instances per file to speed up baseline calculation
    # This is sufficient for accurate baseline estimates
    enriched_data = load_enriched_instances_from_cache(cache_dir, max_instances_per_file=50000)
    if not enriched_data:
        return None, None
    
    print(f"  Loaded data from {len(enriched_data)} model(s)")
    
    # Group by section/topic while loading (single pass, more efficient)
    by_section_inst = defaultdict(list)
    by_topic_inst = defaultdict(list)
    total_loaded = 0
    
    # Process models one at a time to reduce memory footprint
    for model_name, instances in enriched_data.items():
        print(f"    Processing {model_name}: {len(instances):,} instances...", end='\r')
        
        for inst in instances:
            # Apply survey filter early
            if survey_filter and inst.survey != survey_filter:
                continue
            
            total_loaded += 1
            
            # Group by section/topic
            if inst.target_section:
                # Only keep up to max_instances_per_group per section
                if len(by_section_inst[inst.target_section]) < max_instances_per_group:
                    by_section_inst[inst.target_section].append(inst)
            if inst.target_topic_tag:
                # Only keep up to max_instances_per_group per topic
                if len(by_topic_inst[inst.target_topic_tag]) < max_instances_per_group:
                    by_topic_inst[inst.target_topic_tag].append(inst)
    
    print(f"  Processed {total_loaded:,} instances total")
    
    if not by_section_inst and not by_topic_inst:
        return None, None
    
    # Calculate baselines (now much faster since we've limited instances per group)
    print("  Calculating baselines...")
    section_baselines = {sec: calculate_majority_baseline(insts) 
                        for sec, insts in by_section_inst.items() if len(insts) >= 100}
    topic_baselines = {topic: calculate_majority_baseline(insts) 
                       for topic, insts in by_topic_inst.items() if len(insts) >= 50}
    
    return section_baselines, topic_baselines

# =============================================================================
# LOAD DATA
# =============================================================================

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot disaggregated analysis results by region and topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--survey-filter',
        type=str,
        default=None,
        help='Filter heatmap to specific survey (e.g., "wvs", "afrobarometer"). '
             'If not specified, uses all surveys. Requires enriched cache from analyze_disaggregated.py'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for figures (default: analysis/figures)'
    )
    parser.add_argument(
        '--country-in-profile-only',
        action='store_true',
        help='Use disaggregated results filtered to instances with country/region in profile features '
             '(uses *_country_in_profile.json and a separate default output folder).'
    )
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip baseline calculation (uses default values). Much faster for large datasets.'
    )
    return parser.parse_args()

# Parse arguments
args = parse_args()
survey_filter = args.survey_filter
country_in_profile_only = args.country_in_profile_only

# Load all data
analysis_dir = Path(__file__).parent.parent / 'analysis' / 'disaggregated'

def re_aggregate_from_cache(survey_filter, min_n=100, min_n_tag=50):
    """
    Re-aggregate data from enriched cache filtered to a specific survey.
    Returns dictionaries in the same format as the JSON files.
    """
    print(f"\n{'='*80}")
    print(f"Re-aggregating data filtered to survey: {survey_filter}")
    print(f"{'='*80}")
    
    cache_dir = SHARED_CACHE_DIR
    
    if not cache_dir.exists():
        print(f"Error: Cache directory not found at {cache_dir}")
        print(f"Run analyze_disaggregated.py first to create the enriched cache.")
        return None
    
    # Load enriched instances from cache
    enriched_data = load_enriched_instances_from_cache(cache_dir)
    
    if not enriched_data:
        print(f"Error: No cached enriched data found.")
        return None
    
    # Load mappings needed for aggregation
    script_dir = Path(__file__).resolve().parent
    canonical_path = script_dir / "country_canonical_mapping.json"
    region_path = script_dir / "country_to_region.json"
    
    if not canonical_path.exists():
        print(f"Error: Canonical mapping not found at {canonical_path}")
        return None
    
    canonical = _load_canonical_mapping(canonical_path)
    region_mapping = _load_region_mapping(region_path) if region_path.exists() else {}
    
    # Re-aggregate for each model
    by_region_filtered = {}
    by_section_filtered = {}
    by_topic_tag_filtered = {}
    by_country_filtered = {}
    
    for model_name, instances in enriched_data.items():
        # Filter instances to specified survey
        filtered_instances = [inst for inst in instances if inst.survey == survey_filter]
        
        if not filtered_instances:
            print(f"  {model_name}: No instances from {survey_filter}, skipping")
            continue
        
        print(f"  {model_name}: {len(filtered_instances):,} instances from {survey_filter}")
        
        # Use ResultsAnalyzer for section and topic_tag (same as analyze_disaggregated.py)
        analyzer = ResultsAnalyzer(filtered_instances)
        
        # Aggregate by section
        by_sec = analyzer.metrics_by_section(min_n=min_n)
        by_section_filtered[model_name] = {k: v.to_dict() for k, v in by_sec.items()}
        
        # Aggregate by topic tag
        by_tag = analyzer.metrics_by_topic_tag(min_n=min_n_tag)
        by_topic_tag_filtered[model_name] = {k: v.to_dict() for k, v in by_tag.items()}
        
        # Aggregate by country
        by_country_model = defaultdict(list)
        for inst in filtered_instances:
            iso2 = _country_iso2(inst, canonical)
            by_country_model[iso2].append(inst)
        by_country_filtered[model_name] = {
            k: compute_instance_metrics(v).to_dict()
            for k, v in sorted(by_country_model.items())
            if len(v) >= min_n
        }
        
        # Aggregate by region (if region mapping available)
        if region_mapping:
            by_reg = _metrics_by_region(filtered_instances, canonical, region_mapping, min_n=min_n)
            by_region_filtered[model_name] = by_reg
        else:
            # If no region mapping, create empty dict (will be handled gracefully)
            by_region_filtered[model_name] = {}
    
    print(f"\nRe-aggregation complete!")
    return {
        'by_region': by_region_filtered,
        'by_section': by_section_filtered,
        'by_topic_tag': by_topic_tag_filtered,
        'by_country': by_country_filtered,
    }

# Load or re-aggregate data based on flags
if country_in_profile_only:
    if survey_filter:
        print("\n[WARNING] Ignoring --survey-filter when --country-in-profile-only is set;")
        print("          using pre-aggregated *_country_in_profile.json instead.")
    suffix = '_country_in_profile'
    by_region = load_json(analysis_dir / f'by_region{suffix}.json')
    by_section = load_json(analysis_dir / f'by_section{suffix}.json')
    by_topic = load_json(analysis_dir / f'by_topic_tag{suffix}.json')
    by_country = load_json(analysis_dir / f'by_country{suffix}.json')
elif survey_filter and CAN_USE_SURVEY_FILTER:
    filtered_data = re_aggregate_from_cache(survey_filter)
    if filtered_data:
        by_region = filtered_data['by_region']
        by_section = filtered_data['by_section']
        by_topic = filtered_data['by_topic_tag']
        by_country = filtered_data['by_country']
        print(f"\nUsing filtered data for survey: {survey_filter}")
    else:
        print(f"\nWarning: Could not filter by survey. Using all surveys.")
        by_region = load_json(analysis_dir / 'by_region.json')
        by_section = load_json(analysis_dir / 'by_section.json')
        by_topic = load_json(analysis_dir / 'by_topic_tag.json')
        by_country = load_json(analysis_dir / 'by_country.json')
else:
    # Load pre-aggregated data (all surveys)
    by_region = load_json(analysis_dir / 'by_region.json')
    by_section = load_json(analysis_dir / 'by_section.json')
    by_topic = load_json(analysis_dir / 'by_topic_tag.json')
    by_country = load_json(analysis_dir / 'by_country.json')

# Create figures directory
if args.output_dir:
    figures_dir = args.output_dir
else:
    default_subdir = 'disaggregated_analysis_country_in_profile' if country_in_profile_only else 'disaggregated_analysis'
    figures_dir = Path(__file__).parent.parent / 'analysis' / 'figures' / default_subdir
figures_dir.mkdir(parents=True, exist_ok=True)

# Calculate baselines from cache
# Skip slow baseline calculation when using pre-aggregated filtered data or --skip-baselines flag
if country_in_profile_only or args.skip_baselines:
    if country_in_profile_only:
        print("\nSkipping baseline calculation (using pre-aggregated filtered data).")
    else:
        print("\nSkipping baseline calculation (--skip-baselines flag set).")
    print("  Using default baseline values for reference lines.")
    section_baselines = {}
    topic_baselines = {}
    avg_section_baseline = 0.499  # Default fallback
    avg_topic_baseline = 0.499
else:
    print("\nCalculating majority class baselines from cached instances...")
    print("  (This may take a few minutes for large datasets. Use --skip-baselines to skip.)")
    section_baselines, topic_baselines = get_baselines_from_cache(survey_filter)
    if section_baselines:
        print(f"  Calculated baselines for {len(section_baselines)} sections, {len(topic_baselines) if topic_baselines else 0} topics")
        # Calculate average baseline for reference lines
        avg_section_baseline = np.mean(list(section_baselines.values())) if section_baselines else 0.5
        avg_topic_baseline = np.mean(list(topic_baselines.values())) if topic_baselines else 0.5
        print(f"  Average section baseline: {avg_section_baseline:.1%}")
        print(f"  Average topic baseline: {avg_topic_baseline:.1%}")
    else:
        print("  Warning: Could not calculate baselines. Using default values.")
        section_baselines = {}
        topic_baselines = {}
        avg_section_baseline = 0.499  # Default fallback
        avg_topic_baseline = 0.499

random_baseline = 0.219  # Approximate random baseline (1/n_classes for typical questions)

# =============================================================================
# AGGREGATE ACROSS MODELS
# =============================================================================

def aggregate_by_dimension(data, metric='accuracy'):
    """
    Aggregate results across models for each dimension (region/section/topic).
    Filters out dimensions with no valid data (n=0 or NaN values).
    Returns: {dimension: {'mean': float, 'std': float, 'se': float, 'min': float, 'max': float, 'n': int}}
    
    Standard error (se) is computed using binomial standard error: sqrt(p*(1-p)/n)
    where p is the mean accuracy and n is the number of observations.
    This represents the statistical uncertainty in the accuracy estimate.
    """
    dim_values = defaultdict(list)
    dim_n = defaultdict(int)
    
    for model, dims in data.items():
        for dim, metrics in dims.items():
            if dim == 'Unknown':
                continue
            # Skip dimensions with no data or invalid metrics
            n_obs = metrics.get('n', 0)
            acc_value = metrics.get(metric, 0)
            
            # Only include if we have valid data (n > 0 and valid accuracy)
            if n_obs > 0 and not np.isnan(acc_value) and acc_value is not None:
                dim_values[dim].append(acc_value)
                dim_n[dim] = n_obs  # Same across models
    
    results = {}
    for dim, values in dim_values.items():
        if not values:  # Skip if no valid values
            continue
            
        mean_acc = np.mean(values)
        n_obs = dim_n[dim]
        
        # Standard error of proportion (binomial): SE = sqrt(p*(1-p)/n)
        # This represents uncertainty in the accuracy estimate based on sample size
        se = np.sqrt(mean_acc * (1 - mean_acc) / n_obs) if n_obs > 0 else 0
        
        # Also keep std across models for reference (model-to-model variation)
        std_across_models = np.std(values)
        
        results[dim] = {
            'mean': mean_acc,
            'std': std_across_models,  # Variation across models
            'se': se,  # Standard error based on n observations
            'min': np.min(values),
            'max': np.max(values),
            'n': n_obs,
            'n_models': len(values)
        }
    return results


# =============================================================================
# PRINT SUMMARY TABLES
# =============================================================================

print("=" * 80)
print("GEOGRAPHIC VARIATION (By Region)")
print("=" * 80)

region_stats = aggregate_by_dimension(by_region, 'accuracy')
sorted_regions = sorted(region_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

print(f"\n{'Region':<20} {'Mean Acc':>10} {'SE':>8} {'Std':>8} {'Range':>15} {'N':>10}")
print("-" * 75)
for region, stats in sorted_regions:
    range_str = f"{stats['min']:.1%}-{stats['max']:.1%}"
    print(f"{region:<20} {stats['mean']:>9.1%} {stats['se']:>7.1%} {stats['std']:>7.1%} {range_str:>15} {stats['n']:>10,}")

# Compute gap
best_region = sorted_regions[0]
worst_region = sorted_regions[-1]
gap = best_region[1]['mean'] - worst_region[1]['mean']
print(f"\nGap between best ({best_region[0]}) and worst ({worst_region[0]}): {gap:.1%}")


print("\n" + "=" * 80)
print("TOPIC VARIATION (By Thematic Section)")
print("=" * 80)

section_stats = aggregate_by_dimension(by_section, 'accuracy')
sorted_sections = sorted(section_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

print(f"\n{'Section':<25} {'Mean Acc':>10} {'SE':>8} {'Std':>8} {'Range':>15} {'N':>10}")
print("-" * 80)
for section, stats in sorted_sections:
    range_str = f"{stats['min']:.1%}-{stats['max']:.1%}"
    section_name = section.replace('_', ' ').title()
    print(f"{section_name:<25} {stats['mean']:>9.1%} {stats['se']:>7.1%} {stats['std']:>7.1%} {range_str:>15} {stats['n']:>10,}")

best_section = sorted_sections[0]
worst_section = sorted_sections[-1]
gap = best_section[1]['mean'] - worst_section[1]['mean']
print(f"\nGap between best ({best_section[0]}) and worst ({worst_section[0]}): {gap:.1%}")


print("\n" + "=" * 80)
print("TOPIC TAG VARIATION (Finer granularity)")
print("=" * 80)

topic_stats = aggregate_by_dimension(by_topic, 'accuracy')
sorted_topics = sorted(topic_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

print(f"\nTop 5 easiest topics:")
for topic, stats in sorted_topics[:5]:
    print(f"  {topic:<30} {stats['mean']:.1%} (n={stats['n']:,})")

print(f"\nTop 5 hardest topics:")
for topic, stats in sorted_topics[-5:]:
    print(f"  {topic:<30} {stats['mean']:.1%} (n={stats['n']:,})")


print("\n" + "=" * 80)
print("COUNTRY VARIATION")
print("=" * 80)

country_stats = aggregate_by_dimension(by_country, 'accuracy')
sorted_countries = sorted(country_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

print(f"\nTotal countries: {len(country_stats)}")
print(f"\nTop 10 best-predicted countries:")
for country, stats in sorted_countries[:10]:
    print(f"  {country:<25} {stats['mean']:.1%} (n={stats['n']:,})")

print(f"\nTop 10 worst-predicted countries:")
for country, stats in sorted_countries[-10:]:
    print(f"  {country:<25} {stats['mean']:.1%} (n={stats['n']:,})")


# =============================================================================
# GENERATE FIGURES
# =============================================================================

# ICML figure guidelines: lines at least 0.5pt, 9pt font, no titles in figure
plt.rcParams.update({
    'font.size': 9,
    'axes.linewidth': 0.5,
    'font.family': 'sans-serif',
    'lines.linewidth': 0.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Figure 2: Geographic and Topic Variation (two panels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Geographic variation
regions = [r[0] for r in sorted_regions]
means = [r[1]['mean'] for r in sorted_regions]
# Use standard error (based on n observations) for error bars, not std across models
ses = [r[1]['se'] for r in sorted_regions]

# Professional color palette: print-friendly, accessible, distinguishable in grayscale
# Using ColorBrewer-inspired palette with good contrast
def get_region_color(region):
    if 'Africa' in region:
        return '#8B4513'  # Brown/Sienna (distinguishable in grayscale)
    elif 'Europe' in region:
        return '#2E5090'  # Deep blue (darker for better print contrast)
    elif 'America' in region or 'Caribbean' in region:
        return '#4A7C3F'  # Forest green (darker, print-friendly)
    elif 'Asia' in region:
        return '#6B4C93'  # Deep purple (darker, better contrast)
    else:
        return '#666666'  # Medium gray (better than light gray)

colors = [get_region_color(r) for r in regions]

y_pos = np.arange(len(regions))
# Use 95% confidence intervals: ±1.96*SE
ci_95 = [1.96 * se for se in ses]
ax1.barh(y_pos, means, xerr=ci_95, color=colors, edgecolor='black', linewidth=0.5, 
         capsize=2, error_kw={'linewidth': 0.5, 'capthick': 0.5})
ax1.set_yticks(y_pos)
ax1.set_yticklabels(regions, fontsize=8)
ax1.set_xlabel('Accuracy (mean ± 95% CI)', fontsize=9)
ax1.set_xlim(0.15, 0.45)
# Reference lines: use darker colors and ensure 0.5pt thickness
# Reference lines: Random baseline (1/n_classes, approximate) and Majority baseline
# For regions, we use average baseline across all data
random_baseline = 0.219  # Approximate random baseline (would need to know number of classes per question)
region_baseline = avg_section_baseline if section_baselines else 0.499
ax1.axvline(x=random_baseline, color='#666666', linestyle='--', linewidth=0.5, alpha=0.7, label='Random')
ax1.axvline(x=region_baseline, color='#000000', linestyle='--', linewidth=0.5, alpha=0.7, label='Majority')
# Remove title per ICML guidelines (caption serves this function)
ax1.invert_yaxis()

# Add legend for regions
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#8B4513', edgecolor='black', linewidth=0.5, label='Africa'),
    Patch(facecolor='#2E5090', edgecolor='black', linewidth=0.5, label='Europe'),
    Patch(facecolor='#4A7C3F', edgecolor='black', linewidth=0.5, label='Americas'),
    Patch(facecolor='#6B4C93', edgecolor='black', linewidth=0.5, label='Asia'),
    Patch(facecolor='#666666', edgecolor='black', linewidth=0.5, label='Other'),
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=7, frameon=True, framealpha=0.9)

# Panel B: Topic section variation
sections = [s[0].replace('_', ' ').title() for s in sorted_sections]
means = [s[1]['mean'] for s in sorted_sections]
# Use standard error (based on n observations) for error bars
ses = [s[1]['se'] for s in sorted_sections]

y_pos = np.arange(len(sections))
# Use a professional blue color (darker for better print contrast)
# Use 95% confidence intervals: ±1.96*SE
ci_95 = [1.96 * se for se in ses]
ax2.barh(y_pos, means, xerr=ci_95, color='#2E5090', edgecolor='black', linewidth=0.5,
         capsize=2, error_kw={'linewidth': 0.5, 'capthick': 0.5})
ax2.set_yticks(y_pos)
ax2.set_yticklabels(sections, fontsize=8)
ax2.set_xlabel('Accuracy (mean ± 95% CI)', fontsize=9)
ax2.set_xlim(0.15, 0.50)
# Reference lines: use darker colors and ensure 0.5pt thickness
# Reference lines: Random and Majority baselines
section_baseline_val = avg_section_baseline if section_baselines else 0.499
ax2.axvline(x=random_baseline, color='#666666', linestyle='--', linewidth=0.5, alpha=0.7, label='Random')
ax2.axvline(x=section_baseline_val, color='#000000', linestyle='--', linewidth=0.5, alpha=0.7, label='Majority')
# Remove title per ICML guidelines (caption serves this function)
ax2.invert_yaxis()

plt.tight_layout()
# Determine filename/label suffix for this run
output_suffix = ''
label_parts = []
if country_in_profile_only:
    output_suffix = '_country_in_profile'
    label_parts.append('country_in_profile')
elif survey_filter:
    output_suffix = f'_{survey_filter}'
    label_parts.append(survey_filter)
plt.savefig(figures_dir / f'figure2_geographic_topic{output_suffix}.pdf', bbox_inches='tight', dpi=300)
plt.savefig(figures_dir / f'figure2_geographic_topic{output_suffix}.png', bbox_inches='tight', dpi=300)
survey_label = f" ({', '.join(label_parts)})" if label_parts else ""
print(f"\n[OK] Saved Figure 2 (geographic + topic variation{survey_label})")


# =============================================================================
# ALTERNATIVE: Heatmap showing model × region
# =============================================================================

def get_geographic_region_order():
    """
    Define a consistent geographic ordering for regions.
    Groups by continent/sub-region in a logical geographic order.
    This ordering is consistent across all plots (all-surveys and WVS-only).
    
    Returns a list of region names in the desired order.
    Regions not in this list will be appended at the end, sorted alphabetically.
    """
    # Define preferred geographic ordering based on actual region names in data
    # Group by continent, then sub-region (West to East, North to South)
    preferred_order = [
        # Americas (North to South)
        'North America',
        'Central America',
        'Caribbean',
        'South America',
        # Europe (North to South, West to East)
        'Northern Europe',
        'Western Europe',
        'Southern Europe',
        'Eastern Europe',
        # Africa (North to South, West to East)
        'North Africa',
        'West Africa',
        'Central Africa',
        'East Africa',
        'Southern Africa',
        # Middle East & Central Asia
        'Middle East',
        'Central Asia',
        # Asia (West to East, North to South)
        'South Asia',
        'Southeast Asia',
        'East Asia',
        # Oceania
        'Oceania',
    ]
    return preferred_order

def sort_regions_geographically(regions):
    """
    Sort regions using a geographic ordering that groups by continent/sub-region.
    Regions in the preferred order come first, then others alphabetically.
    
    Parameters
    ----------
    regions : list or set
        List of region names to sort
    
    Returns
    -------
    list
        Sorted list of regions
    """
    preferred_order = get_geographic_region_order()
    region_list = list(regions)
    
    # Separate into preferred and other regions
    preferred_regions = []
    other_regions = []
    
    for region in region_list:
        if region in preferred_order:
            preferred_regions.append(region)
        else:
            other_regions.append(region)
    
    # Sort preferred regions by their position in preferred_order
    preferred_regions.sort(key=lambda r: preferred_order.index(r))
    
    # Sort other regions alphabetically
    other_regions.sort()
    
    # Combine: preferred first, then others
    return preferred_regions + other_regions

def create_region_heatmap(by_region_data, output_suffix=""):
    """
    Create a heatmap of model × region accuracy.
    Regions are grouped by continent and sorted by average accuracy within continent.
    Models are sorted by average accuracy (best on left).
    
    Parameters
    ----------
    by_region_data : dict
        Dictionary with structure {model: {region: {metrics...}}}
        Data should already be filtered if survey filter was applied
    output_suffix : str
        Suffix to add to output filename (e.g., '_wvs')
    """
    by_region_filtered = by_region_data
    
    # Prepare data matrix
    models = list(by_region_filtered.keys())
    if not models:
        print("  Error: No data available for heatmap")
        return
    
    # Parse into DataFrame
    records = []
    for model, regions in by_region_filtered.items():
        for region, metrics in regions.items():
            if region == 'Unknown':
                continue
            # Handle key variations if any
            acc = metrics.get('accuracy', metrics.get('correct', 0))
            records.append({'Model': model, 'Region': region, 'Accuracy': acc})
    
    if not records:
        print("  Error: No data records found")
        return
    
    df = pd.DataFrame(records)
    
    # Define Continent Mapping for Grouping
    continent_map = {
        'Central Africa': 'Africa', 'East Africa': 'Africa', 'North Africa': 'Africa', 
        'Southern Africa': 'Africa', 'West Africa': 'Africa',
        'Central America': 'Americas', 'North America': 'Americas', 'South America': 'Americas', 'Caribbean': 'Americas',
        'Central Asia': 'Asia', 'East Asia': 'Asia', 'South Asia': 'Asia', 'Southeast Asia': 'Asia', 'Middle East': 'Asia',
        'Eastern Europe': 'Europe', 'Northern Europe': 'Europe', 'Southern Europe': 'Europe', 'Western Europe': 'Europe',
        'Oceania': 'Oceania'
    }
    df['Continent'] = df['Region'].map(continent_map)
    
    # Fill missing continents (if any regions not in map)
    df['Continent'] = df['Continent'].fillna('Other')
    
    # 1. Sort Regions: First by Continent, then by Average Accuracy within Continent
    region_stats = df.groupby(['Continent', 'Region'])['Accuracy'].mean().reset_index()
    region_stats = region_stats.sort_values(['Continent', 'Accuracy'], ascending=[True, True])
    region_order = region_stats['Region'].tolist()
    
    # 2. Sort Models: By Average Accuracy (Best on Left)
    model_stats = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
    model_order = model_stats.index.tolist()
    
    # Pivot for Heatmap
    pivot_df = df.pivot(index='Region', columns='Model', values='Accuracy')
    pivot_df = pivot_df.reindex(index=region_order, columns=model_order)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create Heatmap using seaborn
    # cmap='RdYlGn' is intuitive (Red=Bad, Green=Good)
    sns_heatmap = sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn", 
        linewidths=0.5, 
        cbar_kws={'label': 'Prediction Accuracy'},
        ax=ax,
        vmin=pivot_df.min().min() if not pivot_df.empty else 0.20,
        vmax=pivot_df.max().max() if not pivot_df.empty else 0.42,
        annot_kws={'size': 8}  # Font size for annotations
    )
    
    # Formatting - use ICML-compliant font sizes
    ax.set_xlabel('Models (Sorted by Performance)', fontsize=9)
    ax.set_ylabel('Regions (Grouped by Continent, Sorted by Difficulty)', fontsize=9)
    
    # Rotate x-labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    # Ensure colorbar formatting matches ICML guidelines
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label('Prediction Accuracy', fontsize=9)
    cbar.ax.tick_params(width=0.5, labelsize=8)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    filename_base = f'figure_heatmap_region{output_suffix}'
    plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
    
    survey_label = f" ({output_suffix[1:]})" if output_suffix else ""  # Remove leading underscore
    print(f"[OK] Saved heatmap (model x region{survey_label})")
    plt.close(fig)

# Create heatmap (data is already filtered if survey_filter was provided)
create_region_heatmap(by_region, output_suffix=output_suffix)


# =============================================================================
# MODEL × SECTION HEATMAP
# =============================================================================

def create_section_heatmap(by_section_data, output_suffix=""):
    """
    Create a heatmap of model × section accuracy.
    
    Parameters
    ----------
    by_section_data : dict
        Dictionary with structure {model: {section: {metrics...}}}
        Data should already be filtered if survey filter was applied
    output_suffix : str
        Suffix to add to output filename (e.g., '_wvs')
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    by_section_filtered = by_section_data
    
    # Prepare data matrix
    models = list(by_section_filtered.keys())
    if not models:
        print("  Error: No data available for section heatmap")
        plt.close(fig)
        return
    
    # Get all unique sections across all models
    all_sections = set()
    for model_data in by_section_filtered.values():
        all_sections.update(model_data.keys())
    all_sections.discard('Unknown')
    section_names = sorted(all_sections)  # Sort sections alphabetically
    
    if not section_names:
        print("  Error: No sections found in data")
        plt.close(fig)
        return

    # Model metadata for consistent sorting and display
    model_metadata = {
        'deepseek-v3p1-terminus': ('DeepSeek', 37, 1, 'DeepSeek'),
        'gemma-3-27b-instruct': ('Gemma', 27, 1, 'Gemma 27B'),
        'gpt_oss': ('GPT-OSS', 120, 1, 'GPT-OSS'),
        'llama3.1-70b-base': ('Llama', 70, 0, 'Llama 70B (b)'),
        'llama3.1-70b-instruct': ('Llama', 70, 1, 'Llama 70B (i)'),
        'llama3.1-8b-base': ('Llama', 8, 0, 'Llama 8B (b)'),
        'llama3.1-8b-instruct': ('Llama', 8, 1, 'Llama 8B (i)'),
        'olmo3-32b-base': ('OLMo', 32, 0, 'OLMo 32B (b)'),
        'olmo3-32b-dpo': ('OLMo', 32, 1, 'OLMo 32B (i)'),
        'olmo3-7b-base': ('OLMo', 7, 0, 'OLMo 7B (b)'),
        'olmo3-7b-dpo': ('OLMo', 7, 1, 'OLMo 7B (i)'),
        'qwen3-32b': ('Qwen', 32, 1, 'Qwen 32B'),
        'qwen3-4b': ('Qwen', 4, 1, 'Qwen 4B'),
    }
    
    # Short model names for display
    model_short = {m: metadata[3] for m, metadata in model_metadata.items()}
    
    # Sort models by: family (alphabetically), then size (largest first), then type (base before instruct)
    def get_model_sort_key(model_name):
        if model_name in model_metadata:
            family, size, model_type, _ = model_metadata[model_name]
            return (family, -size, model_type)  # Negative size for descending order
        else:
            # Fallback: sort by name if not in metadata
            return ('ZZZ', 0, 1)  # Put unknown models at the end
    
    sorted_models = sorted(models, key=get_model_sort_key)
    
    # Build matrix
    matrix = np.zeros((len(sorted_models), len(section_names)))
    for i, model in enumerate(sorted_models):
        for j, section in enumerate(section_names):
            if section in by_section_filtered[model]:
                matrix[i, j] = by_section_filtered[model][section]['accuracy']
    
    # Determine color scale from data
    vmin = np.nanmin(matrix[matrix > 0]) if np.any(matrix > 0) else 0.20
    vmax = np.nanmax(matrix) if np.any(matrix > 0) else 0.50
    
    # Use a colormap where green represents high accuracy (good) and red represents low accuracy (bad)
    # RdYlGn goes: red (low/bad) -> yellow -> green (high/good)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(section_names)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in section_names], 
                      rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(sorted_models)))
    ax.set_yticklabels([model_short.get(m, m) for m in sorted_models], fontsize=8)
    
    # Add colorbar with proper formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy', fontsize=9)
    cbar.ax.tick_params(width=0.5, labelsize=8)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    filename_base = f'figure_heatmap_section{output_suffix}'
    plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
    
    survey_label = f" ({output_suffix[1:]})" if output_suffix else ""  # Remove leading underscore
    print(f"[OK] Saved heatmap (model x section{survey_label})")
    plt.close(fig)

# Create section heatmap
create_section_heatmap(by_section, output_suffix=output_suffix)


# =============================================================================
# TOPIC PLOT: Top/Bottom N Topics
# =============================================================================

def create_topic_plot(by_topic_data, top_n=15, output_suffix=""):
    """
    Create a plot showing top N easiest and bottom N hardest topics.
    Uses a compact design suitable for many topics.
    """
    topic_stats = aggregate_by_dimension(by_topic_data, 'accuracy')
    sorted_topics = sorted(topic_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    if len(sorted_topics) < top_n * 2:
        top_n = len(sorted_topics) // 2
    
    # Get top and bottom topics
    top_topics = sorted_topics[:top_n]
    bottom_topics = sorted_topics[-top_n:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(6, top_n * 0.4)))
    
    # Panel A: Top N easiest topics
    topics_top = [t[0].replace('_', ' ').title() for t in top_topics]
    means_top = [t[1]['mean'] for t in top_topics]
    ses_top = [t[1]['se'] for t in top_topics]
    ci_95_top = [1.96 * se for se in ses_top]
    
    y_pos_top = np.arange(len(topics_top))
    ax1.barh(y_pos_top, means_top, xerr=ci_95_top, color='#2E5090', edgecolor='black', 
             linewidth=0.5, capsize=2, error_kw={'linewidth': 0.5, 'capthick': 0.5})
    ax1.set_yticks(y_pos_top)
    ax1.set_yticklabels(topics_top, fontsize=7)
    ax1.set_xlabel('Accuracy (mean ± 95% CI)', fontsize=9)
    
    # Add baseline if available
    if topic_baselines:
        topic_baseline_val = avg_topic_baseline
        ax1.axvline(x=topic_baseline_val, color='#000000', linestyle='--', 
                   linewidth=0.5, alpha=0.7, label='Majority')
    ax1.axvline(x=random_baseline, color='#666666', linestyle='--', 
               linewidth=0.5, alpha=0.7, label='Random')
    
    ax1.invert_yaxis()
    ax1.set_xlim(0.15, max(0.65, max(means_top) * 1.1))
    
    # Panel B: Bottom N hardest topics
    topics_bottom = [t[0].replace('_', ' ').title() for t in bottom_topics]
    means_bottom = [t[1]['mean'] for t in bottom_topics]
    ses_bottom = [t[1]['se'] for t in bottom_topics]
    ci_95_bottom = [1.96 * se for se in ses_bottom]
    
    y_pos_bottom = np.arange(len(topics_bottom))
    ax2.barh(y_pos_bottom, means_bottom, xerr=ci_95_bottom, color='#8B4513', 
             edgecolor='black', linewidth=0.5, capsize=2, 
             error_kw={'linewidth': 0.5, 'capthick': 0.5})
    ax2.set_yticks(y_pos_bottom)
    ax2.set_yticklabels(topics_bottom, fontsize=7)
    ax2.set_xlabel('Accuracy (mean ± 95% CI)', fontsize=9)
    
    # Add baseline if available
    if topic_baselines:
        ax2.axvline(x=topic_baseline_val, color='#000000', linestyle='--', 
                   linewidth=0.5, alpha=0.7, label='Majority')
    ax2.axvline(x=random_baseline, color='#666666', linestyle='--', 
               linewidth=0.5, alpha=0.7, label='Random')
    
    ax2.invert_yaxis()
    ax2.set_xlim(0.15, max(0.65, max(means_top) * 1.1))
    
    plt.tight_layout()
    filename_base = f'figure_topics{output_suffix}'
    plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
    survey_label = f" ({output_suffix[1:]})" if output_suffix else ""
    print(f"[OK] Saved topic plot (top/bottom {top_n} topics{survey_label})")
    plt.close(fig)

# Create topic plot
create_topic_plot(by_topic, top_n=15, output_suffix=output_suffix)


# =============================================================================
# COMPARISON PLOTS: Sections/Topics Across Regions and Surveys
# =============================================================================

def create_section_region_comparison(by_section_data, by_region_data, output_suffix=""):
    """
    Create a plot showing how models handle the same sections across different regions.
    Shows variation in accuracy for each section across regions.
    """
    # Get all sections and regions
    all_sections = set()
    all_regions = set()
    
    for model_data in by_section_data.values():
        all_sections.update(model_data.keys())
    for model_data in by_region_data.values():
        all_regions.update(model_data.keys())
    
    all_sections.discard('Unknown')
    all_regions.discard('Unknown')
    
    if not all_sections or not all_regions:
        print("  Skipping section-region comparison (insufficient data)")
        return
    
    # For each section, get accuracy by region (aggregate across models)
    section_region_acc = defaultdict(lambda: defaultdict(list))
    
    for model_name, section_data in by_section_data.items():
        for section, metrics in section_data.items():
            if section == 'Unknown':
                continue
            # Find which regions this section appears in
            # We need to map instances to regions - this requires enriched cache
            # For now, use a simplified approach: average across all regions
            section_region_acc[section]['all'].append(metrics['accuracy'])
    
    # Create a heatmap: Section × Region
    # This requires re-aggregating from cache with section+region breakdown
    print("  Note: Section-region comparison requires section×region cross-tabulation")
    print("  This would need enriched cache with both section and region info")
    
    # Alternative: Show section accuracy variation across models (already in by_section)
    # and region accuracy variation (already in by_region)
    # For true cross-tabulation, we'd need to re-aggregate from cache

def create_section_survey_comparison(by_section_data, output_suffix=""):
    """
    Create a plot showing how models handle the same sections across different surveys.
    Requires loading by_survey data and cross-tabulating with sections.
    """
    # Load by_survey data
    by_survey_data = load_json(analysis_dir / 'by_survey.json')
    
    # For each section, get accuracy by survey (aggregate across models)
    section_survey_acc = defaultdict(lambda: defaultdict(list))
    
    # This requires cross-tabulating section × survey from enriched cache
    print("  Note: Section-survey comparison requires section×survey cross-tabulation")
    print("  This would need enriched cache with both section and survey info")

def create_cross_comparison_plots(output_suffix="", survey_filter=None):
    """
    Create comparison plots showing how models handle same sections/topics 
    across regions and surveys. Requires re-aggregating from enriched cache.
    
    Parameters
    ----------
    output_suffix : str
        Suffix to add to output filenames (e.g., '_wvs')
    survey_filter : str, optional
        Survey to filter to (if None, uses all surveys)
    """
    if not CAN_USE_SURVEY_FILTER:
        print("\nSkipping cross-comparison plots (enriched cache not available)")
        return
    
    cache_dir = SHARED_CACHE_DIR
    if not cache_dir.exists():
        print("\nSkipping cross-comparison plots (cache not found)")
        return
    
    print("\n" + "=" * 80)
    print("CREATING CROSS-COMPARISON PLOTS")
    if survey_filter:
        print(f"Filtered to survey: {survey_filter}")
    print("=" * 80)
    
    # Load enriched instances
    enriched_data = load_enriched_instances_from_cache(cache_dir)
    if not enriched_data:
        print("  No enriched data available for cross-comparisons")
        return
    
    # Load mappings
    script_dir = Path(__file__).resolve().parent
    canonical_path = script_dir / "country_canonical_mapping.json"
    region_path = script_dir / "country_to_region.json"
    
    if not canonical_path.exists():
        print("  Cannot create cross-comparisons (missing mappings)")
        return
    
    canonical = _load_canonical_mapping(canonical_path)
    region_mapping = _load_region_mapping(region_path) if region_path.exists() else {}
    
    # Filter by survey if specified
    all_instances = []
    for instances in enriched_data.values():
        if survey_filter:
            all_instances.extend([inst for inst in instances if inst.survey == survey_filter])
        else:
            all_instances.extend(instances)
    
    if not all_instances:
        print("  No instances available for cross-comparisons")
        return
    
    # Create section × region heatmap
    print("\nCreating section × region comparison...")
    section_region_matrix = defaultdict(lambda: defaultdict(list))
    
    for inst in all_instances:
        if inst.target_section and region_mapping:
            iso2 = _country_iso2(inst, canonical)
            region = region_mapping.get(iso2) or "Unknown"
            if region != "Unknown":
                section_region_matrix[inst.target_section][region].append(inst)
    
    # Calculate accuracy for each section×region combination
    section_region_acc = {}
    for section, regions_dict in section_region_matrix.items():
        section_region_acc[section] = {}
        for region, insts in regions_dict.items():
            if len(insts) >= 50:  # Minimum sample size
                correct = sum(1 for inst in insts if inst.correct)
                section_region_acc[section][region] = correct / len(insts)
    
    if section_region_acc:
        # Create heatmap: Section × Region
        sections_list = sorted(section_region_acc.keys())
        # Use geographic ordering for regions (consistent with region heatmap)
        all_regions_in_data = set(r for sec_dict in section_region_acc.values() for r in sec_dict.keys())
        all_regions_in_data.discard('Unknown')
        regions_list = sort_regions_geographically(all_regions_in_data)
        
        if sections_list and regions_list:
            matrix = np.zeros((len(sections_list), len(regions_list)))
            for i, section in enumerate(sections_list):
                for j, region in enumerate(regions_list):
                    if region in section_region_acc[section]:
                        matrix[i, j] = section_region_acc[section][region]
            
            fig, ax = plt.subplots(figsize=(max(10, len(regions_list) * 0.6), max(6, len(sections_list) * 0.4)))
            # Use RdYlGn: red (low/bad) -> yellow -> green (high/good)
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', 
                          vmin=np.nanmin(matrix[matrix > 0]) if np.any(matrix > 0) else 0.2,
                          vmax=np.nanmax(matrix) if np.any(matrix > 0) else 0.5)
            
            ax.set_xticks(np.arange(len(regions_list)))
            ax.set_xticklabels(regions_list, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(np.arange(len(sections_list)))
            ax.set_yticklabels([s.replace('_', ' ').title() for s in sections_list], fontsize=7)
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Accuracy', fontsize=9)
            cbar.ax.tick_params(width=0.5, labelsize=8)
            
            plt.tight_layout()
            filename_base = f'figure_section_region{output_suffix}'
            plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
            print(f"  [OK] Saved section x region heatmap")
            plt.close(fig)
    
    # Create section × survey comparison
    print("\nCreating section × survey comparison...")
    section_survey_matrix = defaultdict(lambda: defaultdict(list))
    
    for inst in all_instances:
        if inst.target_section:
            section_survey_matrix[inst.target_section][inst.survey].append(inst)
    
    # Calculate accuracy for each section×survey combination
    section_survey_acc = {}
    for section, surveys_dict in section_survey_matrix.items():
        section_survey_acc[section] = {}
        for survey, insts in surveys_dict.items():
            if len(insts) >= 50:
                correct = sum(1 for inst in insts if inst.correct)
                section_survey_acc[section][survey] = correct / len(insts)
    
    if section_survey_acc:
        sections_list = sorted(section_survey_acc.keys())
        surveys_list = sorted(set(s for sec_dict in section_survey_acc.values() for s in sec_dict.keys()))
        
        if sections_list and surveys_list:
            matrix = np.zeros((len(sections_list), len(surveys_list)))
            for i, section in enumerate(sections_list):
                for j, survey in enumerate(surveys_list):
                    if survey in section_survey_acc[section]:
                        matrix[i, j] = section_survey_acc[section][survey]
            
            fig, ax = plt.subplots(figsize=(max(8, len(surveys_list) * 0.8), max(6, len(sections_list) * 0.4)))
            # Use RdYlGn: red (low/bad) -> yellow -> green (high/good)
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                          vmin=np.nanmin(matrix[matrix > 0]) if np.any(matrix > 0) else 0.2,
                          vmax=np.nanmax(matrix) if np.any(matrix > 0) else 0.5)
            
            ax.set_xticks(np.arange(len(surveys_list)))
            ax.set_xticklabels([s.replace('_', ' ').title() for s in surveys_list], 
                              rotation=45, ha='right', fontsize=8)
            ax.set_yticks(np.arange(len(sections_list)))
            ax.set_yticklabels([s.replace('_', ' ').title() for s in sections_list], fontsize=7)
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Accuracy', fontsize=9)
            cbar.ax.tick_params(width=0.5, labelsize=8)
            
            plt.tight_layout()
            filename_base = f'figure_section_survey{output_suffix}'
            plt.savefig(figures_dir / f'{filename_base}.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(figures_dir / f'{filename_base}.png', bbox_inches='tight', dpi=300)
            print(f"  [OK] Saved section x survey heatmap")
            plt.close(fig)

# Create cross-comparison plots
create_cross_comparison_plots(output_suffix=output_suffix, survey_filter=survey_filter)


# =============================================================================
# KEY STATISTICS FOR PAPER
# =============================================================================

print("\n" + "=" * 80)
print("KEY STATISTICS FOR PAPER")
print("=" * 80)

# Geographic gap
europe_regions = ['Southern Europe', 'Northern Europe', 'Western Europe', 'Eastern Europe']
africa_regions = ['West Africa', 'East Africa', 'Central Africa', 'Southern Africa', 'North Africa']

europe_acc = np.mean([region_stats[r]['mean'] for r in europe_regions if r in region_stats])
africa_acc = np.mean([region_stats[r]['mean'] for r in africa_regions if r in region_stats])

print(f"\nEurope average: {europe_acc:.1%}")
print(f"Africa average: {africa_acc:.1%}")
print(f"Europe-Africa gap: {europe_acc - africa_acc:.1%}")

# Topic gap
print(f"\nBest topic section: {best_section[0]} ({best_section[1]['mean']:.1%})")
print(f"Worst topic section: {worst_section[0]} ({worst_section[1]['mean']:.1%})")
print(f"Topic gap: {best_section[1]['mean'] - worst_section[1]['mean']:.1%}")

# Country range
print(f"\nBest country: {sorted_countries[0][0]} ({sorted_countries[0][1]['mean']:.1%})")
print(f"Worst country: {sorted_countries[-1][0]} ({sorted_countries[-1][1]['mean']:.1%})")
print(f"Country range: {sorted_countries[0][1]['mean'] - sorted_countries[-1][1]['mean']:.1%}")

plt.close('all')
print("\nDone!")