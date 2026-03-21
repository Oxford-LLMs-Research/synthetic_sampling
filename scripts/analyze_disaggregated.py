#!/usr/bin/env python3
"""
Disaggregated analysis of LLM survey prediction results.

Produces metrics by:
  1. Survey — per survey (e.g. afrobarometer, ess_wave_10)
  2. Topic (thematic section) — coarser
  3. Topic tag — finer-grained
  4. Country — ISO 3166-1 alpha-2, consistent across all surveys
  5. Region — optional aggregation via country_to_region.json (ISO-2 -> region)

Uses country_canonical_mapping.json to map every survey-specific raw country value
(names, numeric codes) to ISO-2. Same country in multiple surveys maps to the same
ISO-2. Region file uses ISO-2 keys only.

Requires input data (main_data) with example_id, country, target_section, target_topic_tag.
Join is on example_id.

IMPORTANT: Use the SAME main_data version that was used to generate the results.
If enrichment rate is low (<80%), check that --inputs points to the matching main_data.
For results generated on 2026-01-20, use: outputs/main_data_smaller_20_jan_26/main_data

Usage:
  python analyze_disaggregated.py --results-dir results --inputs outputs/main_data \\
      --output analysis/disaggregated

  python analyze_disaggregated.py --results-dir results --inputs outputs/main_data \\
      --output out --region-mapping scripts/country_to_region.json

  python analyze_disaggregated.py ... --surveys afrobarometer --models llama3.1-8b-instruct

  python analyze_disaggregated.py ... --by-survey-only   # only by_survey.json (faster)

Inputs: directory of *_instances.jsonl (main_data). Results: model folders with *_results.jsonl.

Caching:
  Enriched instances are cached in {output_dir}/.cache/enriched/ to speed up subsequent runs.
  Cache files are named {model_name}_{cache_key}.json where cache_key is an MD5 hash
  based on model name, filters, and input file modification times.
  
  To load cached enriched data in another script:
    from pathlib import Path
    from analyze_disaggregated import load_enriched_instances_from_cache
    
    cache_dir = Path("analysis/disaggregated/.cache/enriched")
    enriched_data = load_enriched_instances_from_cache(cache_dir)
    # enriched_data is a dict: {model_name: [ParsedInstance, ...]}
    llama_instances = enriched_data.get("llama3.1-8b-instruct", [])
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_src_dir))

from synthetic_sampling.evaluation import (
    load_results,
    enrich_instances_with_metadata,
    ResultsAnalyzer,
    ParsedInstance,
    compute_instance_metrics,
    compute_distribution_metrics,
)
import numpy as np
from shared_data_cache import get_enriched_instances


# =============================================================================
# Helpers
# =============================================================================


def find_jsonl_files(directory: Path, pattern: str = "*.jsonl") -> List[Path]:
    if directory.is_file():
        return [directory] if directory.suffix == ".jsonl" else []
    return sorted(directory.glob(pattern))


def load_model_results(
    model_folder: Path,
    profile_filter: Optional[str],
    surveys_filter: Optional[List[str]] = None,
) -> List[ParsedInstance]:
    """Load JSONL from a model folder; filter by profile and optionally by survey."""
    instances: List[ParsedInstance] = []
    for p in find_jsonl_files(model_folder):
        if surveys_filter:
            if not any(s in p.name for s in surveys_filter):
                continue
        try:
            batch = load_results(str(p))
        except Exception as e:
            print(f"    Warning: Could not load {p.name}: {e}")
            continue
        for inst in batch:
            if profile_filter is None or inst.profile_type == profile_filter:
                instances.append(inst)
    return instances


def collect_input_paths(
    inputs_path: Path,
    surveys_filter: Optional[List[str]] = None,
) -> List[str]:
    """Resolve input JSONL paths (main_data). Prefer *_instances.jsonl."""
    if not inputs_path.exists():
        return []
    files = find_jsonl_files(inputs_path)
    instances = [f for f in files if "_instances.jsonl" in f.name]
    use = instances if instances else files
    if surveys_filter:
        use = [f for f in use if any(s in f.name for s in surveys_filter)]
    return [str(p) for p in use]


def metrics_to_serializable(obj: Any) -> Any:
    """Convert metrics structures to JSON-serializable form."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: metrics_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [metrics_to_serializable(x) for x in obj]
    return obj


def _normalize_raw(raw: Optional[str]) -> str:
    """Strip and remove trailing .0 from numeric strings."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _to_canonical_iso2(
    raw: Optional[str],
    survey: str,
    canonical: Dict[str, Any],
) -> Optional[str]:
    """
    Map survey-specific raw country value to ISO 3166-1 alpha-2.
    Uses country_canonical_mapping.json: by_survey first, then iso_numeric for
    wvs/latinobarometer, ESS pass-through for 2-letter codes.
    """
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


def _country_iso2(inst: ParsedInstance, canonical: Dict[str, Any]) -> str:
    """Resolve ISO-2 for an instance. Fallback: ESS-style respondent_id prefix or UNKNOWN."""
    iso2 = _to_canonical_iso2(inst.country, inst.survey, canonical)
    if iso2:
        return iso2
    rid = str(inst.respondent_id)
    if len(rid) >= 2 and rid[:2].isalpha():
        if len(rid) == 2 or (len(rid) > 2 and (rid[2] in "_" or rid[2].isdigit())):
            return rid[:2].upper()
    return "UNKNOWN"


def _compute_distribution_metrics_for_group(
    instances: List[ParsedInstance],
    min_n_per_target: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Compute distribution metrics (variance ratio, JS divergence) for a group of instances.
    
    Groups instances by target question, computes distribution metrics for each target,
    then aggregates (mean/median) across targets.
    
    Parameters
    ----------
    instances : list[ParsedInstance]
        Instances to analyze
    min_n_per_target : int
        Minimum instances per target question for inclusion (default: 10, lowered for smaller groups)
        
    Returns
    -------
    dict
        Dictionary with variance_ratio_soft, variance_ratio_hard, 
        js_divergence_soft, js_divergence_hard (mean and median)
    """
    if not instances:
        return {
            'variance_ratio_soft_mean': None,
            'variance_ratio_soft_median': None,
            'variance_ratio_hard_mean': None,
            'variance_ratio_hard_median': None,
            'js_divergence_soft_mean': None,
            'js_divergence_soft_median': None,
            'js_divergence_hard_mean': None,
            'js_divergence_hard_median': None,
        }
    
    # Group by target question
    by_target: Dict[str, List[ParsedInstance]] = defaultdict(list)
    for inst in instances:
        if inst.target_code:  # Only include instances with target_code
            by_target[inst.target_code].append(inst)
    
    # If we have very few instances total, use a lower threshold
    # Adaptive threshold: use 5 if total instances < 200, otherwise use min_n_per_target
    adaptive_threshold = 5 if len(instances) < 200 else min_n_per_target
    
    # Compute distribution metrics for each target
    variance_ratios_soft = []
    variance_ratios_hard = []
    js_divergences_soft = []
    js_divergences_hard = []
    
    for target, target_insts in by_target.items():
        # Use adaptive threshold
        if len(target_insts) < adaptive_threshold:
            continue
        
        # Get options from first instance - must have options
        if not target_insts[0].options or len(target_insts[0].options) < 2:
            continue
        
        options = target_insts[0].options
        metrics = compute_distribution_metrics(target_insts, options)
        
        # Variance ratio can be None if empirical variance is too small (all same answer)
        # But JS divergence should always be computed
        if metrics.get('variance_ratio_soft') is not None:
            variance_ratios_soft.append(metrics['variance_ratio_soft'])
        if metrics.get('variance_ratio_hard') is not None:
            variance_ratios_hard.append(metrics['variance_ratio_hard'])
        
        # JS divergence should always be available if we got here
        if metrics.get('js_divergence_soft') is not None:
            js_divergences_soft.append(metrics['js_divergence_soft'])
        if metrics.get('js_divergence_hard') is not None:
            js_divergences_hard.append(metrics['js_divergence_hard'])
    
    # Aggregate across targets
    def safe_mean(arr):
        return float(np.mean(arr)) if arr else None
    
    def safe_median(arr):
        return float(np.median(arr)) if arr else None
    
    return {
        'variance_ratio_soft_mean': safe_mean(variance_ratios_soft),
        'variance_ratio_soft_median': safe_median(variance_ratios_soft),
        'variance_ratio_hard_mean': safe_mean(variance_ratios_hard),
        'variance_ratio_hard_median': safe_median(variance_ratios_hard),
        'js_divergence_soft_mean': safe_mean(js_divergences_soft),
        'js_divergence_soft_median': safe_median(js_divergences_soft),
        'js_divergence_hard_mean': safe_mean(js_divergences_hard),
        'js_divergence_hard_median': safe_median(js_divergences_hard),
    }


def _metrics_by_country_canonical(
    instances: List[ParsedInstance],
    canonical: Dict[str, Any],
    min_n: int,
) -> Dict[str, Any]:
    """Group instances by canonical ISO-2 and compute metrics."""
    by_country: Dict[str, List[ParsedInstance]] = defaultdict(list)
    for inst in instances:
        iso2 = _country_iso2(inst, canonical)
        by_country[iso2].append(inst)
    
    result = {}
    for k, v in sorted(by_country.items()):
        if len(v) >= min_n:
            metrics = compute_instance_metrics(v).to_dict()
            # Add distribution metrics
            dist_metrics = _compute_distribution_metrics_for_group(v)
            metrics.update(dist_metrics)
            result[k] = metrics
    return result


def _metrics_by_region(
    instances: List[ParsedInstance],
    canonical: Dict[str, Any],
    region_mapping: Dict[str, str],
    min_n: int,
) -> Dict[str, Any]:
    """Group instances by region (ISO-2 -> region via mapping) and compute metrics."""
    by_region: Dict[str, List[ParsedInstance]] = defaultdict(list)
    for inst in instances:
        iso2 = _country_iso2(inst, canonical)
        region = region_mapping.get(iso2) or "Unknown"
        by_region[region].append(inst)
    
    result = {}
    for k, v in sorted(by_region.items()):
        if len(v) >= min_n:
            metrics = compute_instance_metrics(v).to_dict()
            # Add distribution metrics
            dist_metrics = _compute_distribution_metrics_for_group(v)
            metrics.update(dist_metrics)
            result[k] = metrics
    return result


def _load_canonical_mapping(path: Path) -> Dict[str, Any]:
    """Load country_canonical_mapping.json. Exclude keys starting with _."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_region_mapping(path: Path) -> Dict[str, str]:
    """Load country_to_region.json (ISO-2 -> region). Exclude keys starting with _."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: str(v) for k, v in data.items() if not k.startswith("_")}


def _load_metadata_for_survey(survey: str, metadata_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load metadata file for a given survey.
    
    Parameters
    ----------
    survey : str
        Survey name (e.g., 'wvs', 'latinobarometer', 'ess_wave_10')
    metadata_dir : Path
        Base directory containing metadata files
    
    Returns
    -------
    dict or None
        Metadata dictionary, or None if not found
    """
    from synthetic_sampling.config.surveys import get_survey_config
    
    try:
        survey_config = get_survey_config(survey)
        metadata_path = metadata_dir / survey_config.metadata_path
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    
    return None


def _normalize_question_text(text: str) -> str:
    """
    Normalize question text for flexible matching.
    Removes variations like "currently", extra whitespace, etc.
    
    Parameters
    ----------
    text : str
        Question text
    
    Returns
    -------
    str
        Normalized question text
    """
    # Lowercase and strip
    text = text.lower().strip()
    # Remove "currently" for flexible matching (ESS 10 vs 11)
    text = text.replace(" currently ", " ").replace("currently ", "").replace(" currently", "")
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def get_country_region_question_wordings(survey: str, metadata_dir: Path) -> Dict[str, List[str]]:
    """
    Get question wordings for country and region variables from metadata.
    
    Parameters
    ----------
    survey : str
        Survey name (e.g., 'wvs', 'latinobarometer', 'ess_wave_10')
    metadata_dir : Path
        Base directory containing metadata files
    
    Returns
    -------
    dict
        Dictionary with keys 'country' and 'region', each containing a list of question wordings
        (normalized and original variations)
    """
    metadata = _load_metadata_for_survey(survey, metadata_dir)
    if not metadata:
        return {'country': [], 'region': []}
    
    # Variable code mappings for country and region
    country_var_codes = {
        'wvs': ['B_COUNTRY'],
        'latinobarometer': ['IDENPA'],
        'afrobarometer': ['COUNTRY'],
        'arabbarometer': ['COUNTRY', 'Q1'],  # Q1 is Governorate (country-like)
        'asianbarometer': ['country'],
        'ess_wave_10': ['cntry'],
        'ess_wave_11': ['cntry'],
    }
    
    region_var_codes = {
        'wvs': ['N_REGION_ISO'],
        'latinobarometer': ['REG'],
        'afrobarometer': ['REGION'],  # Uppercase REGION in metadata: "Which region/province/state do you live in?"
        'arabbarometer': ['Q1'],  # Q1 is Governorate: "In which governorate do you live?"
        'asianbarometer': ['region'],  # Lowercase "region" in metadata: "What is your administrative region?"
        'ess_wave_10': ['region'],  # ESS Wave 10 "region": "In which region or country do you live?"
        'ess_wave_11': ['region'],  # ESS Wave 11 "region" variable
    }
    
    survey_lower = survey.lower()
    
    # Get variable codes for this survey
    # Try exact match first, then partial match
    country_codes = []
    if survey_lower in country_var_codes:
        country_codes = country_var_codes[survey_lower]
    else:
        # Try partial matches (e.g., 'ess_wave_10' contains 'ess')
        for key, codes in country_var_codes.items():
            if key in survey_lower or survey_lower in key:
                country_codes = codes
                break
    
    region_codes = []
    if survey_lower in region_var_codes:
        region_codes = region_var_codes[survey_lower]
    else:
        # Try partial matches
        for key, codes in region_var_codes.items():
            if key in survey_lower or survey_lower in key:
                region_codes = codes
                break
    
    # Extract question wordings from metadata
    country_questions = []
    region_questions = []
    
    def find_question_for_var(var_code: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Find question wording for a variable code in metadata."""
        for section_name, section_data in metadata.items():
            if not isinstance(section_data, dict):
                continue
            if var_code in section_data:
                var_info = section_data[var_code]
                if isinstance(var_info, dict):
                    question = var_info.get('question') or var_info.get('description', '')
                    if question:
                        return question
        return None
    
    # Get country question wordings
    for var_code in country_codes:
        question = find_question_for_var(var_code, metadata)
        if question:
            country_questions.append(question)
            # Also add normalized version for flexible matching
            normalized = _normalize_question_text(question)
            if normalized != question.lower().strip():
                country_questions.append(normalized)
    
    # Get region question wordings
    for var_code in region_codes:
        question = find_question_for_var(var_code, metadata)
        if question:
            region_questions.append(question)
            # Also add normalized version for flexible matching
            normalized = _normalize_question_text(question)
            if normalized != question.lower().strip():
                region_questions.append(normalized)
    
    # Remove duplicates while preserving order
    country_questions = list(dict.fromkeys(country_questions))
    region_questions = list(dict.fromkeys(region_questions))
    
    return {
        'country': country_questions,
        'region': region_questions
    }


def load_input_data_with_profile_features(input_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load input JSONL files and extract profile features (questions) for each example.
    
    Parameters
    ----------
    input_paths : list[str]
        Paths to input JSONL files
    
    Returns
    -------
    dict
        Mapping of example_id -> {'questions': dict, 'country': str, ...}
        where 'questions' contains the profile features
    """
    input_data = {}
    
    for path_str in input_paths:
        path = Path(path_str)
        if not path.exists():
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                example_id = data.get('example_id')
                if example_id:
                    input_data[example_id] = {
                        'questions': data.get('questions', {}),  # Profile features
                        'country': data.get('country'),
                        'target_section': data.get('target_section'),
                        'target_topic_tag': data.get('target_topic_tag'),
                        'target_response_format': data.get('target_response_format'),
                    }
    
    return input_data


def filter_instances_by_country_in_profile(
    instances: List[ParsedInstance],
    input_data: Dict[str, Dict[str, Any]],
    metadata_dir: Optional[Path] = None,
) -> List[ParsedInstance]:
    """
    Filter instances to only those where country or region variable is in profile features.
    Uses question wordings from metadata files for flexible matching.
    
    Parameters
    ----------
    instances : list[ParsedInstance]
        List of instances to filter
    input_data : dict
        Mapping of example_id -> input data with 'questions' field
    metadata_dir : Path, optional
        Base directory containing metadata files. If None, uses default path.
    
    Returns
    -------
    list[ParsedInstance]
        Filtered instances where country or region is in profile features
    """
    if metadata_dir is None:
        metadata_dir = _script_dir.parent / "src" / "synthetic_sampling" / "profiles" / "metadata"
    
    filtered = []
    survey_counts = defaultdict(int)
    
    # Cache question wordings per survey
    survey_question_wordings = {}
    
    def matches_question_wording(question_text: str, target_wordings: List[str]) -> bool:
        """
        Check if a question text matches any of the target wordings (with flexible matching).
        
        Parameters
        ----------
        question_text : str
            Question text from profile features
        target_wordings : list[str]
            List of target question wordings to match against
        
        Returns
        -------
        bool
            True if question_text matches any target wording
        """
        if not question_text or not target_wordings:
            return False
        
        # Normalize both the question and all targets
        normalized_question = _normalize_question_text(question_text)
        question_lower = question_text.lower().strip()
        
        for target in target_wordings:
            if not target:
                continue
            target_lower = target.lower().strip()
            normalized_target = _normalize_question_text(target)
            
            # Exact match (case-insensitive)
            if question_lower == target_lower:
                return True
            
            # Normalized match (handles "currently" variations, whitespace, etc.)
            if normalized_question == normalized_target:
                return True
            
            # Substring match (for variations in wording)
            # Check if one is contained in the other (after normalization)
            # This handles cases like "In which region or country do you live?" matching
            # both region and country targets
            if normalized_target in normalized_question or normalized_question in normalized_target:
                return True
        
        return False
    
    def has_country_or_region_in_questions(
        questions: Dict[str, Any],
        country_wordings: List[str],
        region_wordings: List[str],
    ) -> bool:
        """
        Check if country or region question exists in questions dict.
        Questions dict uses question wordings as keys, not variable codes.
        
        Parameters
        ----------
        questions : dict
            Profile features dict (question_text -> answer)
        country_wordings : list[str]
            List of country question wordings to match
        region_wordings : list[str]
            List of region question wordings to match
        
        Returns
        -------
        bool
            True if any country or region question is found
        """
        if not questions:
            return False
        
        # Questions dict is flat: {question_text: answer}
        for question_text in questions.keys():
            if matches_question_wording(question_text, country_wordings):
                return True
            if matches_question_wording(question_text, region_wordings):
                return True
        
        return False
    
    for inst in instances:
        input_info = input_data.get(inst.example_id)
        if not input_info:
            continue
        
        questions = input_info.get('questions', {})
        if not questions:
            continue
        
        # Get question wordings for this survey (with caching)
        if inst.survey not in survey_question_wordings:
            wordings = get_country_region_question_wordings(inst.survey, metadata_dir)
            survey_question_wordings[inst.survey] = wordings
            # Debug: print found questions for first instance of each survey
            country_wordings = wordings.get('country', [])
            region_wordings = wordings.get('region', [])
            if country_wordings or region_wordings:
                print(f"  [{inst.survey}] Found questions:")
                if country_wordings:
                    print(f"    Country: {country_wordings}")
                if region_wordings:
                    print(f"    Region: {region_wordings}")
        else:
            wordings = survey_question_wordings[inst.survey]
        
        country_wordings = wordings.get('country', [])
        region_wordings = wordings.get('region', [])
        
        if has_country_or_region_in_questions(questions, country_wordings, region_wordings):
            filtered.append(inst)
            survey_counts[inst.survey] += 1
    
    print(f"\nFiltered to instances with country or region in profile features:")
    for survey, count in sorted(survey_counts.items()):
        print(f"  {survey}: {count:,} instances")
    
    return filtered


def _compute_cache_key(
    model_name: str,
    profile_filter: Optional[str],
    surveys_filter: Optional[List[str]],
    input_paths: List[str],
) -> str:
    """Compute a cache key based on model, filters, and input paths."""
    # Include model name, profile filter, surveys filter, and input file paths/timestamps
    key_parts = [
        model_name,
        str(profile_filter) if profile_filter else "all",
        ",".join(sorted(surveys_filter)) if surveys_filter else "all",
    ]
    # Add input file paths and modification times for cache invalidation
    for path_str in sorted(input_paths):
        path = Path(path_str)
        if path.exists():
            key_parts.append(f"{path.name}:{path.stat().st_mtime}")
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_enriched_cache(cache_file: Path) -> Optional[List[ParsedInstance]]:
    """Load enriched instances from cache file."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruct ParsedInstance objects from dicts
        instances = []
        for inst_dict in data:
            # Create ParsedInstance from dict (dataclass can be instantiated with **dict)
            inst = ParsedInstance(**inst_dict)
            instances.append(inst)
        return instances
    except Exception as e:
        print(f"    Warning: Could not load cache {cache_file.name}: {e}")
        return None


def _save_enriched_cache(instances: List[ParsedInstance], cache_file: Path) -> None:
    """Save enriched instances to cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Use dataclasses.asdict() to serialize ParsedInstance objects
        data = [asdict(inst) for inst in instances]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"    Warning: Could not save cache {cache_file.name}: {e}")


def load_enriched_instances_from_cache(
    cache_dir: Path,
    model_name: Optional[str] = None,
) -> Dict[str, List[ParsedInstance]]:
    """
    Load enriched instances from cache directory.
    
    This function allows you to access cached enriched data from other scripts.
    
    Parameters
    ----------
    cache_dir : Path
        Path to the cache directory (typically output_dir / ".cache" / "enriched")
    model_name : Optional[str]
        If provided, only load cache files for this model. If None, load all.
        
    Returns
    -------
    Dict[str, List[ParsedInstance]]
        Dictionary mapping model names to lists of enriched ParsedInstance objects
        
    Example
    -------
    >>> from pathlib import Path
    >>> from analyze_disaggregated import load_enriched_instances_from_cache
    >>> cache_dir = Path("analysis/disaggregated/.cache/enriched")
    >>> enriched_data = load_enriched_instances_from_cache(cache_dir)
    >>> # Access instances for a specific model
    >>> llama_instances = enriched_data.get("llama3.1-8b-instruct", [])
    """
    if not cache_dir.exists():
        return {}
    
    enriched_data = {}
    pattern = f"{model_name}_*.json" if model_name else "*.json"
    
    for cache_file in cache_dir.glob(pattern):
        # Extract model name from filename (format: {model_name}_{cache_key}.json)
        filename = cache_file.stem  # filename without .json
        if "_" in filename:
            # Find the last underscore (cache keys are MD5 hashes, no underscores)
            # Model names might have underscores, so we need to be smarter
            # Try to find where the cache key starts (32-char hex string)
            parts = filename.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 32:  # MD5 hash is 32 chars
                model = parts[0]
            else:
                # Fallback: use everything before last underscore
                model = parts[0]
        else:
            continue
            
        instances = _load_enriched_cache(cache_file)
        if instances is not None:
            if model not in enriched_data:
                enriched_data[model] = []
            enriched_data[model].extend(instances)
    
    return enriched_data


# =============================================================================
# Main analysis
# =============================================================================


def run_disaggregated(
    results_dir: Path,
    inputs_path: Path,
    output_dir: Path,
    *,
    profile_filter: Optional[str] = "s6m4",
    min_n: int = 100,
    min_n_tag: int = 50,
    by_topic_only: bool = False,
    by_country_only: bool = False,
    by_survey_only: bool = False,
    model_whitelist: Optional[List[str]] = None,
    surveys_filter: Optional[List[str]] = None,
    canonical_mapping_path: Path,
    region_mapping_path: Optional[Path] = None,
    country_in_profile_only: bool = False,
) -> None:
    only_flags = sum([by_topic_only, by_country_only, by_survey_only])
    assert only_flags <= 1, "Use at most one of --by-topic-only, --by-country-only, --by-survey-only"

    canonical: Dict[str, Any] = {}
    region_mapping: Dict[str, str] = {}
    if not by_survey_only:
        if not canonical_mapping_path.exists():
            print(f"Error: Canonical mapping not found: {canonical_mapping_path}")
            sys.exit(1)
        canonical = _load_canonical_mapping(canonical_mapping_path)
        print(f"Canonical mapping: by_survey {list((canonical.get('by_survey') or {}).keys())}, iso_numeric {len(canonical.get('iso_numeric') or {})} codes")
        if region_mapping_path and region_mapping_path.exists():
            region_mapping = _load_region_mapping(region_mapping_path)
            print(f"Region mapping: {len(region_mapping)} ISO-2 -> region entries")
    else:
        print("Mode: by-survey-only (skipping section, tag, country, region)")

    input_paths = collect_input_paths(inputs_path, surveys_filter)
    if not input_paths:
        print("Error: No input JSONL found. Use --inputs pointing to main_data (dir of *_instances.jsonl).")
        sys.exit(1)
    print(f"Input metadata: {len(input_paths)} file(s)")
    if surveys_filter:
        print(f"Surveys filter: {surveys_filter}")
    
    # Load input data with profile features if filtering by country in profile
    input_data_with_features = {}
    if country_in_profile_only:
        print("Loading input data with profile features for country filtering...")
        input_data_with_features = load_input_data_with_profile_features([str(p) for p in input_paths])
        print(f"Loaded profile features for {len(input_data_with_features):,} examples")

    model_folders = [f for f in results_dir.iterdir() if f.is_dir()]
    if model_whitelist:
        model_folders = [f for f in model_folders if f.name in model_whitelist]
    model_folders = sorted(model_folders)
    if not model_folders:
        print("Error: No model folders found in results dir.")
        sys.exit(1)
    print(f"Models: {[f.name for f in model_folders]}")
    if profile_filter:
        print(f"Profile filter: {profile_filter}")
    if country_in_profile_only:
        print("Filter: Only instances where country is in profile features")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up cache directory
    # Use shared cache directory so all analysis scripts can reuse the same cache
    # Default to shared cache location (analysis/.cache/enriched)
    # This allows profile_richness, profile_richness_by_topic, and disaggregated to share cache
    cache_dir = Path("analysis/.cache/enriched")
    cache_dir.mkdir(parents=True, exist_ok=True)

    by_survey: Dict[str, Dict[str, Any]] = {}
    by_section: Dict[str, Dict[str, Any]] = {}
    by_topic_tag: Dict[str, Dict[str, Any]] = {}
    by_country: Dict[str, Dict[str, Any]] = {}
    by_region: Dict[str, Dict[str, Any]] = {}

    for folder in model_folders:
        model_name = folder.name
        print(f"Loading {model_name}...")
        
        # Use shared cache system (loads, enriches, and caches in one step)
        instances = get_enriched_instances(
            model_folder=folder,
            input_paths=[Path(p) for p in input_paths],
            cache_dir=cache_dir,
            profile_filter=profile_filter,
            surveys_filter=surveys_filter,
            force_reload=False,
            verbose=True,
        )
        
        if not instances:
            print(f"  No instances (profile={profile_filter}), skip.")
            continue
        
        # Filter by country in profile if requested
        # NOTE: This filtering only affects which instances are included in the analysis.
        # The main disaggregated analysis (by_country, by_region, etc.) uses inst.country
        # from enrichment metadata, NOT from profile features. So the main analysis logic
        # works the same way regardless of this filter - it handles all instances that
        # have country information from enrichment, whether or not country appears in
        # the profile features.
        if country_in_profile_only:
            instances_before = len(instances)
            metadata_dir = _script_dir.parent / "src" / "synthetic_sampling" / "profiles" / "metadata"
            instances = filter_instances_by_country_in_profile(
                instances, 
                input_data_with_features,
                metadata_dir=metadata_dir
            )
            if not instances:
                print(f"  No instances with country in profile features, skip.")
                continue
            print(f"  Filtered: {instances_before:,} -> {len(instances):,} instances (country/region in profile)")
        
        # Check enrichment rate and warn if low
        enriched_count = sum(1 for inst in instances if inst.country is not None or inst.target_section is not None)
        enrichment_rate = enriched_count / len(instances) if instances else 0
        if enrichment_rate < 0.8:
            print(f"  ⚠ WARNING: Low enrichment rate ({enrichment_rate:.1%}).")
            print(f"     This suggests main_data version mismatch with results.")
            print(f"     Results were generated from a specific main_data version.")
            print(f"     Ensure --inputs points to the matching main_data (e.g., main_data_smaller_20_jan_26).")
        
        analyzer = ResultsAnalyzer(instances)

        n_total = len(instances)
        overall = analyzer.overall_metrics()
        print(f"  n={n_total:,} acc={overall.accuracy:.1%}")

        # By survey (always)
        by_surv = analyzer.metrics_by_survey()
        # Add distribution metrics for each survey
        by_survey_dict = {}
        for survey_name, metrics_summary in by_surv.items():
            survey_instances = [inst for inst in instances if inst.survey == survey_name]
            metrics_dict = metrics_summary.to_dict()
            dist_metrics = _compute_distribution_metrics_for_group(survey_instances)
            metrics_dict.update(dist_metrics)
            by_survey_dict[survey_name] = metrics_dict
        by_survey[model_name] = by_survey_dict

        if not by_survey_only:
            if not by_country_only:
                by_sec = analyzer.metrics_by_section(min_n=min_n)
                # Add distribution metrics for each section
                by_section_dict = {}
                for section, metrics_summary in by_sec.items():
                    section_instances = [inst for inst in instances if inst.target_section == section]
                    metrics_dict = metrics_summary.to_dict()
                    dist_metrics = _compute_distribution_metrics_for_group(section_instances)
                    metrics_dict.update(dist_metrics)
                    by_section_dict[section] = metrics_dict
                by_section[model_name] = by_section_dict

                by_tag = analyzer.metrics_by_topic_tag(min_n=min_n_tag)
                # Add distribution metrics for each topic tag
                by_topic_tag_dict = {}
                for tag, metrics_summary in by_tag.items():
                    tag_instances = [inst for inst in instances if inst.target_topic_tag == tag]
                    metrics_dict = metrics_summary.to_dict()
                    dist_metrics = _compute_distribution_metrics_for_group(tag_instances)
                    metrics_dict.update(dist_metrics)
                    by_topic_tag_dict[tag] = metrics_dict
                by_topic_tag[model_name] = by_topic_tag_dict

            if not by_topic_only:
                by_cnt = _metrics_by_country_canonical(instances, canonical, min_n)
                by_country[model_name] = by_cnt
            if region_mapping and not by_topic_only:
                by_reg = _metrics_by_region(instances, canonical, region_mapping, min_n)
                by_region[model_name] = by_reg

    # Determine output file suffix based on filters
    output_suffix = ""
    if country_in_profile_only:
        output_suffix = "_country_in_profile"
    
    # Write outputs
    print(f"\n{'='*80}")
    print(f"SUMMARY: Processed {len(by_survey)} model(s): {list(by_survey.keys())}")
    if by_section and not by_survey_only:
        print(f"  by_section: {len(by_section)} model(s): {list(by_section.keys())}")
    if by_country and not by_survey_only:
        print(f"  by_country: {len(by_country)} model(s): {list(by_country.keys())}")
    if by_region and not by_survey_only:
        print(f"  by_region: {len(by_region)} model(s): {list(by_region.keys())}")
    print(f"{'='*80}\n")
    
    if by_survey:
        path = output_dir / f"by_survey{output_suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_to_serializable(by_survey), f, indent=2)
        print(f"\nSaved {path}")
    if by_section and not by_survey_only:
        path = output_dir / f"by_section{output_suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_to_serializable(by_section), f, indent=2)
        print(f"Saved {path}")
    if by_topic_tag and not by_survey_only:
        path = output_dir / f"by_topic_tag{output_suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_to_serializable(by_topic_tag), f, indent=2)
        print(f"Saved {path}")
    if by_country and not by_survey_only:
        path = output_dir / f"by_country{output_suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_to_serializable(by_country), f, indent=2)
        print(f"Saved {path}")
    if by_region and not by_survey_only:
        path = output_dir / f"by_region{output_suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_to_serializable(by_region), f, indent=2)
        print(f"Saved {path}")

    # Summary tables to stdout (use ASCII dash to avoid encoding issues)
    def _print_table(
        title: str,
        data: Dict[str, Dict[str, Any]],
        sort_key: str = "n",
        top_n: int = 25,
    ) -> None:
        if not data:
            return
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        first_model = next(iter(data))
        groups = list(data[first_model].keys())
        for group in sorted(groups, key=lambda g: -data[first_model][g].get("n", 0))[:top_n]:
            row = [f"  {group:<36}"]
            for m in list(data.keys())[:5]:
                sm = data[m].get(group)
                if sm:
                    row.append(f"n={sm['n']:,} acc={sm['accuracy']:.1%}")
                else:
                    row.append("-")
            print(" ".join(row))
        if len(data) > 5:
            print("  ... (further models in JSON)")

    if by_survey:
        _print_table("BY SURVEY - first 5 models", by_survey, top_n=20)
    if by_section and not by_survey_only:
        _print_table("BY SECTION (thematic) - first 5 models", by_section)
    if by_topic_tag and not by_survey_only:
        _print_table("BY TOPIC TAG - first 5 models", by_topic_tag, top_n=20)
    if by_country and not by_survey_only:
        _print_table("BY COUNTRY - first 5 models", by_country, top_n=20)
    if by_region and not by_survey_only:
        _print_table("BY REGION - first 5 models", by_region, top_n=15)

    print("\nDone.")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Disaggregated analysis: by survey, topic (section, tag), and country.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--results-dir", type=Path, required=True, help="Results root (model folders with JSONL)")
    ap.add_argument("--inputs", type=Path, required=True, help="Main data dir (e.g. outputs/main_data) with *_instances.jsonl")
    ap.add_argument("--output", type=Path, default=Path("analysis/disaggregated"), help="Output directory")
    ap.add_argument("--profile", type=str, default="s6m4", choices=["s3m2", "s4m3", "s6m4", "all"],
                    help="Profile filter (default: s6m4). Use 'all' for no filter.")
    ap.add_argument("--min-n", type=int, default=100, help="Min instances for section/country (default 100)")
    ap.add_argument("--min-n-tag", type=int, default=50, help="Min instances for topic tag (default 50)")
    ap.add_argument("--by-topic-only", action="store_true", help="Only by section and topic tag")
    ap.add_argument("--by-country-only", action="store_true", help="Only by country")
    ap.add_argument("--by-survey-only", action="store_true", help="Only by survey (by_survey.json); skip section, tag, country, region")
    ap.add_argument("--models", type=str, nargs="*", help="Restrict to these model folder names")
    ap.add_argument("--surveys", type=str, nargs="*", help="Restrict to these surveys (e.g. afrobarometer ess_wave_10) for faster runs")
    ap.add_argument("--country-in-profile-only", action="store_true", 
                    help="Only include instances where country variable is in profile features")
    ap.add_argument("--canonical-mapping", type=Path, default=None,
                    help="JSON file mapping raw country -> ISO-2 per survey (default: scripts/country_canonical_mapping.json)")
    ap.add_argument("--region-mapping", type=Path, default=None,
                    help="JSON file mapping ISO-2 -> region for by_region output (default: scripts/country_to_region.json)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    canonical_path = args.canonical_mapping or script_dir / "country_canonical_mapping.json"
    region_path = args.region_mapping or script_dir / "country_to_region.json"

    profile = None if args.profile == "all" else args.profile
    run_disaggregated(
        results_dir=args.results_dir,
        inputs_path=args.inputs,
        output_dir=args.output,
        profile_filter=profile,
        min_n=args.min_n,
        min_n_tag=args.min_n_tag,
        by_topic_only=args.by_topic_only,
        by_country_only=args.by_country_only,
        by_survey_only=args.by_survey_only,
        model_whitelist=args.models,
        surveys_filter=args.surveys,
        canonical_mapping_path=canonical_path,
        region_mapping_path=region_path,
        country_in_profile_only=args.country_in_profile_only,
    )


if __name__ == "__main__":
    main()
