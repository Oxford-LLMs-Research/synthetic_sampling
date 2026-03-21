#!/usr/bin/env python3
"""
Shared data cache and loader for analysis scripts.

This module provides:
1. Cached loading of enriched instances (avoids redundant enrichment)
2. Shared data structures across analysis scripts
3. Efficient data access patterns

Usage:
    from shared_data_cache import get_enriched_instances
    
    # Load and cache enriched instances for a model
    instances = get_enriched_instances(
        model_folder=Path("results/llama3.1-8b-instruct"),
        input_paths=[Path("outputs/main_data")],
        cache_dir=Path("analysis/.cache")
    )
"""

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_src_dir))

from synthetic_sampling.evaluation import (
    load_results,
    enrich_instances_with_metadata,
    ParsedInstance,
)


def _compute_cache_key(
    model_name: str,
    profile_filter: Optional[str],
    surveys_filter: Optional[List[str]],
    input_paths: List[Path],
) -> str:
    """Compute a cache key based on model, filters, and input paths."""
    key_parts = [
        model_name,
        str(profile_filter) if profile_filter else "all",
        ",".join(sorted(surveys_filter)) if surveys_filter else "all",
    ]
    # Add input file paths and modification times for cache invalidation
    for path in sorted(input_paths):
        if path.exists():
            if path.is_file():
                key_parts.append(f"{path.name}:{path.stat().st_mtime}")
            elif path.is_dir():
                # Include all JSONL files in directory
                jsonl_files = sorted(path.glob("*.jsonl"))
                for f in jsonl_files:
                    key_parts.append(f"{f.name}:{f.stat().st_mtime}")
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_enriched_cache(cache_file: Path) -> Optional[List[ParsedInstance]]:
    """Load enriched instances from cache file."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        instances = []
        for inst_dict in data:
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
        data = [asdict(inst) for inst in instances]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"    Warning: Could not save cache {cache_file.name}: {e}")


def get_enriched_instances(
    model_folder: Path,
    input_paths: List[Path],
    cache_dir: Path,
    *,
    profile_filter: Optional[str] = None,
    surveys_filter: Optional[List[str]] = None,
    force_reload: bool = False,
    verbose: bool = True,
) -> List[ParsedInstance]:
    """
    Load enriched instances for a model, using cache when available.
    
    Parameters
    ----------
    model_folder : Path
        Folder containing model results JSONL files
    input_paths : List[Path]
        Paths to input data (main_data) for enrichment
    cache_dir : Path
        Directory for cache files
    profile_filter : Optional[str]
        Filter by profile type (e.g., 's6m4')
    surveys_filter : Optional[List[str]]
        Filter by survey names
    force_reload : bool
        If True, ignore cache and reload
    verbose : bool
        Print progress messages
        
    Returns
    -------
    List[ParsedInstance]
        Enriched instances
    """
    model_name = model_folder.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute cache key
    cache_key = _compute_cache_key(model_name, profile_filter, surveys_filter, input_paths)
    cache_file = cache_dir / f"{model_name}_{cache_key}.json"
    
    # Try to load from cache
    if not force_reload:
        cached_instances = _load_enriched_cache(cache_file)
        if cached_instances is not None:
            # Check if cached instances are actually enriched
            # (if input files exist but cache has no enrichment, we should reload)
            if input_paths:
                # Check if any input files exist
                has_input_files = False
                for input_path in input_paths:
                    if input_path.is_file() or (input_path.is_dir() and list(input_path.rglob("*.jsonl"))):
                        has_input_files = True
                        break
                
                # If input files exist but cache has no enrichment, force reload
                if has_input_files and cached_instances:
                    enriched_count = sum(1 for inst in cached_instances if inst.country is not None or inst.target_section is not None)
                    if enriched_count == 0:
                        if verbose:
                            print(f"  [WARNING] Cache exists but has no enrichment (input files now available)")
                            print(f"  (Will reload and re-enrich instances...)")
                        cached_instances = None  # Force reload
            
            if cached_instances is not None:
                if verbose:
                    print(f"  [OK] Loaded {len(cached_instances):,} instances from cache")
                return cached_instances
        elif verbose:
            # Debug: show what cache files exist for this model
            existing_cache_files = list(cache_dir.glob(f"{model_name}_*.json"))
            if existing_cache_files:
                print(f"  [WARNING] Cache file not found: {cache_file.name}")
                print(f"  (Found {len(existing_cache_files)} other cache file(s) for this model)")
                print(f"  (Looking for cache key: {cache_key[:16]}...)")
                print(f"  (Current request: profile_filter={profile_filter if profile_filter else 'all'}, "
                      f"surveys_filter={surveys_filter if surveys_filter else 'all'})")
                print(f"  (Cache key depends on: profile_filter, surveys_filter, and input file timestamps)")
                print(f"  (If these differ from previous run, cache won't match)")
                print(f"  (Will reload and create new cache file...)")
    
    # Load raw instances
    if verbose:
        print(f"  Loading instances from {model_folder.name}...")
        # Show why cache wasn't used
        existing_cache_files = list(cache_dir.glob(f"{model_name}_*.json"))
        if existing_cache_files and not force_reload:
            print(f"  (Note: Found {len(existing_cache_files)} cache file(s) but key didn't match)")
            print(f"  (Current request: profile_filter={profile_filter if profile_filter else 'all'}, "
                  f"surveys_filter={surveys_filter if surveys_filter else 'all'})")
            print(f"  (Cache key: {cache_key[:16]}...)")
            print(f"  (Cache files have different keys - likely different profile_filter or input file timestamps)")
    
    raw_instances = []
    for jsonl_file in sorted(model_folder.glob("*.jsonl")):
        if surveys_filter:
            if not any(s in jsonl_file.name for s in surveys_filter):
                continue
        try:
            batch = load_results(str(jsonl_file))
            for inst in batch:
                if profile_filter is None or inst.profile_type == profile_filter:
                    raw_instances.append(inst)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not load {jsonl_file.name}: {e}")
            continue
    
    if not raw_instances:
        return []
    
    # Enrich with metadata
    if verbose:
        print(f"  Enriching {len(raw_instances):,} instances with metadata...")
    
    # Collect input file paths
    input_file_paths = []
    for input_path in input_paths:
        if input_path.is_file():
            input_file_paths.append(str(input_path))
        elif input_path.is_dir():
            # Search recursively for JSONL files
            jsonl_files = sorted(input_path.rglob("*.jsonl"))
            # Prefer *_instances.jsonl files
            instances_files = [f for f in jsonl_files if "_instances.jsonl" in f.name]
            use_files = instances_files if instances_files else jsonl_files
            if surveys_filter:
                use_files = [f for f in use_files if any(s in f.name for s in surveys_filter)]
            input_file_paths.extend([str(f) for f in use_files])
    
    if not input_file_paths:
        if verbose:
            print(f"    Warning: No input files found for enrichment")
        enriched_instances = raw_instances
    else:
        enriched_instances = enrich_instances_with_metadata(raw_instances, input_file_paths)
    
    # Save to cache
    _save_enriched_cache(enriched_instances, cache_file)
    if verbose:
        print(f"  [OK] Cached {len(enriched_instances):,} enriched instances")
    
    return enriched_instances


def get_all_models_enriched(
    results_dir: Path,
    input_paths: List[Path],
    cache_dir: Path,
    *,
    profile_filter: Optional[str] = None,
    surveys_filter: Optional[List[str]] = None,
    model_whitelist: Optional[List[str]] = None,
    force_reload: bool = False,
    verbose: bool = True,
) -> Dict[str, List[ParsedInstance]]:
    """
    Load enriched instances for all models, using cache.
    
    Returns
    -------
    Dict[str, List[ParsedInstance]]
        Dictionary mapping model_name -> list of enriched instances
    """
    model_folders = [f for f in results_dir.iterdir() if f.is_dir()]
    if model_whitelist:
        model_folders = [f for f in model_folders if f.name in model_whitelist]
    model_folders = sorted(model_folders)
    
    all_instances = {}
    
    for folder in model_folders:
        model_name = folder.name
        if verbose:
            print(f"Loading {model_name}...")
        
        instances = get_enriched_instances(
            model_folder=folder,
            input_paths=input_paths,
            cache_dir=cache_dir,
            profile_filter=profile_filter,
            surveys_filter=surveys_filter,
            force_reload=force_reload,
            verbose=verbose,
        )
        
        if instances:
            all_instances[model_name] = instances
            if verbose:
                enriched_count = sum(1 for inst in instances if inst.country is not None or inst.target_section is not None)
                enrichment_rate = enriched_count / len(instances) if instances else 0
                print(f"  {len(instances):,} instances (enrichment: {enrichment_rate:.1%})")
    
    return all_instances
