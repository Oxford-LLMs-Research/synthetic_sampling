"""
Survey data loading module.

This module provides the SurveyLoader class which handles loading survey data
and metadata from local directories, applying any necessary preprocessing.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from ..config.base import DataPaths
from ..config.surveys import SurveyConfig, SURVEY_REGISTRY, get_survey_config
from .file_io import find_data_files, load_file, load_multiple_files


class SurveyLoader:
    """
    Loads survey data and metadata from local directories.
    
    This class handles:
    - Finding and loading survey data files (CSV, DTA, SAV)
    - Loading corresponding metadata JSON files
    - Applying survey-specific preprocessing (e.g., ID column construction)
    - Validation of loaded data
    
    Example:
        >>> from synthetic_sampling.config import DataPaths
        >>> paths = DataPaths(
        ...     raw_data_dir='~/data/surveys',
        ...     metadata_dir='./src/synthetic_sampling/profiles/metadata',
        ...     output_dir='./outputs'
        ... )
        >>> loader = SurveyLoader(paths)
        >>> 
        >>> # Load a single survey
        >>> df, metadata = loader.load_survey('wvs')
        >>> 
        >>> # Load multiple surveys
        >>> all_data = loader.load_all(['wvs', 'ess_wave_10'])
    """
    
    def __init__(self, paths: DataPaths, verbose: bool = True):
        """
        Initialize the survey loader.
        
        Args:
            paths: DataPaths configuration with directory locations
            verbose: If True, print progress messages
        """
        self.paths = paths
        self.verbose = verbose
        self._cache: Dict[str, Tuple[pd.DataFrame, dict]] = {}
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_survey(
        self,
        survey_id: str,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load a single survey's data and metadata.
        
        Args:
            survey_id: Survey identifier (e.g., 'wvs', 'afrobarometer')
            use_cache: If True, return cached data if available
            
        Returns:
            Tuple of (DataFrame with survey data, metadata dict)
        """
        # Check cache
        if use_cache and survey_id in self._cache:
            self._log(f"✓ {survey_id}: loaded from cache")
            return self._cache[survey_id]
        
        config = get_survey_config(survey_id)
        self._log(f"Loading {config.name}...")
        
        # Load data
        df = self._load_data(config)
        self._log(f"  Data: {len(df):,} rows, {len(df.columns)} columns")
        
        # Apply preprocessing
        df = self._preprocess(df, config)
        
        # Load metadata
        metadata = self._load_metadata(config)
        n_sections = len(metadata)
        n_vars = sum(len(v) for v in metadata.values() if isinstance(v, dict))
        self._log(f"  Metadata: {n_sections} sections, {n_vars} variables")
        
        # Validate
        issues = self._validate(df, metadata, config)
        for issue in issues:
            warnings.warn(f"{survey_id}: {issue}")
        
        self._log(f"✓ {survey_id}: loaded successfully")
        
        # Cache and return
        result = (df, metadata)
        self._cache[survey_id] = result
        return result
    
    def _load_data(self, config: SurveyConfig) -> pd.DataFrame:
        """Load survey data files."""
        survey_dir = self.paths.raw_data_dir / config.folder_name
        
        if not survey_dir.exists():
            raise FileNotFoundError(
                f"Survey directory not found: {survey_dir}\n"
                f"Expected folder '{config.folder_name}' in {self.paths.raw_data_dir}"
            )
        
        # Find matching files
        files = find_data_files(survey_dir, config.get_file_patterns())
        
        if config.multi_file:
            df = load_multiple_files(files, encoding=config.encoding)
        else:
            df = load_file(files[0], encoding=config.encoding)
        
        return df
    
    def _preprocess(self, df: pd.DataFrame, config: SurveyConfig) -> pd.DataFrame:
        """Apply survey-specific preprocessing."""
        df = df.copy()
        
        # Construct composite ID if needed (e.g., Latinobarometer)
        if config.id_columns_to_combine:
            df = self._construct_composite_id(df, config)
        
        return df
    
    def _construct_composite_id(
        self,
        df: pd.DataFrame,
        config: SurveyConfig
    ) -> pd.DataFrame:
        """Construct respondent ID from multiple columns."""
        columns = config.id_columns_to_combine
        separator = config.id_separator
        
        # Check that all columns exist
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"ID columns not found in data: {missing}. "
                f"Available columns: {list(df.columns)[:10]}..."
            )
        
        # Construct composite ID
        id_parts = [df[col].astype(str) for col in columns]
        df[config.respondent_id_col] = id_parts[0]
        for part in id_parts[1:]:
            df[config.respondent_id_col] = df[config.respondent_id_col] + separator + part
        
        self._log(f"  Constructed ID from columns: {list(columns)}")
        
        return df
    
    def _load_metadata(self, config: SurveyConfig) -> dict:
        """Load survey metadata JSON."""
        metadata_path = self.paths.metadata_dir / config.metadata_path
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Expected at: {config.metadata_path} relative to {self.paths.metadata_dir}"
            )
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _validate(
        self,
        df: pd.DataFrame,
        metadata: dict,
        config: SurveyConfig
    ) -> List[str]:
        """Validate loaded data and metadata."""
        issues = []
        
        # Check ID column exists
        if config.respondent_id_col not in df.columns:
            issues.append(f"Respondent ID column '{config.respondent_id_col}' not found")
        
        # Check country column exists  
        if config.country_col not in df.columns:
            issues.append(f"Country column '{config.country_col}' not found")
        
        # Check for duplicate IDs
        if config.respondent_id_col in df.columns:
            n_unique = df[config.respondent_id_col].nunique()
            n_total = len(df)
            if n_unique < n_total:
                issues.append(f"Duplicate respondent IDs: {n_total - n_unique:,} duplicates")
        
        return issues
    
    def load_all(
        self,
        survey_ids: Optional[List[str]] = None,
        skip_errors: bool = True,
    ) -> Dict[str, Tuple[pd.DataFrame, dict]]:
        """
        Load multiple surveys.
        
        Args:
            survey_ids: List of survey IDs to load (None = all available)
            skip_errors: If True, continue loading other surveys if one fails
            
        Returns:
            Dictionary mapping survey_id to (DataFrame, metadata) tuples
        """
        if survey_ids is None:
            survey_ids = list(SURVEY_REGISTRY.keys())
        
        results = {}
        
        for survey_id in survey_ids:
            try:
                results[survey_id] = self.load_survey(survey_id)
            except Exception as e:
                if skip_errors:
                    warnings.warn(f"Failed to load {survey_id}: {e}")
                else:
                    raise
        
        return results
    
    def clear_cache(self, survey_id: Optional[str] = None) -> None:
        """
        Clear cached survey data.
        
        Args:
            survey_id: Specific survey to clear (None = clear all)
        """
        if survey_id:
            self._cache.pop(survey_id, None)
        else:
            self._cache.clear()
    
    def get_info(self, survey_id: str) -> Dict[str, Any]:
        """
        Get information about a survey configuration.
        
        Args:
            survey_id: Survey identifier
            
        Returns:
            Dictionary with survey configuration details
        """
        config = get_survey_config(survey_id)
        
        info = {
            'name': config.name,
            'survey_id': config.survey_id,
            'respondent_id_col': config.respondent_id_col,
            'country_col': config.country_col,
            'data_path': str(self.paths.raw_data_dir / config.folder_name),
            'metadata_path': str(self.paths.metadata_dir / config.metadata_path),
        }
        
        # Add loaded stats if cached
        if survey_id in self._cache:
            df, metadata = self._cache[survey_id]
            info['n_respondents'] = len(df)
            info['n_columns'] = len(df.columns)
            info['n_metadata_sections'] = len(metadata)
        
        return info


def scan_survey_directory(raw_data_dir: Path, verbose: bool = True) -> Dict[str, List[Path]]:
    """
    Scan a directory to discover available survey data files.
    
    Useful for checking what data is available before configuring paths.
    
    Args:
        raw_data_dir: Root directory to scan
        verbose: If True, print discovered files
        
    Returns:
        Dictionary mapping folder names to lists of data files found
    """
    raw_data_dir = Path(raw_data_dir)
    
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_data_dir}")
    
    results = {}
    patterns = ['*.csv', '*.dta', '*.sav']
    
    for folder in sorted(raw_data_dir.iterdir()):
        if folder.is_dir():
            files = []
            for pattern in patterns:
                files.extend(folder.glob(pattern))
                files.extend(folder.glob(f'*/{pattern}'))  # Check subdirs too
            
            if files:
                results[folder.name] = sorted(files)
                if verbose:
                    print(f"{folder.name}/")
                    for f in files[:3]:
                        print(f"  {f.name}")
                    if len(files) > 3:
                        print(f"  ... and {len(files) - 3} more files")
    
    return results