"""
File loaders for different survey data formats.

This module provides functions to load survey data from various file formats
(CSV, Stata DTA, SPSS SAV) with appropriate handling for each format.
"""

import pandas as pd
from pathlib import Path
from typing import List
import warnings


def load_csv(
    filepath: Path,
    encoding: str = 'utf-8',
    **kwargs
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding (default: utf-8)
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with the loaded data
    """
    encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252']
    
    # Set low_memory=False to avoid mixed type warnings
    kwargs.setdefault('low_memory', False)
    
    for enc in encodings_to_try:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    
    # Last resort: ignore errors
    warnings.warn(f"Could not decode {filepath} with standard encodings, using errors='ignore'")
    return pd.read_csv(filepath, encoding='utf-8', errors='ignore', **kwargs)


def load_stata(filepath: Path, **kwargs) -> pd.DataFrame:
    """Load a Stata DTA file into a DataFrame."""
    return pd.read_stata(filepath, **kwargs)


def load_spss(filepath: Path, convert_categoricals: bool = False, **kwargs) -> pd.DataFrame:
    """
    Load an SPSS SAV file into a DataFrame.
    
    Args:
        filepath: Path to the SAV file
        convert_categoricals: If False (default), keep numeric codes.
                              If True, convert to category labels.
        **kwargs: Additional arguments passed to pd.read_spss
    
    Requires pyreadstat to be installed.
    """
    try:
        return pd.read_spss(filepath, convert_categoricals=convert_categoricals, **kwargs)
    except ImportError:
        raise ImportError(
            "Loading SPSS files requires pyreadstat. "
            "Install it with: pip install pyreadstat"
        )


def load_file(
    filepath: Path,
    encoding: str = 'utf-8',
    **kwargs
) -> pd.DataFrame:
    """
    Load a data file, automatically detecting format from extension.
    
    Supported formats: .csv, .dta, .sav
    
    Args:
        filepath: Path to the data file
        encoding: Encoding for CSV files
        **kwargs: Additional arguments passed to the appropriate loader
        
    Returns:
        DataFrame with the loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        return load_csv(filepath, encoding=encoding, **kwargs)
    elif suffix == '.dta':
        return load_stata(filepath, **kwargs)
    elif suffix == '.sav':
        return load_spss(filepath, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .csv, .dta, .sav"
        )


def find_data_files(
    directory: Path,
    patterns: List[str],
) -> List[Path]:
    """
    Find data files in a directory matching given patterns.
    
    Patterns are tried in order; returns files matching the first 
    pattern that finds any files.
    
    Args:
        directory: Directory to search in
        patterns: Glob patterns to try (e.g., ['*.csv', '*.dta'])
        
    Returns:
        List of matching file paths (sorted for deterministic ordering)
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return sorted(matches)
    
    raise ValueError(
        f"No data files found in {directory} matching patterns: {patterns}"
    )


def load_multiple_files(
    filepaths: List[Path],
    encoding: str = 'utf-8',
    **kwargs
) -> pd.DataFrame:
    """
    Load and concatenate multiple data files.
    
    Useful for surveys split across multiple files (e.g., one per country).
    
    Args:
        filepaths: List of file paths to load
        encoding: Encoding for CSV files
        **kwargs: Additional arguments passed to loaders
        
    Returns:
        Concatenated DataFrame from all files
    """
    if not filepaths:
        raise ValueError("No files provided to load")
    
    dfs = []
    for fp in filepaths:
        try:
            df = load_file(fp, encoding=encoding, **kwargs)
            df['_source_file'] = fp.name
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to load {fp}: {e}")
    
    if not dfs:
        raise ValueError("Failed to load any files")
    
    return pd.concat(dfs, ignore_index=True)