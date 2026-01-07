"""
Base configuration classes for the synthetic sampling pipeline.

This module provides path management and general configuration loading,
allowing the pipeline to work across different environments (local, cluster, Colab)
without hardcoded paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import os

# yaml is optional - only needed if loading from YAML files
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class DataPaths:
    """
    Paths configuration for survey data and outputs.
    
    All paths are resolved at runtime, allowing the same code to work
    across different environments by swapping config files.
    
    Attributes:
        raw_data_dir: Root directory containing survey data files (CSV, DTA, SAV)
                      Expected structure: raw_data_dir/{survey_folder}/files...
        metadata_dir: Root directory containing metadata JSON files
        output_dir: Directory for generated datasets and artifacts
    
    Example:
        >>> paths = DataPaths.from_yaml("configs/local.yaml")
        >>> paths.raw_data_dir
        PosixPath('/Users/maksim/data/surveys')
        
        >>> # Or create directly
        >>> paths = DataPaths(
        ...     raw_data_dir='~/data/surveys',
        ...     metadata_dir='./src/synthetic_sampling/profiles/metadata',
        ...     output_dir='./outputs'
        ... )
    """
    raw_data_dir: Path
    metadata_dir: Path
    output_dir: Path
    
    def __post_init__(self):
        """Convert string paths to Path objects and expand user/env vars."""
        self.raw_data_dir = self._resolve_path(self.raw_data_dir)
        self.metadata_dir = self._resolve_path(self.metadata_dir)
        self.output_dir = self._resolve_path(self.output_dir)
    
    @staticmethod
    def _resolve_path(path: Any) -> Path:
        """Resolve a path string, expanding ~ and environment variables."""
        if isinstance(path, Path):
            path_str = str(path)
        else:
            path_str = path
        # Expand ~ and $ENV_VAR
        expanded = os.path.expandvars(os.path.expanduser(path_str))
        return Path(expanded)
    
    @classmethod
    def from_yaml(cls, config_path: Path | str) -> 'DataPaths':
        """
        Load paths from a YAML configuration file.
        
        Expected YAML structure:
            paths:
              raw_data: /path/to/surveys
              metadata: /path/to/metadata
              output: /path/to/output
        
        Args:
            config_path: Path to the YAML config file
            
        Returns:
            DataPaths instance with resolved paths
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If required keys are missing from config
            ImportError: If pyyaml is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "Loading from YAML requires pyyaml. "
                "Install it with: pip install pyyaml"
            )
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        paths_cfg = cfg.get('paths', {})
        
        required_keys = ['raw_data', 'metadata', 'output']
        missing = [k for k in required_keys if k not in paths_cfg]
        if missing:
            raise KeyError(f"Missing required path keys in config: {missing}")
        
        return cls(
            raw_data_dir=paths_cfg['raw_data'],
            metadata_dir=paths_cfg['metadata'],
            output_dir=paths_cfg['output'],
        )
    
    @classmethod
    def from_dict(cls, paths_dict: Dict[str, str]) -> 'DataPaths':
        """
        Create DataPaths from a dictionary.
        
        Useful for programmatic configuration or testing.
        
        Args:
            paths_dict: Dictionary with keys 'raw_data', 'metadata', 'output'
        """
        return cls(
            raw_data_dir=paths_dict['raw_data'],
            metadata_dir=paths_dict['metadata'],
            output_dir=paths_dict['output'],
        )
    
    def validate(self, check_writable: bool = True) -> List[str]:
        """
        Validate that configured paths exist and are accessible.
        
        Args:
            check_writable: If True, also check that output_dir is writable
            
        Returns:
            List of warning/error messages (empty if all valid)
        """
        issues = []
        
        if not self.raw_data_dir.exists():
            issues.append(f"raw_data_dir does not exist: {self.raw_data_dir}")
        
        if not self.metadata_dir.exists():
            issues.append(f"metadata_dir does not exist: {self.metadata_dir}")
        
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                issues.append(f"Cannot create output_dir: {self.output_dir}")
        
        if check_writable and self.output_dir.exists():
            test_file = self.output_dir / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                issues.append(f"output_dir is not writable: {self.output_dir}")
        
        return issues
    
    def __repr__(self) -> str:
        return (
            f"DataPaths(\n"
            f"  raw_data_dir={self.raw_data_dir},\n"
            f"  metadata_dir={self.metadata_dir},\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )


@dataclass
class GeneratorConfig:
    """
    Configuration for the RespondentProfileGenerator.
    
    These settings control how profiles are generated, including
    missing value handling and semantic filtering.
    """
    # Missing value configuration
    missing_value_labels: List[str] = field(default_factory=lambda: [
        'Missing', 'No answer', 'Refused', 'Not applicable'
    ])
    missing_value_patterns: List[str] = field(default_factory=lambda: [
        'not asked', "don't know", 'missing', 'refused', 'nan', 'na',
        'not available', 'no response'
    ])
    
    # Semantic similarity filtering
    use_semantic_filtering: bool = True
    similarity_model: str = 'all-MiniLM-L6-v2'
    similarity_threshold: float = 0.7
    
    @classmethod
    def from_yaml(cls, config_path: Path | str) -> 'GeneratorConfig':
        """Load generator config from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("Loading from YAML requires pyyaml.")
        
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        gen_cfg = cfg.get('generator', {})
        
        return cls(
            missing_value_labels=gen_cfg.get(
                'missing_value_labels', 
                ['Missing', 'No answer', 'Refused', 'Not applicable']
            ),
            missing_value_patterns=gen_cfg.get(
                'missing_value_patterns',
                ['not asked', "don't know", 'missing', 'refused', 'nan', 'na']
            ),
            use_semantic_filtering=gen_cfg.get('use_semantic_filtering', True),
            similarity_model=gen_cfg.get('similarity_model', 'all-MiniLM-L6-v2'),
            similarity_threshold=gen_cfg.get('similarity_threshold', 0.7),
        )


@dataclass 
class DatasetConfig:
    """
    Configuration for dataset generation.
    
    Controls sampling parameters and output format for generating
    prediction instances.
    """
    # Sampling parameters
    n_respondents_per_survey: int = 1000
    n_targets_per_respondent: int = 5
    n_sections: int = 3
    m_features_per_section: int = 3
    
    # Output format
    profile_format: str = 'qa'  # 'qa', 'bullet', 'narrative'
    
    # Reproducibility
    seed: int = 42
    
    # Survey selection (None = all available)
    surveys: Optional[List[str]] = None
    
    @classmethod
    def from_yaml(cls, config_path: Path | str) -> 'DatasetConfig':
        """Load dataset config from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("Loading from YAML requires pyyaml.")
            
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        ds_cfg = cfg.get('dataset', {})
        
        return cls(
            n_respondents_per_survey=ds_cfg.get('n_respondents_per_survey', 100),
            n_targets_per_respondent=ds_cfg.get('n_targets_per_respondent', 3),
            n_sections=ds_cfg.get('n_sections', 3),
            m_features_per_section=ds_cfg.get('m_features_per_section', 2),
            profile_format=ds_cfg.get('profile_format', 'qa'),
            seed=ds_cfg.get('seed', 42),
            surveys=ds_cfg.get('surveys', None),
        )
    
    @property
    def profile_type_code(self) -> str:
        """Generate profile type code like 's3m2' for 3 sections, 2 features each."""
        return f"s{self.n_sections}m{self.m_features_per_section}"


def load_config(config_path: Path | str) -> Dict[str, Any]:
    """
    Load a complete configuration file and return all config objects.
    
    This is a convenience function for loading all configuration
    components from a single YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with keys 'paths', 'generator', 'dataset'
    """
    config_path = Path(config_path)
    
    return {
        'paths': DataPaths.from_yaml(config_path),
        'generator': GeneratorConfig.from_yaml(config_path),
        'dataset': DatasetConfig.from_yaml(config_path),
    }