"""Path and sampling configuration for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class DataPaths:
    """Resolved paths for survey microdata, bundled metadata, and outputs."""

    raw_data_dir: Path
    metadata_dir: Path
    output_dir: Path

    def __post_init__(self) -> None:
        self.raw_data_dir = self._resolve_path(self.raw_data_dir)
        self.metadata_dir = self._resolve_path(self.metadata_dir)
        self.output_dir = self._resolve_path(self.output_dir)

    @staticmethod
    def _resolve_path(path: Any) -> Path:
        path_str = str(path) if not isinstance(path, Path) else str(path)
        return Path(os.path.expandvars(os.path.expanduser(path_str)))

    @classmethod
    def from_yaml(cls, config_path: Union[Path, str]) -> "DataPaths":
        if not YAML_AVAILABLE:
            raise ImportError("Loading from YAML requires pyyaml.")
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        paths_cfg = cfg.get("paths", {})
        missing = [k for k in ("raw_data", "metadata", "output") if k not in paths_cfg]
        if missing:
            raise KeyError(f"Missing required path keys in config: {missing}")
        return cls(
            raw_data_dir=paths_cfg["raw_data"],
            metadata_dir=paths_cfg["metadata"],
            output_dir=paths_cfg["output"],
        )

    @classmethod
    def from_dict(cls, paths_dict: Dict[str, str]) -> "DataPaths":
        return cls(
            raw_data_dir=paths_dict["raw_data"],
            metadata_dir=paths_dict["metadata"],
            output_dir=paths_dict["output"],
        )

    @classmethod
    def default_bundled(cls, raw_data_dir: Union[Path, str],
                        output_dir: Union[Path, str] = "./outputs") -> "DataPaths":
        """Use package-bundled metadata under surveys/metadata/."""
        from . import metadata as meta_pkg
        meta_dir = Path(meta_pkg.__file__).resolve().parent
        return cls(raw_data_dir=raw_data_dir, metadata_dir=meta_dir,
                   output_dir=output_dir)

    def validate(self, check_writable: bool = True) -> List[str]:
        issues: List[str] = []
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
            test_file = self.output_dir / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                issues.append(f"output_dir is not writable: {self.output_dir}")
        return issues


@dataclass
class GeneratorConfig:
    """Missing-value and optional semantic-filter settings for profile generation."""

    missing_value_labels: List[str] = field(default_factory=lambda: [
        "Missing", "No answer", "Refused", "Not applicable", "Not asked",
    ])
    missing_value_patterns: List[str] = field(default_factory=lambda: [
        "not asked", "missing", "refused", "nan", "na",
        "not available", "no response", "not applicable",
    ])
    use_semantic_filtering: bool = False
    similarity_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7

    @classmethod
    def from_yaml(cls, config_path: Union[Path, str]) -> "GeneratorConfig":
        if not YAML_AVAILABLE:
            raise ImportError("Loading from YAML requires pyyaml.")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        gen_cfg = cfg.get("generator", {})
        return cls(
            missing_value_labels=gen_cfg.get(
                "missing_value_labels",
                ["Missing", "No answer", "Refused", "Not applicable", "Not asked"],
            ),
            missing_value_patterns=gen_cfg.get(
                "missing_value_patterns",
                ["not asked", "missing", "refused", "nan", "na", "not applicable"],
            ),
            use_semantic_filtering=gen_cfg.get("use_semantic_filtering", False),
            similarity_model=gen_cfg.get("similarity_model", "all-MiniLM-L6-v2"),
            similarity_threshold=gen_cfg.get("similarity_threshold", 0.7),
        )


@dataclass
class DatasetConfig:
    """Sampling parameters for instance generation."""

    n_respondents_per_survey: int = 1000
    n_targets_per_respondent: int = 5
    n_sections: int = 3
    m_features_per_section: int = 3
    profile_format: str = "qa"
    seed: int = 42
    surveys: Optional[List[str]] = None

    @classmethod
    def from_yaml(cls, config_path: Union[Path, str]) -> "DatasetConfig":
        if not YAML_AVAILABLE:
            raise ImportError("Loading from YAML requires pyyaml.")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg.get("dataset", {})
        return cls(
            n_respondents_per_survey=ds_cfg.get("n_respondents_per_survey", 1000),
            n_targets_per_respondent=ds_cfg.get("n_targets_per_respondent", 5),
            n_sections=ds_cfg.get("n_sections", 3),
            m_features_per_section=ds_cfg.get("m_features_per_section", 3),
            profile_format=ds_cfg.get("profile_format", "qa"),
            seed=ds_cfg.get("seed", 42),
            surveys=ds_cfg.get("surveys", None),
        )

    @property
    def profile_type_code(self) -> str:
        return f"s{self.n_sections}m{self.m_features_per_section}"


def load_config(config_path: Union[Path, str]) -> Dict[str, Any]:
    return {
        "paths": DataPaths.from_yaml(config_path),
        "generator": GeneratorConfig.from_yaml(config_path),
        "dataset": DatasetConfig.from_yaml(config_path),
    }
