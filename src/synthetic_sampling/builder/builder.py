"""
Dataset Builder for LLM Survey Prediction Experiments.

This module provides the DatasetBuilder class which orchestrates:
- Loading survey data via SurveyLoader
- Configuring RespondentProfileGenerator
- Sampling target questions
- Generating prediction instances
- Saving to JSONL format
"""

import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np

from ..config import DataPaths, DatasetConfig, GeneratorConfig, get_survey_config
from ..loaders import SurveyLoader
from ..profiles.generator import RespondentProfileGenerator


class DatasetBuilder:
    """
    Orchestrates dataset generation across surveys.
    
    This is a thin wrapper that coordinates:
    - SurveyLoader for data/metadata loading
    - RespondentProfileGenerator for instance generation
    - Target question sampling
    - Output serialization
    
    Example
    -------
    >>> from synthetic_sampling.config import DataPaths, DatasetConfig, GeneratorConfig
    >>> from synthetic_sampling.datasets import DatasetBuilder
    >>> 
    >>> paths = DataPaths(
    ...     raw_data_dir='~/data/surveys',
    ...     metadata_dir='./src/synthetic_sampling/profiles/metadata',
    ...     output_dir='./outputs'
    ... )
    >>> dataset_config = DatasetConfig(
    ...     n_respondents_per_survey=100,
    ...     n_targets_per_respondent=3,
    ...     n_sections=3,
    ...     m_features_per_section=2,
    ...     seed=42
    ... )
    >>> generator_config = GeneratorConfig()
    >>> 
    >>> builder = DatasetBuilder(paths, dataset_config, generator_config)
    >>> instances = builder.build_dataset(['wvs', 'ess_wave_10'])
    >>> builder.save_jsonl(instances, 'my_dataset.jsonl')
    """
    
    def __init__(
        self,
        paths: DataPaths,
        dataset_config: DatasetConfig,
        generator_config: Optional[GeneratorConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize the DatasetBuilder.
        
        Parameters
        ----------
        paths : DataPaths
            Configuration for data/metadata/output directories
        dataset_config : DatasetConfig
            Configuration for dataset generation (n_respondents, n_sections, etc.)
        generator_config : GeneratorConfig, optional
            Configuration for profile generator (missing values, similarity).
            If None, uses defaults.
        verbose : bool, default True
            Print progress messages
        """
        self.paths = paths
        self.dataset_config = dataset_config
        self.generator_config = generator_config or GeneratorConfig()
        self.verbose = verbose
        
        self.loader = SurveyLoader(paths, verbose=verbose)
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    # -------------------------------------------------------------------------
    # Target Sampling
    # -------------------------------------------------------------------------
    
    def sample_target_questions(
        self,
        metadata: dict,
        n: int,
        seed: Optional[int] = None,
        exclude_sections: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Sample n random target questions from survey metadata.
        
        Parameters
        ----------
        metadata : dict
            Survey metadata (section -> var_code -> info)
        n : int
            Number of questions to sample
        seed : int, optional
            Random seed for reproducibility
        exclude_sections : list[str], optional
            Sections to exclude from sampling (e.g., demographics)
            
        Returns
        -------
        list[dict]
            List of dicts with keys: 'section', 'var_code', 'question', 'values'
        """
        if seed is not None:
            random.seed(seed)
        
        # Flatten metadata structure
        all_questions = []
        for section, variables in metadata.items():
            if exclude_sections and section in exclude_sections:
                continue
            if not isinstance(variables, dict):
                continue
            for var_code, var_info in variables.items():
                if not isinstance(var_info, dict):
                    continue
                # Only include questions that have values defined
                if isinstance(var_info.get('values'), dict) and var_info.get('values') != {}:
                    all_questions.append({
                        'section': section,
                        'var_code': var_code,
                        'question': var_info.get('question', var_info.get('description', var_code)),
                        'values': var_info.get('values', {}),
                    })
        
        n = min(n, len(all_questions))
        if n == 0:
            return []
        
        return random.sample(all_questions, n)
    
    # -------------------------------------------------------------------------
    # Single Survey Processing
    # -------------------------------------------------------------------------
    
    def _create_generator(
        self,
        survey_id: str,
        df,
        metadata: dict
    ) -> RespondentProfileGenerator:
        """Create a configured RespondentProfileGenerator for a survey."""
        survey_config = get_survey_config(survey_id)
        
        return RespondentProfileGenerator(
            survey_data=df,
            metadata=metadata,
            respondent_id_col=survey_config.respondent_id_col,
            country_col=survey_config.country_col,
            survey=survey_id,
            missing_value_labels=self.generator_config.missing_value_labels,
            missing_value_patterns=self.generator_config.missing_value_patterns,
            similarity_model=self.generator_config.similarity_model if self.generator_config.use_semantic_filtering else None,
            similarity_threshold=self.generator_config.similarity_threshold
        )
    
    def _sample_respondents(
        self,
        df,
        respondent_id_col: str,
        n: int,
        seed: int
    ) -> List[Any]:
        """Sample n respondent IDs from a survey DataFrame."""
        np.random.seed(seed)
        all_ids = df[respondent_id_col].dropna().unique()
        n = min(n, len(all_ids))
        sampled_indices = np.random.choice(len(all_ids), size=n, replace=False)
        return [all_ids[i] for i in sampled_indices]
    
    def build_survey_dataset(
        self,
        survey_id: str,
        target_codes: Optional[List[str]] = None,
        respondent_ids: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build dataset for a single survey.
        
        Parameters
        ----------
        survey_id : str
            Survey identifier (e.g., 'wvs', 'ess_wave_10')
        target_codes : list[str], optional
            Specific target question codes. If None, samples randomly.
        respondent_ids : list, optional
            Specific respondent IDs. If None, samples randomly.
            
        Returns
        -------
        list[dict]
            List of prediction instances as dicts
        """
        self._log(f"\n{'='*60}")
        self._log(f"Building dataset for: {survey_id}")
        self._log(f"{'='*60}")
        
        # Load data and metadata
        df, metadata = self.loader.load_survey(survey_id)
        survey_config = get_survey_config(survey_id)
        
        # Create generator
        generator = self._create_generator(survey_id, df, metadata)
        
        # Sample or use provided targets
        if target_codes is None:
            sampled_targets = self.sample_target_questions(
                metadata,
                n=self.dataset_config.n_targets_per_respondent,
                seed=self.dataset_config.seed
            )
            target_codes = [t['var_code'] for t in sampled_targets]
            self._log(f"Sampled {len(target_codes)} target questions")
        
        # Set targets in generator
        generator.set_target_questions(target_codes)
        
        # Sample or use provided respondents
        if respondent_ids is None:
            respondent_ids = self._sample_respondents(
                df,
                survey_config.respondent_id_col,
                n=self.dataset_config.n_respondents_per_survey,
                seed=self.dataset_config.seed
            )
            self._log(f"Sampled {len(respondent_ids)} respondents")
        
        # Generate instances using existing generator method
        self._log(f"Generating instances...")
        instances = generator.generate_dataset_as_list(
            respondent_ids=respondent_ids,
            n_sections=self.dataset_config.n_sections,
            m_features_per_section=self.dataset_config.m_features_per_section,
            seed=self.dataset_config.seed,
            target_codes=target_codes,
            as_dicts=True
        )
        
        self._log(f"✓ Generated {len(instances)} instances for {survey_id}")
        
        return instances
    
    # -------------------------------------------------------------------------
    # Multi-Survey Processing
    # -------------------------------------------------------------------------
    
    def build_dataset(
        self,
        survey_ids: Optional[List[str]] = None,
        skip_errors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Build dataset across multiple surveys.
        
        Parameters
        ----------
        survey_ids : list[str], optional
            Surveys to include. If None, uses dataset_config.surveys or all available.
        skip_errors : bool, default True
            If True, continue with other surveys if one fails.
            
        Returns
        -------
        list[dict]
            Combined list of prediction instances from all surveys
        """
        # Determine which surveys to process
        if survey_ids is None:
            survey_ids = self.dataset_config.surveys
        if survey_ids is None:
            from ..config import ALL_SURVEYS
            survey_ids = ALL_SURVEYS
        
        self._log(f"\nBuilding dataset for {len(survey_ids)} surveys: {survey_ids}")
        
        all_instances = []
        survey_counts = {}
        
        for survey_id in survey_ids:
            try:
                instances = self.build_survey_dataset(survey_id)
                all_instances.extend(instances)
                survey_counts[survey_id] = len(instances)
            except Exception as e:
                if skip_errors:
                    self._log(f"⚠ Failed to process {survey_id}: {e}")
                    survey_counts[survey_id] = 0
                else:
                    raise
        
        # Summary
        self._log(f"\n{'='*60}")
        self._log(f"DATASET COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"Total instances: {len(all_instances)}")
        for survey_id, count in survey_counts.items():
            self._log(f"  {survey_id}: {count}")
        
        return all_instances
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    
    def save_jsonl(
        self,
        instances: List[Dict[str, Any]],
        filename: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save instances to JSONL file.
        
        Parameters
        ----------
        instances : list[dict]
            Prediction instances to save
        filename : str
            Output filename (e.g., 'dataset.jsonl')
        output_dir : Path, optional
            Output directory. If None, uses paths.output_dir
            
        Returns
        -------
        Path
            Full path to saved file
        """
        if output_dir is None:
            output_dir = self.paths.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for instance in instances:
                f.write(json.dumps(instance, default=str, ensure_ascii=False) + '\n')
        
        self._log(f"\n✓ Saved {len(instances)} instances to {output_path}")
        
        return output_path
    
    def save_json(
        self,
        instances: List[Dict[str, Any]],
        filename: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save instances to JSON file (as array).
        
        Parameters
        ----------
        instances : list[dict]
            Prediction instances to save
        filename : str
            Output filename (e.g., 'dataset.json')
        output_dir : Path, optional
            Output directory. If None, uses paths.output_dir
            
        Returns
        -------
        Path
            Full path to saved file
        """
        if output_dir is None:
            output_dir = self.paths.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instances, f, default=str, ensure_ascii=False, indent=2)
        
        self._log(f"\n✓ Saved {len(instances)} instances to {output_path}")
        
        return output_path