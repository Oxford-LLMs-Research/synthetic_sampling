"""
Respondent Profile Generator for LLM Survey Prediction Experiments

This module provides a class for constructing respondent profiles and 
prediction instances for evaluating LLMs on individual-level survey prediction.

Key capabilities:
- Stratified random sampling across survey thematic sections
- Seedable, perfectly reproducible sampling
- Profile expansion (add features while preserving existing ones)
- Target question handling (automatic exclusion from feature pool)
- Generation of prediction instances combining profiles + targets
- Required feature inclusion (e.g., country, key demographics)
- Missing value handling (exclude survey artifacts from features and options)
- Country-specific answer options (auto-detected for party vote, religion, etc.)
- Semantic similarity filtering (exclude features too similar to target questions)
- Flexible profile formatting (multiple prompt styles for robustness testing)
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Iterator
import hashlib
from copy import deepcopy

from .dataclasses import (
    ProfileConfig,
    RespondentProfile,
    TargetQuestion,
    PredictionInstance
)


# =============================================================================
# Main Generator Class
# =============================================================================


class RespondentProfileGenerator:
    """
    Generates respondent profiles with stratified random feature sampling.
    
    Key design decisions:
    - Features are sampled WITHOUT replacement within a respondent
    - Sampling is stratified across thematic sections
    - Profiles are deterministically reproducible given (respondent_id, seed)
    - Expansion adds features while preserving all existing selections
    - Missing/artifact values can be excluded from features and targets
    - Semantic similarity filtering prevents information leakage from features to targets
    
    Parameters
    ----------
    survey_data : pd.DataFrame
        Survey response data. Rows are respondents, columns are question codes.
        Must have an index or column that serves as respondent identifier.
    metadata : dict
        Nested dict: section -> question_code -> {description, question, values}
    respondent_id_col : str, optional
        Column name for respondent IDs. If None, uses DataFrame index.
    country_col : str, optional
        Column name for country variable (e.g., 'B_COUNTRY', 'Q202'). 
        If None, country will not be included in prediction instances.
    survey : str, optional
        Name of the survey. If None, survey will not be included in prediction instances.
    missing_value_labels : list[str], optional
        Exact value labels to treat as missing/artifacts and exclude.
        E.g., ["Missing", "Not asked", "No answer", "Refused", "Don't know"]
        These are excluded from: (1) answer options for targets, 
        (2) feature values (respondent's answer is checked).
    missing_value_patterns : list[str], optional
        Substring patterns to match against value labels (case-insensitive).
        E.g., ["not asked", "missing", "refused"] will match 
        "Not asked in this country", "Missing data", etc.
        Use this for flexible matching across surveys with varied phrasing.
    similarity_model : str, optional
        Name of sentence-transformers model for semantic similarity filtering.
        E.g., "all-MiniLM-L6-v2" (fast) or "all-mpnet-base-v2" (accurate).
        If None, semantic similarity filtering is disabled.
    similarity_threshold : float, default 0.7
        Cosine similarity threshold for excluding features similar to targets.
        Features with similarity >= threshold are excluded for that target.
        Higher = more permissive (fewer exclusions), lower = stricter.
    
    Example
    -------
    >>> generator = RespondentProfileGenerator(
    ...     df, metadata, 
    ...     respondent_id_col='respondent_id',
    ...     country_col='B_COUNTRY',
    ...     survey='survey_name',
    ...     missing_value_labels=['Missing', 'No answer'],
    ...     missing_value_patterns=['not asked', 'refused'],
    ...     similarity_model='all-MiniLM-L6-v2',
    ...     similarity_threshold=0.7
    ... )
    >>> generator.set_target_questions(['Q35A', 'Q35B'])
    >>> instance = generator.generate_prediction_instance(
    ...     respondent_id=12345,
    ...     target_code='Q35A',
    ...     n_sections=3,
    ...     m_features_per_section=2,
    ...     seed=42
    ... )
    """
    
    def __init__(
        self,
        survey_data: pd.DataFrame,
        metadata: dict,
        respondent_id_col: Optional[str] = None,
        country_col: Optional[str] = None,
        survey: Optional[str] = None,
        missing_value_labels: Optional[list[str]] = None,
        missing_value_patterns: Optional[list[str]] = None,
        similarity_model: Optional[str] = None,
        similarity_threshold: float = 0.7
    ):
        self.survey_data = survey_data
        self.metadata = metadata
        self.respondent_id_col = respondent_id_col
        self.country_col = country_col
        self.survey = survey
        
        # Missing value configuration
        self.missing_value_labels = set(missing_value_labels) if missing_value_labels else set()
        self.missing_value_patterns = [p.lower() for p in (missing_value_patterns or [])]
        
        # Semantic similarity configuration
        self.similarity_model_name = similarity_model
        self.similarity_threshold = similarity_threshold
        self._embedder = None  # Lazy load
        self._question_embeddings = None  # Cache
        self._target_similar_features: dict[str, set[str]] = {}  # target_code -> excluded features
        
        # Validate country_col if provided
        if country_col is not None and country_col not in survey_data.columns:
            raise ValueError(
                f"country_col '{country_col}' not found in survey data columns"
            )
        
        # Build internal indices
        self._section_to_features = self._build_section_index()
        self._feature_to_section = self._build_reverse_index()
        self._all_features = set(self._feature_to_section.keys())
        
        # Build question text index for similarity computation
        self._code_to_question_text = self._build_question_text_index()
        
        # Exclusion/inclusion sets
        self._excluded_features: set[str] = set()
        self._always_include: list[str] = []
        
        # Validation
        self._validate_metadata_structure()
        self._validate_data_metadata_alignment()
        
        # Report missing value config
        if self.missing_value_labels or self.missing_value_patterns:
            self._report_missing_value_config()
        
        # Report similarity config
        if self.similarity_model_name:
            print(f"Semantic similarity filtering enabled:")
            print(f"  Model: {self.similarity_model_name}")
            print(f"  Threshold: {self.similarity_threshold}")
    
    def _build_section_index(self) -> dict[str, list[str]]:
        """Map section names to list of feature codes."""
        section_to_features = {}
        for section, questions in self.metadata.items():
            section_to_features[section] = list(questions.keys())
        return section_to_features
    
    def _build_reverse_index(self) -> dict[str, str]:
        """Map feature codes to their section."""
        feature_to_section = {}
        for section, questions in self.metadata.items():
            for q_code in questions.keys():
                feature_to_section[q_code] = section
        return feature_to_section
    
    def _build_question_text_index(self) -> dict[str, str]:
        """Map question codes to their question text for similarity computation."""
        code_to_text = {}
        for section, questions in self.metadata.items():
            for q_code, q_info in questions.items():
                # Prefer 'question' field, fall back to 'description'
                text = q_info.get('question', q_info.get('description', q_code))
                code_to_text[q_code] = text
        return code_to_text
    
    def _get_embedder(self):
        """Lazy load the sentence transformer model."""
        if self._embedder is None:
            if self.similarity_model_name is None:
                raise ValueError(
                    "Similarity model not configured. "
                    "Set similarity_model in __init__ to enable semantic filtering."
                )
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.similarity_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required for semantic similarity. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder
    
    def _compute_question_embeddings(self, question_codes: list[str]) -> np.ndarray:
        """
        Compute embeddings for a list of question codes.
        
        Uses caching to avoid recomputation.
        """
        embedder = self._get_embedder()
        
        # Get question texts
        texts = [self._code_to_question_text.get(code, code) for code in question_codes]
        
        # Compute embeddings
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        return embeddings
    
    def _compute_similarity_exclusions(self, target_codes: list[str]) -> dict[str, set[str]]:
        """
        Compute per-target feature exclusions based on semantic similarity.
        
        For each target, identifies features with cosine similarity >= threshold.
        
        Parameters
        ----------
        target_codes : list[str]
            Target question codes
            
        Returns
        -------
        dict[str, set[str]]
            Maps target_code -> set of feature codes to exclude
        """
        if not self.similarity_model_name:
            return {code: set() for code in target_codes}
        
        # Get all feature codes (excluding targets themselves)
        feature_codes = [c for c in self._all_features if c not in target_codes]
        
        if not feature_codes:
            return {code: set() for code in target_codes}
        
        # Compute embeddings for targets and features
        print(f"  Computing semantic similarity (model: {self.similarity_model_name})...")
        
        target_embeddings = self._compute_question_embeddings(target_codes)
        feature_embeddings = self._compute_question_embeddings(feature_codes)
        
        # Normalize for cosine similarity
        target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)
        feature_norms = np.linalg.norm(feature_embeddings, axis=1, keepdims=True)
        
        target_normalized = target_embeddings / target_norms
        feature_normalized = feature_embeddings / feature_norms
        
        # Compute similarity matrix: targets x features
        similarity_matrix = target_normalized @ feature_normalized.T
        
        # Build exclusion sets
        exclusions = {}
        for i, target_code in enumerate(target_codes):
            similar_indices = np.where(similarity_matrix[i] >= self.similarity_threshold)[0]
            similar_features = {feature_codes[j] for j in similar_indices}
            exclusions[target_code] = similar_features
            
            if similar_features:
                # Report similar features with their similarity scores
                similar_with_scores = [
                    (feature_codes[j], similarity_matrix[i, j]) 
                    for j in similar_indices
                ]
                similar_with_scores.sort(key=lambda x: -x[1])
                
                print(f"    {target_code}: excluding {len(similar_features)} similar features")
                for feat, score in similar_with_scores[:3]:  # Show top 3
                    feat_text = self._code_to_question_text.get(feat, feat)[:50]
                    print(f"      - {feat} (sim={score:.3f}): \"{feat_text}...\"")
                if len(similar_with_scores) > 3:
                    print(f"      ... and {len(similar_with_scores) - 3} more")
        
        return exclusions
    
    def get_similar_features(self, target_code: str) -> dict[str, float]:
        """
        Get features similar to a target with their similarity scores.
        
        Useful for inspection and debugging.
        
        Parameters
        ----------
        target_code : str
            Target question code
            
        Returns
        -------
        dict[str, float]
            Feature codes mapped to their similarity scores
        """
        if not self.similarity_model_name:
            return {}
        
        feature_codes = [c for c in self._all_features if c != target_code]
        
        target_emb = self._compute_question_embeddings([target_code])
        feature_emb = self._compute_question_embeddings(feature_codes)
        
        # Normalize
        target_norm = target_emb / np.linalg.norm(target_emb)
        feature_norms = feature_emb / np.linalg.norm(feature_emb, axis=1, keepdims=True)
        
        similarities = (target_norm @ feature_norms.T).flatten()
        
        return {code: float(sim) for code, sim in zip(feature_codes, similarities)}
    
    def _validate_metadata_structure(self):
        """Ensure metadata has expected nested structure."""
        for section, questions in self.metadata.items():
            if not isinstance(questions, dict):
                raise ValueError(
                    f"Section '{section}' should map to dict of questions, "
                    f"got {type(questions)}"
                )
            for q_code, q_info in questions.items():
                if not isinstance(q_info, dict):
                    raise ValueError(
                        f"Question '{q_code}' in section '{section}' should have "
                        f"dict metadata, got {type(q_info)}"
                    )
                # Check for required fields (flexible - only warn)
                if 'values' not in q_info:
                    import warnings
                    warnings.warn(
                        f"Question '{q_code}' missing 'values' mapping. "
                        "Value labels may not be available."
                    )
    
    def _validate_data_metadata_alignment(self):
        """Check that metadata features exist in survey data."""
        data_cols = set(self.survey_data.columns)
        metadata_features = self._all_features
        
        missing_in_data = metadata_features - data_cols
        if missing_in_data:
            import warnings
            warnings.warn(
                f"{len(missing_in_data)} metadata features not found in data: "
                f"{list(missing_in_data)[:5]}..."
            )
    
    def _report_missing_value_config(self):
        """Report the missing value configuration."""
        print(f"Missing value exclusion configured:")
        if self.missing_value_labels:
            print(f"  Exact labels: {self.missing_value_labels}")
        if self.missing_value_patterns:
            print(f"  Patterns (case-insensitive): {self.missing_value_patterns}")
    
    def _is_missing_value_label(self, label: str) -> bool:
        """
        Check if a value label represents a missing/artifact value.
        
        Returns True if the label should be excluded.
        """
        if label is None:
            return True
        
        label_str = str(label)
        
        # Check exact matches
        if label_str in self.missing_value_labels:
            return True
        
        # Check pattern matches (case-insensitive)
        label_lower = label_str.lower()
        for pattern in self.missing_value_patterns:
            if pattern in label_lower:
                return True
        
        return False
    
    def _filter_valid_options(self, values_map: dict[str, str]) -> list[str]:
        """
        Filter answer options to exclude missing/artifact values.
        
        Returns list of valid option labels only.
        """
        valid_options = []
        for raw_value, label in values_map.items():
            if not self._is_missing_value_label(label):
                valid_options.append(label)
        return valid_options
    
    def _respondent_has_valid_value(
        self, 
        feature_code: str, 
        respondent_data: pd.Series
    ) -> bool:
        """
        Check if respondent has a non-missing value for a feature.
        
        Returns False if the respondent's answer maps to a missing/artifact label.
        """
        raw_value = respondent_data.get(feature_code)
        if pd.isna(raw_value):
            return False
        
        label = self._get_value_label(feature_code, raw_value)
        return not self._is_missing_value_label(label)
    
    # -------------------------------------------------------------------------
    # Configuration methods
    # -------------------------------------------------------------------------
    
    def set_exclusions(self, feature_codes: list[str]):
        """
        Set features to exclude from sampling (e.g., target questions).
        
        These features will never appear in generated profiles.
        Call multiple times to add to exclusion set, or pass complete list.
        """
        self._excluded_features = set(feature_codes)
        
        # Validate exclusions exist
        unknown = self._excluded_features - self._all_features
        if unknown:
            import warnings
            warnings.warn(f"Excluded features not in metadata: {unknown}")
    
    def add_exclusions(self, feature_codes: list[str]):
        """Add to existing exclusion set."""
        self._excluded_features.update(feature_codes)
    
    def set_always_include(self, feature_codes: list[str]):
        """
        Set features that must appear in every profile (e.g., country).
        
        These features are added BEFORE random sampling and don't count
        toward the n_sections * m_features budget.
        
        Order is preserved - features appear in profiles in this order.
        """
        # Validate all exist
        for code in feature_codes:
            if code not in self._all_features:
                raise ValueError(f"Always-include feature '{code}' not in metadata")
            if code in self._excluded_features:
                raise ValueError(
                    f"Feature '{code}' cannot be both excluded and always-included"
                )
        self._always_include = list(feature_codes)
    
    def get_available_pool(self) -> dict[str, list[str]]:
        """
        Return the effective feature pool after exclusions and always-include.
        
        Returns dict mapping section -> available features in that section.
        Useful for understanding effective pool size.
        """
        pool = {}
        reserved = set(self._always_include) | self._excluded_features
        
        for section, features in self._section_to_features.items():
            available = [f for f in features if f not in reserved]
            if available:
                pool[section] = available
        return pool
    
    def get_pool_statistics(self) -> dict:
        """Return statistics about the available feature pool."""
        pool = self.get_available_pool()
        return {
            'n_sections': len(pool),
            'n_total_features': sum(len(f) for f in pool.values()),
            'features_per_section': {s: len(f) for s, f in pool.items()},
            'n_excluded': len(self._excluded_features),
            'n_always_include': len(self._always_include)
        }
    
    def get_available_pool_for_target(self, target_code: str) -> dict[str, list[str]]:
        """
        Return the effective feature pool for a specific target question.
        
        This applies both global exclusions AND per-target semantic similarity exclusions.
        
        Parameters
        ----------
        target_code : str
            The target question code
            
        Returns
        -------
        dict[str, list[str]]
            Maps section -> available features in that section
        """
        # Start with global pool
        pool = {}
        reserved = set(self._always_include) | self._excluded_features
        
        # Add target-specific exclusions from semantic similarity
        if hasattr(self, '_target_similar_features') and target_code in self._target_similar_features:
            reserved = reserved | self._target_similar_features[target_code]
        
        for section, features in self._section_to_features.items():
            available = [f for f in features if f not in reserved]
            if available:
                pool[section] = available
        return pool
    
    def get_pool_statistics_for_target(self, target_code: str) -> dict:
        """Return statistics about the available feature pool for a specific target."""
        pool = self.get_available_pool_for_target(target_code)
        n_similar_excluded = len(self._target_similar_features.get(target_code, set()))
        return {
            'n_sections': len(pool),
            'n_total_features': sum(len(f) for f in pool.values()),
            'features_per_section': {s: len(f) for s, f in pool.items()},
            'n_excluded_global': len(self._excluded_features),
            'n_excluded_similar': n_similar_excluded,
            'n_always_include': len(self._always_include)
        }
    
    # -------------------------------------------------------------------------
    # Core generation methods
    # -------------------------------------------------------------------------
    
    def _get_respondent_seed(self, respondent_id: Union[int, str], base_seed: int) -> int:
        """
        Generate a unique but reproducible seed for a respondent.
        
        Combines base_seed with respondent_id via hashing to ensure:
        1. Same (respondent_id, seed) always produces same profile
        2. Different respondents get different random sequences
        3. Changing base_seed changes all profiles
        """
        combined = f"{base_seed}_{respondent_id}"
        hash_val = int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)
        return hash_val
    
    def _get_respondent_data(self, respondent_id: Union[int, str]) -> pd.Series:
        """Retrieve a respondent's survey responses."""
        if self.respondent_id_col is not None:
            mask = self.survey_data[self.respondent_id_col] == respondent_id
            if mask.sum() == 0:
                raise KeyError(f"Respondent {respondent_id} not found")
            if mask.sum() > 1:
                raise ValueError(f"Multiple rows for respondent {respondent_id}")
            return self.survey_data[mask].iloc[0]
        else:
            return self.survey_data.loc[respondent_id]
    
    def _get_value_label(self, feature_code: str, raw_value) -> str:
        """
        Convert raw response value to human-readable label.
        
        Handles multiple data formats:
        1. Numeric codes that need mapping via metadata (e.g., 1 -> "Strongly agree")
        2. Pre-mapped string labels (e.g., "Never" stays "Never")
        3. Float representations of integers (e.g., 1.0 -> "1" -> mapped)
        """
        section = self._feature_to_section.get(feature_code)
        if section is None:
            return str(raw_value)
        
        q_info = self.metadata[section].get(feature_code, {})
        values_map = q_info.get('values', {})
        
        # If no values map, just return raw value as string
        if not values_map:
            return str(raw_value)
        
        # Try exact match first (works for both numeric and string keys)
        if raw_value in values_map:
            return values_map[raw_value]
        if str(raw_value) in values_map:
            return values_map[str(raw_value)]
        
        # Try float -> int conversion (e.g., 1.0 -> "1")
        # Only attempt if raw_value looks numeric
        try:
            int_key = str(int(float(raw_value)))
            if int_key in values_map:
                return values_map[int_key]
        except (ValueError, TypeError):
            # raw_value is not numeric (e.g., "Never") - that's fine
            pass
        
        # Check if raw_value is already a valid label (pre-mapped data)
        # This handles surveys where data contains labels instead of codes
        raw_str = str(raw_value)
        if raw_str in values_map.values():
            return raw_str
        
        # Fallback: return raw value as string
        return raw_str
    
    def _build_feature_info(
        self, 
        feature_code: str, 
        respondent_data: pd.Series
    ) -> dict:
        """Build complete feature information dict."""
        section = self._feature_to_section[feature_code]
        q_info = self.metadata[section].get(feature_code, {})
        raw_value = respondent_data.get(feature_code)
        
        return {
            'code': feature_code,
            'section': section,
            'raw_value': raw_value,
            'value_label': self._get_value_label(feature_code, raw_value),
            'question': q_info.get('question', ''),
            'description': q_info.get('description', '')
        }
    
    def generate_profile(
        self,
        respondent_id: Union[int, str],
        n_sections: int,
        m_features_per_section: int,
        seed: int,
        shuffle_features: bool = False,
        target_code: Optional[str] = None
    ) -> RespondentProfile:
        """
        Generate a respondent profile with stratified random sampling.
        
        Parameters
        ----------
        respondent_id : int or str
            Identifier for the respondent in survey_data
        n_sections : int
            Number of thematic sections to sample from
        m_features_per_section : int
            Number of features to sample from each selected section
        seed : int
            Base random seed for reproducibility
        shuffle_features : bool, default False
            If True, shuffle the order of features in the profile.
            If False, features appear grouped by section (always-include first,
            then section 1 features, section 2 features, etc.)
            Shuffling uses the same RNG so it's reproducible.
        target_code : str, optional
            If provided, applies per-target semantic similarity exclusions.
            Features similar to this target will be excluded from the pool.
            
        Returns
        -------
        RespondentProfile
            Profile object containing sampled features and metadata
            
        Raises
        ------
        ValueError
            If n_sections exceeds available sections or m_features exceeds
            smallest section size
        """
        config = ProfileConfig(
            n_sections=n_sections,
            m_features_per_section=m_features_per_section,
            seed=seed,
            shuffle_features=shuffle_features
        )
        
        # Get respondent's data
        respondent_data = self._get_respondent_data(respondent_id)
        
        # Get available pool (applying target-specific exclusions if provided)
        if target_code is not None:
            pool = self.get_available_pool_for_target(target_code)
        else:
            pool = self.get_available_pool()
        available_sections = list(pool.keys())
        
        # Validate request is satisfiable
        if n_sections > len(available_sections):
            raise ValueError(
                f"Requested {n_sections} sections but only {len(available_sections)} "
                f"available after exclusions"
            )
        
        min_section_size = min(len(f) for f in pool.values())
        if m_features_per_section > min_section_size:
            small_sections = [s for s, f in pool.items() if len(f) < m_features_per_section]
            raise ValueError(
                f"Requested {m_features_per_section} features/section but sections "
                f"{small_sections} have fewer features. Consider reducing "
                f"m_features_per_section or excluding these sections."
            )
        
        # Initialize RNG with respondent-specific seed
        rng_seed = self._get_respondent_seed(respondent_id, seed)
        rng = np.random.RandomState(rng_seed)
        
        # Sample sections
        section_indices = rng.choice(
            len(available_sections), 
            size=n_sections, 
            replace=False
        )
        selected_sections = [available_sections[i] for i in sorted(section_indices)]
        
        # Sample features using oversample-and-filter strategy
        # This ensures fixed profile sizes regardless of missing values
        sampled_features = {}
        sections_with_insufficient_features = []
        
        for section in selected_sections:
            section_features = pool[section]
            
            # Shuffle the section features to randomize selection order
            shuffled_section_features = section_features.copy()
            rng.shuffle(shuffled_section_features)
            
            # Iterate through shuffled features, keeping only valid ones
            # until we have enough or exhaust the pool
            valid_sampled = []
            for feature_code in shuffled_section_features:
                if self._respondent_has_valid_value(feature_code, respondent_data):
                    valid_sampled.append(feature_code)
                    if len(valid_sampled) >= m_features_per_section:
                        break
            
            # Check if we got enough features
            if len(valid_sampled) < m_features_per_section:
                sections_with_insufficient_features.append(
                    (section, len(valid_sampled), m_features_per_section)
                )
            
            # Add the valid features we found
            for feature_code in valid_sampled:
                sampled_features[feature_code] = self._build_feature_info(
                    feature_code, respondent_data
                )
        
        # Warn if any sections couldn't provide enough valid features
        if sections_with_insufficient_features:
            import warnings
            details = [f"{s}: got {got}/{needed}" for s, got, needed 
                      in sections_with_insufficient_features]
            warnings.warn(
                f"Respondent {respondent_id}: some sections had insufficient valid features "
                f"after filtering missing values: {details}"
            )
        
        # Add always-include features at the beginning (if they have valid values)
        final_features = {}
        for code in self._always_include:
            if self._respondent_has_valid_value(code, respondent_data):
                final_features[code] = self._build_feature_info(code, respondent_data)
            else:
                import warnings
                warnings.warn(
                    f"Respondent {respondent_id}: always-include feature '{code}' "
                    f"has missing value, skipping"
                )
        final_features.update(sampled_features)
        
        # Shuffle if requested (always-include features participate in shuffle too)
        if shuffle_features:
            feature_codes = list(final_features.keys())
            rng.shuffle(feature_codes)
            final_features = {code: final_features[code] for code in feature_codes}
        
        return RespondentProfile(
            respondent_id=respondent_id,
            features=final_features,
            config=config,
            sections_sampled=selected_sections,
            always_included=self._always_include.copy()
        )
    
    def expand_profile(
        self,
        profile: RespondentProfile,
        add_sections: int = 0,
        add_features_per_section: int = 0
    ) -> RespondentProfile:
        """
        Expand an existing profile by adding more sections and/or features.
        
        Critical: Existing features are ALWAYS preserved. New features are 
        sampled from the remaining pool, excluding already-selected features.
        This guarantees strict subset behavior regardless of expansion path.
        
        Uses oversample-and-filter strategy to ensure fixed feature counts
        per section despite missing values.
        
        Parameters
        ----------
        profile : RespondentProfile
            Existing profile to expand
        add_sections : int
            Additional sections to sample (beyond current n_sections)
        add_features_per_section : int
            Additional features to sample per section (applies to all sections,
            including newly added ones)
            
        Returns
        -------
        RespondentProfile
            New profile with additional features. Original profile is unchanged.
        """
        if add_sections == 0 and add_features_per_section == 0:
            return deepcopy(profile)
        
        # Get respondent data
        respondent_data = self._get_respondent_data(profile.respondent_id)
        
        # Build pool excluding already-selected features
        already_selected = set(profile.feature_codes)
        reserved = already_selected | self._excluded_features
        
        expansion_pool = {}
        for section, features in self._section_to_features.items():
            available = [f for f in features if f not in reserved]
            if available:
                expansion_pool[section] = available
        
        # Create expansion-specific RNG
        # Use a different but deterministic seed for expansion
        expansion_seed = self._get_respondent_seed(
            f"{profile.respondent_id}_expand_{profile.config.n_sections}_{profile.config.m_features_per_section}",
            profile.config.seed
        )
        rng = np.random.RandomState(expansion_seed)
        
        new_features = deepcopy(profile.features)
        current_sections = set(profile.sections_sampled)
        
        # Add new sections if requested
        new_sections = []
        if add_sections > 0:
            available_new_sections = [s for s in expansion_pool.keys() 
                                      if s not in current_sections]
            if add_sections > len(available_new_sections):
                raise ValueError(
                    f"Requested {add_sections} new sections but only "
                    f"{len(available_new_sections)} available"
                )
            new_section_indices = rng.choice(
                len(available_new_sections),
                size=add_sections,
                replace=False
            )
            new_sections = [available_new_sections[i] for i in sorted(new_section_indices)]
        
        all_sections = profile.sections_sampled + new_sections
        
        # Sample features using oversample-and-filter for fixed sizes
        for section in all_sections:
            if section not in expansion_pool or not expansion_pool[section]:
                continue
                
            section_pool = expansion_pool[section]
            
            if section in new_sections:
                # New section: sample m + add_features_per_section
                n_to_sample = profile.config.m_features_per_section + add_features_per_section
            else:
                # Existing section: add more features
                n_to_sample = add_features_per_section
            
            if n_to_sample <= 0:
                continue
            
            # Shuffle pool and take first n valid features
            shuffled_pool = section_pool.copy()
            rng.shuffle(shuffled_pool)
            
            valid_sampled = []
            for feature_code in shuffled_pool:
                if self._respondent_has_valid_value(feature_code, respondent_data):
                    valid_sampled.append(feature_code)
                    if len(valid_sampled) >= n_to_sample:
                        break
            
            # Add valid features
            for feature_code in valid_sampled:
                new_features[feature_code] = self._build_feature_info(
                    feature_code, respondent_data
                )
        
        new_config = ProfileConfig(
            n_sections=profile.config.n_sections + add_sections,
            m_features_per_section=profile.config.m_features_per_section + add_features_per_section,
            seed=profile.config.seed
        )
        
        return RespondentProfile(
            respondent_id=profile.respondent_id,
            features=new_features,
            config=new_config,
            sections_sampled=all_sections,
            always_included=profile.always_included.copy()
        )
    
    # -------------------------------------------------------------------------
    # Batch generation methods
    # -------------------------------------------------------------------------
    
    def generate_profiles_batch(
        self,
        respondent_ids: list[Union[int, str]],
        n_sections: int,
        m_features_per_section: int,
        seed: int,
        shuffle_features: bool = False
    ) -> dict[Union[int, str], RespondentProfile]:
        """Generate profiles for multiple respondents."""
        return {
            rid: self.generate_profile(
                rid, n_sections, m_features_per_section, seed, shuffle_features
            )
            for rid in respondent_ids
        }
    
    def generate_richness_levels(
        self,
        respondent_id: Union[int, str],
        levels: list[tuple[int, int]],
        seed: int,
        shuffle_features: bool = False
    ) -> dict[str, RespondentProfile]:
        """
        Generate sparse, medium, and rich profiles for a respondent.
        
        Parameters
        ----------
        respondent_id : int or str
        levels : list of (n_sections, m_features) tuples
            E.g., [(2, 2), (3, 3), (4, 4)] for sparse/medium/rich
        seed : int
        shuffle_features : bool, default False
            If True, shuffle feature order within each profile
        
        Returns
        -------
        dict mapping level names to profiles
            Keys are 'level_0', 'level_1', etc.
        """
        # Sort levels to ensure proper nesting (smallest first)
        sorted_levels = sorted(levels, key=lambda x: x[0] * x[1])
        
        profiles = {}
        for i, (n_sec, m_feat) in enumerate(sorted_levels):
            profiles[f'level_{i}'] = self.generate_profile(
                respondent_id, n_sec, m_feat, seed, shuffle_features
            )
        
        return profiles

    # -------------------------------------------------------------------------
    # Target question handling
    # -------------------------------------------------------------------------
    
    def _detect_country_specific_targets(
        self, 
        target_codes: list[str],
        threshold: float = 0.5
    ) -> set[str]:
        """
        Automatically detect which targets should have country-specific options.
        
        A target is considered country-specific if the average number of unique
        response values per country is much smaller than the global count.
        
        Parameters
        ----------
        target_codes : list[str]
            Target question codes to check
        threshold : float, default 0.5
            If (avg_per_country / global_count) < threshold, mark as country-specific.
            Lower threshold = more conservative (only flag obvious cases like party vote).
            
        Returns
        -------
        set[str]
            Target codes that should use country-specific options
        """
        if self.country_col is None:
            return set()
        
        country_specific = set()
        countries = self.survey_data[self.country_col].dropna().unique()
        
        if len(countries) <= 1:
            return set()
        
        for code in target_codes:
            if code not in self.survey_data.columns:
                continue
            
            # Get global unique valid values
            global_values = self.survey_data[code].dropna().unique()
            
            # Filter to non-missing values
            if hasattr(self, '_target_questions') and code in self._target_questions:
                target = self._target_questions[code]
                global_valid = [
                    v for v in global_values 
                    if not self._is_missing_value_label(target.get_label_for_value(v))
                ]
            else:
                # Pre-target setup: just count raw values
                global_valid = list(global_values)
            
            n_global = len(global_valid)
            if n_global <= 1:
                continue
            
            # Count unique values per country
            per_country_counts = []
            for country in countries:
                country_mask = self.survey_data[self.country_col] == country
                country_values = self.survey_data.loc[country_mask, code].dropna().unique()
                per_country_counts.append(len(country_values))
            
            avg_per_country = sum(per_country_counts) / len(per_country_counts)
            ratio = avg_per_country / n_global
            
            if ratio < threshold:
                country_specific.add(code)
        
        return country_specific
    
    def set_target_questions(
        self, 
        target_codes: list[str],
        country_specific_targets: Optional[list[str]] = None,
        auto_detect_country_specific: bool = True,
        country_specific_threshold: float = 0.5
    ):
        """
        Set the list of target questions for prediction.
        
        These questions are automatically excluded from the feature pool
        and can be used to generate prediction instances.
        
        Missing/artifact values (as configured in __init__) are automatically
        filtered from the answer options.
        
        Parameters
        ----------
        target_codes : list[str]
            List of question codes that will serve as prediction targets
        country_specific_targets : list[str], optional
            Explicit list of targets that should have country-specific options.
            If None and auto_detect_country_specific=True, will be detected automatically.
            If provided, auto-detection is skipped for these codes.
        auto_detect_country_specific : bool, default True
            If True, automatically detect which targets need country-specific options
            based on response distribution variance across countries.
        country_specific_threshold : float, default 0.5
            Threshold for auto-detection. Lower = more conservative.
            A target is flagged if (avg_options_per_country / global_options) < threshold.
        """
        self._target_questions = {}
        unknown_codes = []
        
        for code in target_codes:
            found = False
            for section, questions in self.metadata.items():
                if code in questions:
                    q_info = questions[code]
                    values_map = q_info.get('values', {})
                    
                    # Filter out missing/artifact values from options
                    valid_options = self._filter_valid_options(values_map)
                    
                    self._target_questions[code] = TargetQuestion(
                        code=code,
                        question=q_info.get('question', q_info.get('description', code)),
                        description=q_info.get('description', ''),
                        section=section,
                        options=valid_options,  # Global options (may be overridden per-country)
                        values_map=values_map
                    )
                    found = True
                    break
            
            if not found:
                unknown_codes.append(code)
        
        if unknown_codes:
            import warnings
            warnings.warn(f"Target question codes not found in metadata: {unknown_codes}")
        
        # Determine country-specific targets
        if country_specific_targets is not None:
            # User provided explicit list
            self._country_specific_targets = set(country_specific_targets)
        elif auto_detect_country_specific and self.country_col:
            # Auto-detect
            self._country_specific_targets = self._detect_country_specific_targets(
                target_codes, threshold=country_specific_threshold
            )
        else:
            self._country_specific_targets = set()
        
        # Validate country_specific_targets
        invalid_country_specific = self._country_specific_targets - set(target_codes)
        if invalid_country_specific:
            import warnings
            warnings.warn(
                f"country_specific_targets contains codes not in target_codes: "
                f"{invalid_country_specific}"
            )
        
        # Build country-specific options lookup if needed
        if self._country_specific_targets and self.country_col:
            self._build_country_specific_options()
        elif self._country_specific_targets and not self.country_col:
            import warnings
            warnings.warn(
                "country_specific_targets specified but no country_col configured. "
                "Country-specific options will not be used."
            )
            self._country_specific_targets = set()
        
        # Automatically add target questions to exclusions
        self.add_exclusions(target_codes)
        
        # Compute semantic similarity exclusions if model is configured
        if self.similarity_model_name:
            self._target_similar_features = self._compute_similarity_exclusions(target_codes)
        else:
            self._target_similar_features = {code: set() for code in target_codes}
        
        # Report
        total_original = sum(
            len(self.metadata[self._feature_to_section[code]][code].get('values', {}))
            for code in self._target_questions
        )
        total_filtered = sum(len(t.options) for t in self._target_questions.values())
        
        print(f"Set {len(self._target_questions)} target questions. "
              f"These are now excluded from the feature pool.")
        if total_original != total_filtered:
            print(f"  Filtered {total_original - total_filtered} missing/artifact options "
                  f"(from {total_original} to {total_filtered} total options)")
        if self._country_specific_targets:
            print(f"  Country-specific options: {list(self._country_specific_targets)} (auto-detected)" 
                  if auto_detect_country_specific and country_specific_targets is None
                  else f"  Country-specific options: {list(self._country_specific_targets)}")
    
    def _build_country_specific_options(self):
        """
        Build lookup table of country-specific valid options for target questions.
        
        Creates: self._country_options[country_value][target_code] = [valid_labels]
        """
        self._country_options = {}
        
        # Get unique countries
        countries = self.survey_data[self.country_col].dropna().unique()
        
        for country in countries:
            self._country_options[country] = {}
            country_mask = self.survey_data[self.country_col] == country
            country_data = self.survey_data[country_mask]
            
            for code in self._country_specific_targets:
                if code not in self._target_questions:
                    continue
                    
                target = self._target_questions[code]
                
                # Get unique raw values for this question in this country
                raw_values = country_data[code].dropna().unique()
                
                # Map to labels and filter missing values
                valid_labels = []
                for raw_val in raw_values:
                    label = target.get_label_for_value(raw_val)
                    if not self._is_missing_value_label(label):
                        valid_labels.append(label)
                
                # Store unique labels (preserving order from metadata where possible)
                # Use metadata order for consistency
                ordered_labels = [
                    opt for opt in target.options if opt in valid_labels
                ]
                self._country_options[country][code] = ordered_labels
        
        # Report statistics
        for code in self._country_specific_targets:
            if code not in self._target_questions:
                continue
            global_count = len(self._target_questions[code].options)
            country_counts = [
                len(self._country_options[c].get(code, []))
                for c in countries
            ]
            avg_country = sum(country_counts) / len(country_counts) if country_counts else 0
            print(f"    {code}: {global_count} global options → avg {avg_country:.1f} per country")
    
    def _get_country_specific_options(
        self, 
        target_code: str, 
        country_value
    ) -> Optional[list[str]]:
        """
        Get country-specific options for a target question.
        
        Returns None if not a country-specific target or country not found.
        """
        if target_code not in self._country_specific_targets:
            return None
        
        if not hasattr(self, '_country_options'):
            return None
            
        country_opts = self._country_options.get(country_value, {})
        return country_opts.get(target_code)
    
    def get_target_questions(self) -> dict[str, TargetQuestion]:
        """Return the configured target questions."""
        if not hasattr(self, '_target_questions'):
            return {}
        return self._target_questions
    
    def get_target_question(self, code: str) -> Optional[TargetQuestion]:
        """Get a specific target question by code."""
        return self._target_questions.get(code)
    
    # -------------------------------------------------------------------------
    # Prediction instance generation
    # -------------------------------------------------------------------------
    
    def _get_country_value(self, respondent_data: pd.Series) -> Optional[Union[int, str]]:
        """Extract country value from respondent data using configured country_col."""
        if self.country_col is None:
            return None
        return respondent_data.get(self.country_col)
    
    def generate_prediction_instance(
        self,
        respondent_id: Union[int, str],
        target_code: str,
        n_sections: int,
        m_features_per_section: int,
        seed: int,
        shuffle_features: bool = False,
        skip_missing_targets: bool = True
    ) -> Optional[PredictionInstance]:
        """
        Generate a single prediction instance combining profile + target.
        
        This is the main method for creating instances to feed to LLM evaluation.
        
        Parameters
        ----------
        respondent_id : int or str
            Respondent identifier
        target_code : str
            Code of the target question (must be in set_target_questions)
        n_sections : int
            Number of sections for profile
        m_features_per_section : int
            Features per section
        seed : int
            Random seed
        shuffle_features : bool, default False
            If True, shuffle feature order in profile
        skip_missing_targets : bool, default True
            If True, return None when respondent's target answer is missing/artifact.
            If False, include even if answer is a missing value.
            
        Returns
        -------
        PredictionInstance or None
            Complete instance ready for LLM evaluation, or None if target is missing
        """
        if not hasattr(self, '_target_questions') or target_code not in self._target_questions:
            raise ValueError(
                f"Target question '{target_code}' not found. "
                f"Call set_target_questions() first with this code."
            )
        
        target = self._target_questions[target_code]
        
        # Get respondent's data
        respondent_data = self._get_respondent_data(respondent_id)
        
        # Check if target answer is valid (not missing/artifact)
        raw_answer = respondent_data.get(target_code)
        answer_label = target.get_label_for_value(raw_answer)
        
        if skip_missing_targets and self._is_missing_value_label(answer_label):
            return None
        
        # Generate profile with target-specific exclusions
        # (features semantically similar to target are excluded)
        profile = self.generate_profile(
            respondent_id=respondent_id,
            n_sections=n_sections,
            m_features_per_section=m_features_per_section,
            seed=seed,
            shuffle_features=shuffle_features,
            target_code=target_code  # Apply per-target semantic exclusions
        )
        
        # Get country
        country_value = self._get_country_value(respondent_data)
        
        # Get options: use country-specific if available, otherwise global
        options = self._get_country_specific_options(target_code, country_value)
        if options is None:
            options = target.options
        
        # Validate that respondent's answer is in the options
        # (it should be, but this catches edge cases)
        if answer_label not in options and not self._is_missing_value_label(answer_label):
            import warnings
            warnings.warn(
                f"Respondent {respondent_id}: answer '{answer_label}' for {target_code} "
                f"not in options for country {country_value}. Adding it."
            )
            options = options + [answer_label]
        
        # Convert profile features to {question_text: answer_label} format
        features_dict = {}
        for code, info in profile.features.items():
            question_text = info.get('question', info.get('description', code))
            features_dict[question_text] = info['value_label']
        
        return PredictionInstance(
            id=respondent_id,
            country=country_value,
            features=features_dict,
            target_question=target.question,
            target_code=target_code,
            answer=answer_label,
            answer_raw=raw_answer,
            options=options,
            survey=self.survey or "",
            profile_config=profile.config,
            target_section=target.section
        )
    
    def generate_prediction_instance_from_profile(
        self,
        profile: RespondentProfile,
        target_code: str,
        skip_missing_targets: bool = True
    ) -> Optional[PredictionInstance]:
        """
        Generate prediction instance from an existing profile.
        
        Useful when you want to reuse the same profile across multiple targets.
        
        Parameters
        ----------
        profile : RespondentProfile
            Pre-generated profile
        target_code : str
            Target question code
        skip_missing_targets : bool, default True
            If True, return None when respondent's target answer is missing/artifact.
            
        Returns
        -------
        PredictionInstance or None
        """
        if not hasattr(self, '_target_questions') or target_code not in self._target_questions:
            raise ValueError(f"Target question '{target_code}' not found.")
        
        target = self._target_questions[target_code]
        respondent_data = self._get_respondent_data(profile.respondent_id)
        
        raw_answer = respondent_data.get(target_code)
        answer_label = target.get_label_for_value(raw_answer)
        
        # Check if target answer is valid
        if skip_missing_targets and self._is_missing_value_label(answer_label):
            return None
        
        country_value = self._get_country_value(respondent_data)
        
        # Get options: use country-specific if available, otherwise global
        options = self._get_country_specific_options(target_code, country_value)
        if options is None:
            options = target.options
        
        # Validate that respondent's answer is in the options
        if answer_label not in options and not self._is_missing_value_label(answer_label):
            import warnings
            warnings.warn(
                f"Respondent {profile.respondent_id}: answer '{answer_label}' for {target_code} "
                f"not in options for country {country_value}. Adding it."
            )
            options = options + [answer_label]
        
        features_dict = {
            info.get('question', info.get('description', code)): info['value_label']
            for code, info in profile.features.items()
        }
        
        return PredictionInstance(
            id=profile.respondent_id,
            country=country_value,
            features=features_dict,
            target_question=target.question,
            target_code=target_code,
            answer=answer_label,
            answer_raw=raw_answer,
            options=options,
            survey=self.survey or "",
            profile_config=profile.config,
            target_section=target.section
        )
    
    def generate_all_instances_for_respondent(
        self,
        respondent_id: Union[int, str],
        n_sections: int,
        m_features_per_section: int,
        seed: int,
        target_codes: Optional[list[str]] = None,
        shuffle_features: bool = False,
        skip_missing_targets: bool = True
    ) -> list[PredictionInstance]:
        """
        Generate prediction instances for all (or specified) targets for one respondent.
        
        Note: When semantic similarity filtering is enabled, each target gets its own
        profile with target-specific exclusions. When disabled, all targets share
        the same profile.
        
        Parameters
        ----------
        respondent_id : int or str
        n_sections, m_features_per_section, seed : profile params
        target_codes : list[str], optional
            Subset of targets. If None, uses all configured targets.
        shuffle_features : bool, default False
            If True, shuffle feature order in profile
        skip_missing_targets : bool, default True
            If True, skip targets where respondent's answer is missing/artifact.
            
        Returns
        -------
        list[PredictionInstance]
            One instance per valid target question (may be fewer than target_codes
            if some targets have missing answers)
        """
        if target_codes is None:
            target_codes = list(self._target_questions.keys())
        
        instances = []
        
        # Check if we need per-target profiles (semantic similarity enabled)
        has_per_target_exclusions = (
            self.similarity_model_name is not None and 
            any(self._target_similar_features.get(code) for code in target_codes)
        )
        
        if has_per_target_exclusions:
            # Generate separate profile for each target
            for code in target_codes:
                instance = self.generate_prediction_instance(
                    respondent_id, code, n_sections, m_features_per_section,
                    seed, shuffle_features, skip_missing_targets
                )
                if instance is not None:
                    instances.append(instance)
        else:
            # No per-target exclusions: share profile across targets
            profile = self.generate_profile(
                respondent_id, n_sections, m_features_per_section, seed, shuffle_features
            )
            for code in target_codes:
                instance = self.generate_prediction_instance_from_profile(
                    profile, code, skip_missing_targets
                )
                if instance is not None:
                    instances.append(instance)
        
        return instances
    
    def generate_dataset(
        self,
        respondent_ids: list[Union[int, str]],
        n_sections: int,
        m_features_per_section: int,
        seed: int,
        target_codes: Optional[list[str]] = None,
        shuffle_features: bool = False,
        skip_missing_targets: bool = True
    ) -> Iterator[PredictionInstance]:
        """
        Generate full dataset of prediction instances.
        
        Yields instances one at a time (memory efficient for large datasets).
        
        Parameters
        ----------
        respondent_ids : list
            Respondents to include
        n_sections, m_features_per_section, seed : profile params
        target_codes : list[str], optional
            Subset of targets
        shuffle_features : bool, default False
            If True, shuffle feature order in profiles
        skip_missing_targets : bool, default True
            If True, skip instances where respondent's target answer is missing
        
        Yields
        ------
        PredictionInstance
            One instance at a time (only valid instances)
        """
        for rid in respondent_ids:
            instances = self.generate_all_instances_for_respondent(
                rid, n_sections, m_features_per_section, seed, 
                target_codes, shuffle_features, skip_missing_targets
            )
            for instance in instances:
                yield instance
    
    def generate_dataset_as_list(
        self,
        respondent_ids: list[Union[int, str]],
        n_sections: int,
        m_features_per_section: int,
        seed: int,
        target_codes: Optional[list[str]] = None,
        shuffle_features: bool = False,
        skip_missing_targets: bool = True,
        as_dicts: bool = True
    ) -> list:
        """
        Generate full dataset as a list (convenience method).
        
        Parameters
        ----------
        shuffle_features : bool, default False
            If True, shuffle feature order in profiles
        skip_missing_targets : bool, default True
            If True, skip instances where respondent's target answer is missing
        as_dicts : bool
            If True, return list of dicts. If False, return PredictionInstance objects.
            
        Returns
        -------
        list
            List of prediction instances (as dicts or objects)
        """
        instances = list(self.generate_dataset(
            respondent_ids, n_sections, m_features_per_section, seed, 
            target_codes, shuffle_features, skip_missing_targets
        ))
        
        if as_dicts:
            return [inst.to_dict() for inst in instances]
        return instances