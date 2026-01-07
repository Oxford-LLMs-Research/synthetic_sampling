"""
Data Classes for Respondent Profile Generation

This module defines the core data structures used for profile generation
and prediction instances in LLM survey prediction experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Callable

from .formats import get_profile_formatter


@dataclass
class ProfileConfig:
    """Configuration for profile generation."""
    n_sections: int  # Number of thematic sections to sample from
    m_features_per_section: int  # Features to sample per section
    seed: int
    shuffle_features: bool = False  # Whether features are shuffled within profile
    
    @property
    def total_features(self) -> int:
        """Total features excluding always-include features."""
        return self.n_sections * self.m_features_per_section
    
    @property
    def profile_type(self) -> str:
        """
        Generate a profile type identifier from config.
        
        Format: s{n_sections}m{m_features_per_section}
        Examples: 's1m2' (sparse), 's2m4' (medium), 's4m6' (rich)
        """
        return f"s{self.n_sections}m{self.m_features_per_section}"


@dataclass 
class RespondentProfile:
    """A single respondent's profile at a given information richness level."""
    respondent_id: Union[int, str]
    features: dict[str, dict]  # feature_code -> {value, question, description, section}
    config: ProfileConfig
    sections_sampled: list[str]
    always_included: list[str] = field(default_factory=list)
    
    @property
    def feature_codes(self) -> list[str]:
        return list(self.features.keys())
    
    @property
    def n_features(self) -> int:
        return len(self.features)
    
    def to_qa_format(self) -> str:
        """Convert profile to Q&A format for LLM prompting."""
        lines = []
        for code, info in self.features.items():
            q = info.get('question', info.get('description', code))
            v = info['value_label']
            lines.append(f"Q: {q}\nA: {v}")
        return "\n\n".join(lines)
    
    def to_dict(self) -> dict:
        """Serialize profile for storage/logging."""
        return {
            'respondent_id': self.respondent_id,
            'features': self.features,
            'config': {
                'n_sections': self.config.n_sections,
                'm_features_per_section': self.config.m_features_per_section,
                'seed': self.config.seed,
                'profile_type': self.config.profile_type,
            },
            'sections_sampled': self.sections_sampled,
            'always_included': self.always_included
        }


@dataclass
class TargetQuestion:
    """A target question for prediction."""
    code: str  # Question code in the survey (e.g., 'Q35A')
    question: str  # Full question text
    description: str  # Short description
    section: str  # Thematic section this question belongs to
    options: list[str]  # List of answer option labels
    values_map: dict[str, str]  # raw_value -> label mapping
    
    def get_label_for_value(self, raw_value) -> str:
        """Convert raw response value to label."""
        if raw_value in self.values_map:
            return self.values_map[raw_value]
        if str(raw_value) in self.values_map:
            return self.values_map[str(raw_value)]
        try:
            if str(int(raw_value)) in self.values_map:
                return self.values_map[str(int(raw_value))]
        except (ValueError, TypeError):
            pass
        return str(raw_value)


@dataclass
class PredictionInstance:
    """
    A complete instance for LLM prediction: profile + target + ground truth.
    
    This is the final output format that gets fed to the LLM evaluation pipeline.
    
    Key identifiers:
    - example_id: Unique identifier for this specific instance
      Format: {survey}_{respondent_id}_{target_code}_{profile_type}
      Example: "ess_wave_10_12345_Q5A_s2m4"
    - base_id: Shared identifier across profile types for the same respondent+target
      Format: {survey}_{respondent_id}_{target_code}
      Example: "ess_wave_10_12345_Q5A"
    
    This allows:
    - Unique identification of each prediction instance
    - Grouping instances by respondent+target to compare across profile richness levels
    - Easy joining of predictions back to ground truth
    """
    id: Union[int, str]
    country: Optional[Union[int, str]]
    features: dict[str, str]  # question_text -> answer_label
    target_question: str
    target_code: str  # Keep track of the question code for analysis
    answer: str  # Ground truth label
    answer_raw: any  # Raw value for verification
    options: list[str]
    
    # Survey identifier
    survey: str = ""  # Survey identifier (e.g., 'ess_wave_10', 'afrobarometer')
    
    # Metadata for analysis
    profile_config: Optional[ProfileConfig] = None
    target_section: Optional[str] = None
    
    # Computed identifiers (populated in __post_init__)
    example_id: str = field(default="", init=False)
    base_id: str = field(default="", init=False)
    
    def __post_init__(self):
        """Generate example_id and base_id from components."""
        self._generate_ids()
    
    def _generate_ids(self):
        """Generate example_id and base_id from components."""
        # Sanitize respondent_id for use in ID string
        rid_str = str(self.id).replace(' ', '_').replace('/', '_')
        
        # Base ID: shared across profile types
        if self.survey:
            self.base_id = f"{self.survey}_{rid_str}_{self.target_code}"
        else:
            self.base_id = f"{rid_str}_{self.target_code}"
        
        # Example ID: unique per profile type
        if self.profile_config:
            profile_type = self.profile_config.profile_type
            self.example_id = f"{self.base_id}_{profile_type}"
        else:
            self.example_id = self.base_id
    
    def set_survey(self, survey: str):
        """Set survey identifier and regenerate IDs."""
        self.survey = survey
        self._generate_ids()
    
    @property
    def profile_type(self) -> Optional[str]:
        """Get profile type string from config."""
        if self.profile_config:
            return self.profile_config.profile_type
        return None
    
    def to_dict(self) -> dict:
        """
        Convert to dict format for LLM prompting/evaluation.
        
        Includes example_id and base_id for tracking.
        """
        return {
            'example_id': self.example_id,
            'base_id': self.base_id,
            'survey': self.survey,
            'id': self.id,
            'country': self.country,
            'questions': self.features,
            'target_question': self.target_question,
            'target_code': self.target_code,
            'answer': self.answer,
            'options': self.options,
            'profile_type': self.profile_type,
        }
    
    def to_full_dict(self) -> dict:
        """Return full dict including all metadata for analysis."""
        d = self.to_dict()
        d['_metadata'] = {
            'answer_raw': self.answer_raw,
            'target_section': self.target_section,
            'profile_config': {
                'n_sections': self.profile_config.n_sections,
                'm_features_per_section': self.profile_config.m_features_per_section,
                'seed': self.profile_config.seed,
                'profile_type': self.profile_config.profile_type,
            } if self.profile_config else None
        }
        return d
    
    def format_profile(
        self, 
        format_spec: Union[str, Callable[[str, str], str]] = 'colon',
        separator: str = '\n'
    ) -> str:
        """
        Format the profile features as a string for LLM prompting.
        
        Parameters
        ----------
        format_spec : str or callable, default 'colon'
            Either a preset name or a custom callable(question, answer) -> str.
            
            Presets:
            - 'qa': "Q: {question}\\nA: {answer}"
            - 'interview': "Interviewer: {question}\\nRespondent: {answer}"
            - 'bullet': "- {question}: {answer}"
            - 'colon': "{question}: {answer}"
            - 'arrow': "{question} → {answer}"
            - 'brackets': "[{question}] {answer}"
            - 'xml': "<question>...</question>\\n<answer>...</answer>"
            - 'json': '{"q": "...", "a": "..."}'
            - 'narrative': 'When asked "{question}", they answered "{answer}".'
            - 'card': "{question} | {answer}"
            
        separator : str, default '\\n'
            String to join formatted feature lines.
            Use '\\n\\n' for double-spacing, ', ' for inline, etc.
            
        Returns
        -------
        str
            Formatted profile string ready for LLM prompt
        """
        formatter = get_profile_formatter(format_spec)
        lines = [formatter(q, a) for q, a in self.features.items()]
        return separator.join(lines)
    
    def format_target(
        self,
        include_options: bool = True,
        options_format: str = 'list'
    ) -> str:
        """
        Format the target question for LLM prompting.
        
        Parameters
        ----------
        include_options : bool, default True
            Whether to include answer options
        options_format : str, default 'list'
            How to format options: 'list' (numbered), 'inline' (comma-separated),
            'bullets' (bulleted list)
            
        Returns
        -------
        str
            Formatted target question string
        """
        result = self.target_question
        
        if include_options and self.options:
            if options_format == 'list':
                opts = '\n'.join(f"{i+1}. {opt}" for i, opt in enumerate(self.options))
                result += f"\n\nOptions:\n{opts}"
            elif options_format == 'inline':
                result += f"\n\nOptions: {', '.join(self.options)}"
            elif options_format == 'bullets':
                opts = '\n'.join(f"• {opt}" for opt in self.options)
                result += f"\n\nOptions:\n{opts}"
            else:
                raise ValueError(f"Unknown options_format: {options_format}")
        
        return result
    
    def to_prompt(
        self,
        profile_format: Union[str, Callable[[str, str], str]] = 'colon',
        profile_separator: str = '\n',
        include_options: bool = True,
        options_format: str = 'list',
        template: Optional[str] = None
    ) -> str:
        """
        Generate a complete prompt string for LLM evaluation.
        
        Parameters
        ----------
        profile_format : str or callable, default 'colon'
            Format for profile features (see format_profile)
        profile_separator : str, default '\\n'
            Separator between profile lines
        include_options : bool, default True
            Whether to include answer options in target
        options_format : str, default 'list'
            How to format options (see format_target)
        template : str, optional
            Custom template with placeholders: {profile}, {target}, {options}.
            If None, uses a default template.
            
        Returns
        -------
        str
            Complete prompt ready for LLM
        """
        profile_str = self.format_profile(profile_format, profile_separator)
        target_str = self.format_target(include_options, options_format)
        
        if template:
            return template.format(
                profile=profile_str,
                target=self.target_question,
                options='\n'.join(f"{i+1}. {opt}" for i, opt in enumerate(self.options)),
                target_with_options=target_str
            )
        
        # Default template
        return f"""Here is information about a survey respondent:

{profile_str}

Based on this information, please answer:
{target_str}"""