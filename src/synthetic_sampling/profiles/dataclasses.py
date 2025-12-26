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
                'seed': self.config.seed
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
    Matches the structure you specified:
    {
        'id': respondent_id,
        'country': country_value,
        'questions': {question_text: answer_label, ...},  # features
        'target_question': question_text,
        'answer': ground_truth_label,
        'options': [option1, option2, ...]
    }
    """
    id: Union[int, str]
    country: Optional[Union[int, str]]
    features: dict[str, str]  # question_text -> answer_label
    target_question: str
    target_code: str  # Keep track of the question code for analysis
    answer: str  # Ground truth label
    answer_raw: any  # Raw value for verification
    options: list[str]
    
    # Metadata for analysis
    profile_config: Optional[ProfileConfig] = None
    target_section: Optional[str] = None
    
    def to_dict(self) -> dict:
        """
        Convert to the exact dict format you specified.
        
        Returns the clean format for LLM prompting/evaluation.
        """
        return {
            'id': self.id,
            'country': self.country,
            'questions': self.features,
            'target_question': self.target_question,
            'answer': self.answer,
            'options': self.options
        }
    
    def to_full_dict(self) -> dict:
        """Return full dict including metadata for analysis."""
        d = self.to_dict()
        d['_metadata'] = {
            'target_code': self.target_code,
            'answer_raw': self.answer_raw,
            'target_section': self.target_section,
            'profile_config': {
                'n_sections': self.profile_config.n_sections,
                'm_features_per_section': self.profile_config.m_features_per_section,
                'seed': self.profile_config.seed
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
            
        Examples
        --------
        >>> instance.format_profile('bullet')
        '- How old are you?: 30-45\\n- What is your gender?: Female\\n...'
        
        >>> instance.format_profile('qa', separator='\\n\\n')
        'Q: How old are you?\\nA: 30-45\\n\\nQ: What is your gender?\\nA: Female\\n\\n...'
        
        >>> instance.format_profile(lambda q, a: f"* {q} = {a}")
        '* How old are you? = 30-45\\n* What is your gender? = Female\\n...'
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
            
        Examples
        --------
        >>> prompt = instance.to_prompt(profile_format='bullet')
        >>> print(prompt)
        Here is information about a survey respondent:
        
        - How old are you?: 30-45
        - What is your gender?: Female
        ...
        
        Based on this information, please answer:
        Which party would you vote for?
        
        Options:
        1. Democrats
        2. Republicans
        ...
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

