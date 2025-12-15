"""
Profile Format Presets for LLM Survey Prediction Experiments

This module provides preset formatters for converting respondent profiles
into various text formats for LLM prompting. Each formatter takes a
(question, answer) pair and returns a formatted string.
"""

import json
from typing import Callable, Union


# =============================================================================
# Format Functions
# =============================================================================

def _format_qa(question: str, answer: str) -> str:
    """Q&A format: 'Q: ... A: ...'"""
    return f"Q: {question}\nA: {answer}"


def _format_interview(question: str, answer: str) -> str:
    """Interview format: 'Interviewer: ... Respondent: ...'"""
    return f"Interviewer: {question}\nRespondent: {answer}"


def _format_bullet(question: str, answer: str) -> str:
    """Bullet format: '- Question: Answer'"""
    return f"- {question}: {answer}"


def _format_colon(question: str, answer: str) -> str:
    """Simple colon format: 'Question: Answer'"""
    return f"{question}: {answer}"


def _format_arrow(question: str, answer: str) -> str:
    """Arrow format: 'Question → Answer'"""
    return f"{question} → {answer}"


def _format_brackets(question: str, answer: str) -> str:
    """Bracketed format: '[Question] Answer'"""
    return f"[{question}] {answer}"


def _format_xml(question: str, answer: str) -> str:
    """XML-like format: '<question>Q</question><answer>A</answer>'"""
    return f"<question>{question}</question>\n<answer>{answer}</answer>"


def _format_json_line(question: str, answer: str) -> str:
    """JSON line format: '{"q": "...", "a": "..."}'"""
    return json.dumps({"q": question, "a": answer})


def _format_narrative(question: str, answer: str) -> str:
    """Narrative format: 'When asked "Q", they answered "A".'"""
    return f'When asked "{question}", they answered "{answer}".'


def _format_profile_card(question: str, answer: str) -> str:
    """Profile card format: 'Question | Answer'"""
    return f"{question} | {answer}"


# =============================================================================
# Format Registry
# =============================================================================

# Registry of preset formatters
PROFILE_FORMATS: dict[str, Callable[[str, str], str]] = {
    'qa': _format_qa,
    'interview': _format_interview,
    'bullet': _format_bullet,
    'colon': _format_colon,
    'arrow': _format_arrow,
    'brackets': _format_brackets,
    'xml': _format_xml,
    'json': _format_json_line,
    'narrative': _format_narrative,
    'card': _format_profile_card,
}


# =============================================================================
# Apply formatter to a profile
# =============================================================================

def get_profile_formatter(
    format_spec: Union[str, Callable[[str, str], str]]
) -> Callable[[str, str], str]:
    """
    Get a profile formatter from a preset name or custom callable.
    
    Parameters
    ----------
    format_spec : str or callable
        Either a preset name ('qa', 'interview', 'bullet', 'colon', 'arrow',
        'brackets', 'xml', 'json', 'narrative', 'card') or a custom callable
        that takes (question, answer) and returns a formatted string.
        
    Returns
    -------
    callable
        Formatter function (question, answer) -> str
        
    Examples
    --------
    >>> fmt = get_profile_formatter('bullet')
    >>> fmt("How old are you?", "30-45")
    '- How old are you?: 30-45'
    
    >>> fmt = get_profile_formatter(lambda q, a: f"** {q} ** => {a}")
    >>> fmt("How old are you?", "30-45")
    '** How old are you? ** => 30-45'
    """
    if callable(format_spec):
        return format_spec
    
    if format_spec not in PROFILE_FORMATS:
        available = ', '.join(sorted(PROFILE_FORMATS.keys()))
        raise ValueError(
            f"Unknown format preset '{format_spec}'. "
            f"Available presets: {available}. "
            f"Or pass a custom callable(question, answer) -> str."
        )
    
    return PROFILE_FORMATS[format_spec]


def list_profile_formats() -> dict[str, str]:
    """
    List available profile format presets with examples.
    
    Returns
    -------
    dict[str, str]
        Preset name -> example output
    """
    example_q = "How old are you?"
    example_a = "30-45"
    return {name: fmt(example_q, example_a) for name, fmt in PROFILE_FORMATS.items()}

