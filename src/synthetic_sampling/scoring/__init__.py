"""Completions-based scoring: label_num primary, echo + PMI retained."""

from .arms import DEFAULT_ARMS, KEPT_ARMS, REPLICATE, score_arm, score_echo, score_labels
from .client import make_session, post_with_retries
from .prompts import PROMPT_TEMPLATE, build_prompt, render_profile
from .runner import parse_arms, run_scoring

__all__ = [
    "DEFAULT_ARMS",
    "KEPT_ARMS",
    "REPLICATE",
    "PROMPT_TEMPLATE",
    "build_prompt",
    "render_profile",
    "make_session",
    "post_with_retries",
    "score_arm",
    "score_echo",
    "score_labels",
    "parse_arms",
    "run_scoring",
]
