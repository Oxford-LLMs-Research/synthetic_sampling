from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch


@dataclass
class EvaluationConfig:
    """
    Configuration class for the LLM evaluation pipeline.

    Attributes:
        question_ids (List[str]): List of question IDs to evaluate.
        special_codes (Dict[str, Optional[str]]): Mapping of special codes to their replacements or None to drop.
        profile_prompt_template (str): Template for generating profile prompts.
        model (torch.nn.Module): The language model to evaluate.
        tokenizer (Any): The tokenizer for the language model.
        device (torch.device): The device (CPU/GPU) to run the model on.
        batch_size (int): Batch size for evaluation. Defaults to 32.
    """
    question_ids: List[str]
    special_codes: Dict[str, Optional[str]]  # Map from code -> replacement or None to drop
    profile_prompt_template: str
    model: torch.nn.Module
    tokenizer: Any  # Replace with the actual type if known
    device: torch.device
    batch_size: int = 32  # Increased batch size for better GPU utilization


# Default configuration
DEFAULT_CONFIG = EvaluationConfig(
    question_ids=["Q17", "Q29", "Q206"],
    special_codes={
        "No answer": None,
        "Not asked": None,
        "Don´t know": 'Don´t know',
        "Missing; Not available": None,
        "Missing; Unknown": None,
        "Don't know": 'Don´t know',
        "-5": None
    },
    profile_prompt_template=(
        "Imagine you are a {Q262}-year old {Q260} living in {B_COUNTRY}. Your highest education is {Q275}. "
    ),
    model=None,  # To be initialized
    tokenizer=None,  # To be initialized
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=4
)
