import re
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from config import EvaluationConfig

class Mapper:
    """
    A class to map survey responses to their corresponding values based on a mapping dictionary.

    Attributes:
        survey_mappings (Dict[str, Dict[str, Any]]): The survey mappings dictionary.
        feature_to_section (Dict[str, str]): A mapping from feature names to their sections for quick lookup.
    """

    def __init__(self, survey_mappings: Dict[str, Dict[str, Any]]):
        """
        Initialize the Mapper with survey mappings.

        Args:
            survey_mappings (Dict[str, Dict[str, Any]]): The survey mappings dictionary.
        """
        self.survey_mappings = survey_mappings
        self.feature_to_section = {
            feature: section
            for section, features in self.survey_mappings.items()
            for feature in features
        }

    def map_value(self, feature_name: str, value) -> str:
        """
        Map a feature value to its corresponding mapped value.

        Args:
            feature_name (str): The name of the feature to map.
            value: The value to map.

        Returns:
            str: The mapped value or the original value if no mapping is found.
        """
        section = self.feature_to_section.get(feature_name)
        if not section:
            return str(value)  # Feature not found in mappings

        feature_mapping = self.survey_mappings[section].get(feature_name)
        if not feature_mapping:
            return str(value)  # Feature mapping not found

        values_mapping = feature_mapping.get('values', {})
        if pd.isnull(value):
            return "Missing"

        if isinstance(value, float) and value.is_integer():
            value_key = str(int(value))
        elif isinstance(value, (int, np.integer)):
            value_key = str(value)
        else:
            value_key = str(value)

        return values_mapping.get(value_key, str(value))

    def fill_prompt(self, respondent: pd.Series, prompt_template: str) -> str:
        """
        Fill a prompt template with respondent data.

        Args:
            respondent (pd.Series): A row of respondent data.
            prompt_template (str): The prompt template with placeholders.

        Returns:
            str: The filled prompt.
        """
        placeholders = {}
        placeholder_pattern = re.compile(r"\{(\w+)\}")
        placeholder_names = placeholder_pattern.findall(prompt_template)

        for placeholder in placeholder_names:
            if placeholder in respondent:
                value = respondent[placeholder]
                if placeholder in ['agea']:
                    placeholders[placeholder] = str(int(value)) if pd.notnull(value) else f"unknown {placeholder}"
                else:
                    mapped_value = self.map_value(placeholder, value)
                    placeholders[placeholder] = mapped_value
            else:
                placeholders[placeholder] = "Unknown"

        filled_prompt = prompt_template.format(**placeholders)
        return filled_prompt


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply value mappings to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to apply mappings to.
        mapping (Dict[str, Dict[str, Any]]): The mapping dictionary.

    Returns:
        pd.DataFrame: The DataFrame with mapped values.
    """
    for category, variables in mapping.items():
        for col, details in variables.items():
            if col in df.columns:
                value_mapping = details.get("values", {})
                mapped_column = df[col].astype(str).map(value_mapping)
                df[col] = mapped_column.where(mapped_column.notna(), df[col].astype(str))
    return df


def replace_special_codes(series: pd.Series, code_map: Dict[str, Optional[str]]) -> pd.Series:
    """
    Replace or exclude special codes in a pandas Series based on a mapping.

    Args:
        series (pd.Series): The pandas Series to process.
        code_map (Dict[str, Optional[str]]): A dictionary mapping codes to their replacements.
                                             If a code maps to `None`, rows with that code are excluded.

    Returns:
        pd.Series: A pandas Series with replacements made and specified codes excluded.
    """
    drop_codes = [code for code, replacement in code_map.items() if replacement is None]
    drop_mask = series.isin(drop_codes)
    replaced_series = series.replace(
        {code: replacement for code, replacement in code_map.items() if replacement is not None})
    replaced_series[drop_mask] = np.nan
    return replaced_series


def get_question_text_from_mapping(survey_mappings: Dict[str, Dict[str, Any]], question_id: str) -> str:
    """
    Retrieve the question text for a given question ID from the survey mappings.

    Args:
        survey_mappings (Dict[str, Dict[str, Any]]): Nested dictionary containing survey sections and questions.
        question_id (str): The question ID to retrieve text for.

    Returns:
        str: The question text if found, else a default message.
    """
    for category, variables in survey_mappings.items():
        if question_id in variables:
            q_info = variables[question_id]
            return q_info.get("question", f"Question text not found for {question_id}")
    return f"No text found for {question_id}"


class SurveyDataset(Dataset):
    """
    A Dataset that converts a survey DataFrame into evaluation samples.
    Each sample is a dictionary with:
      - "chat": A chat structure (list of dicts) for the prompt.
      - "true_label": The respondent's answer (mapped to text).

    All samples from the same question have the same candidate label options,
    which are stored as an attribute.
    """

    def __init__(self, df: pd.DataFrame, qid: str, mapper: Mapper,
                 config: EvaluationConfig, survey_mappings: Dict[str, Dict[str, Any]]):
        # Replace special codes and filter out invalid responses.
        col_series = replace_special_codes(df[qid], config.special_codes)
        valid_mask = col_series.notna()
        sub_df = df[valid_mask].copy()

        # Map the target values from numeric to text using the mapper.
        sub_df[qid] = sub_df[qid].apply(lambda x: mapper.map_value(qid, x))

        if sub_df.empty:
            raise ValueError(f"No valid responses for question {qid}.")

        self.label_options = sorted(sub_df[qid].unique().tolist())

        # Construct dialog prompt
        question_text = get_question_text_from_mapping(survey_mappings, qid)
        question_prompt = f"Please answer the following question:\n{question_text}"
        system_content = (
            "You are a helpful AI assistant for public opinion research. "
            "You are skillful at using your knowledge to make good judgment "
            "about people's preferences when given some background information."
        )
        self.samples = []
        for _, row in sub_df.iterrows():
            filled_profile = mapper.fill_prompt(row, config.profile_prompt_template)
            user_content = filled_profile + "\n" + question_prompt
            chat = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            self.samples.append({
                "chat": chat,
                "true_label": row[qid]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def simple_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    A simple collate function that simply returns the batch (a list of sample dictionaries).
    """
    return batch
