import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from .config import EvaluationConfig

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
    replaced_series = series.replace({code: replacement for code, replacement in code_map.items() if replacement is not None})
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

def build_prompt_data_batch(
    df: pd.DataFrame,
    qid: str,
    mapper,
    config,
    survey_mappings: Dict[str, Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """
    Prepare the data required for prompt generation and evaluation for a specific question in batches.

    Args:
        df: The survey DataFrame.
        qid: The question ID (column name) to evaluate.
        mapper: An instance of the Mapper class for value mapping.
        config: The EvaluationConfig instance containing configuration parameters.
        survey_mappings: Nested dictionary containing survey sections and questions.

    Returns:
        A list of batches, where each batch is a list of dictionaries, each containing:
            - 'chat': A chat-like structure for the respondent.
            - 'label_options': List of possible answer labels.
            - 'true_label': The respondent's actual answer.
    """
    if qid not in df.columns:
        print(f"Warning: {qid} not in DataFrame columns. Skipping.")
        return []

    # 1. Replace special codes
    col_series = replace_special_codes(df[qid], config.special_codes)

    # 2. Filter out rows that are now NaN
    valid_mask = col_series.notna()
    sub_df = df[valid_mask].copy()
    sub_df[qid] = col_series[valid_mask]

    if sub_df.empty:
        print(f"No valid responses for {qid}. Skipping.")
        return []

    # 3. Gather label options
    label_options = sorted(sub_df[qid].unique().tolist())

    # 4. Retrieve question text
    question_text = get_question_text_from_mapping(survey_mappings, qid)
    question_prompt = f"Please answer the following question:\n{question_text}"

    # 5. Build chat prompts
    results = []
    system_content = (
        "You are a helpful AI assistant for public opinion research. "
        "You are skillful at using your knowledge to make good judgment "
        "about people's preferences when given some background information."
    )

    for _, row in sub_df.iterrows():
        # Fill the profile prompt
        filled_profile = mapper.fill_prompt(row, config.profile_prompt_template)
        user_content = filled_profile + "\n" + question_prompt

        # Construct chat format
        chat = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        results.append({
            "chat": chat,
            "label_options": label_options,
            "true_label": row[qid]
        })

    # Split results into batches
    batched_results = [results[i:i + config.batch_size] for i in range(0, len(results), config.batch_size)]
    return batched_results
