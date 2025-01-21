
import json
import os
import pandas as pd
import logging
from survey_profile_generator import SurveyProfileGenerator
from typing import List, Optional

logger = logging.getLogger(__name__)

def generate_profiles_from_data(data_dir: str,
                                data_file: str,
                                mapping_dir: str,
                                respondent_id: str,
                                output_dir: str,
                                country_field: Optional[str] = None,
                                survey_year: Optional[str] = None,
                                cntry_spec_vars_path: Optional[str] = None,
                                fixed_features: Optional[List[str]] = None,
                                max_sections: int = 4,
                                max_features: int = 3,
                                n_profiles_per_respondent: int = 10
                                ):

    """

    Takes a survey data file and generates text profile narratives for each respondent (row).
        Parameters:
            data_dir (str): Directory containing the dataset.
            data_file (str): Name of the dataset.
            mapping_dir (str): Directory containing nested dictionary mapping of survey questions
            respondent_id (str): identifier for the column in the dataset containing respondent ids
            output_dir (str): Directory where output mappings will be stored.
            country_field (str, optional): The column name for country information.
            survey_year (str, optional): Year of survey, for surveys with multiple waves.
            cntry_spec_vars_path (str): Path to json file containing country-specific variables.
            fixed_features (List[str], optional): List of feature names that are fixed and always included.
            max_sections (int): Maximum number of thematic sections to choose from when generating survey profiles
            max_features (int): Maximum number of features to choose from each section.
            n_profiles_per_respondent (int): Number of profiles per respondent to create.

    """


    #################################
    # 1) load survey-specific mappings
    #################################

    data_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_path)

    with open(mapping_dir, 'r', encoding='utf-8') as file:
        survey_mappings = json.load(file)

    #################################
    # 2) generate profiles
    #################################

    if cntry_spec_vars_path:
        with open(cntry_spec_vars_path, 'r', encoding='utf-8') as file:
            country_specific_variables = json.load(file)
    else:
        country_specific_variables = None

    prof_generator = SurveyProfileGenerator(
        data=df,
        respondent_id=respondent_id,  # survey-specific respondent identifier
        survey_mappings=survey_mappings,
        country_specific_variables=country_specific_variables[survey_year],
        max_sections=max_sections,
        max_features=max_features,
        fixed_features=fixed_features,  # survey-specific variable names
        country_field=country_field,
        random_state=42
    )


    profiles = prof_generator.generate_profiles(num_profiles_per_respondent=n_profiles_per_respondent)


    #################################
    # 3) save profiles
    #################################

    output_file = "profiles.csv"
    output_path = os.path.join(output_dir, output_file)

    os.makedirs(output_dir, exist_ok=True)

    ids = []
    prof_descriptions = []
    for profile in profiles:
        resp_id = profile['respondent_id']
        preamble, question, response = prof_generator.profile_to_text(profile)
        prof_text = f"Profile: \n{preamble}. \nQuestion: {question} \nResponse: {response}"

        ids.append(resp_id)
        prof_descriptions.append(prof_text)

    df = pd.DataFrame({'id': ids, 'text': prof_descriptions})
    df.to_csv(output_path, index=False)

    logger.info(f"Profiles have been saved to {output_path}")
