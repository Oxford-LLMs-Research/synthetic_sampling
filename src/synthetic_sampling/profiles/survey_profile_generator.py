
import random
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class SurveyProfileGenerator:
    def __init__(
        self,
        data: pd.DataFrame,
        respondent_id: str,
        survey_mappings: Dict[str, Dict[str, Any]],
        max_sections: int = 3,
        max_features: int = 3,
        fixed_features: Optional[List[str]] = None,
        country_field: Optional[str] = None,
        country_specific_variables: Optional[Dict[str, Dict[str, str]]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initializes the SurveyProfileGenerator.

        Parameters:
            data (pd.DataFrame): The survey dataset.
            respondent_id (str): The column name for respondent IDs.
            survey_mappings (dict): Nested dictionary mapping of survey questions.
            max_sections (int): Maximum number of sections to randomly select.
            max_features (int): Maximum number of features to randomly select per section.
            fixed_features (List[str], optional): List of feature names that are fixed and always included.
            country_field (str, optional): The column name for country information.
            country_specific_variables (dict, optional): Dictionary of country-specific variables.
            random_state (int, optional): Seed for random number generators to ensure reproducibility.
        """
        self.data = data
        self.respondent_id = respondent_id
        self.survey_mappings = survey_mappings
        self.max_sections = max_sections
        self.max_features = max_features
        self.fixed_features = fixed_features or []
        self.country_field = country_field
        self.country_specific_variables = country_specific_variables or {}

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self.feature_to_section = self._build_feature_to_section_mapping()
        self.generic_to_actual_features = self._build_generic_to_actual_features_mapping()

        if not self.country_field or not self.country_specific_variables:
            self.adjusted_features_cache = {
                section: list(features.keys())
                for section, features in self.survey_mappings.items()
            }
            self.feature_to_countries = {}
        else:
            self.adjusted_features_cache = None
            self.feature_to_countries = self._build_feature_to_countries_mapping()

    def _build_feature_to_section_mapping(self) -> Dict[str, str]:
        """Builds a mapping from feature names to their respective sections."""
        mapping = {}
        for section, features in self.survey_mappings.items():
            for feature in features:
                mapping[feature] = section
        return mapping

    def _build_generic_to_actual_features_mapping(self) -> Dict[str, str]:
        """Builds a mapping from generic feature names to actual variable names per country."""
        mapping = {}
        for generic_feature, country_vars in self.country_specific_variables.items():
            for country_code, actual_feature in country_vars.items():
                mapping[actual_feature] = generic_feature
        return mapping

    def _build_feature_to_countries_mapping(self) -> Dict[str, set]:
        """Builds a reverse mapping from feature to countries."""
        feature_countries = {}
        for feature_type, country_vars in self.country_specific_variables.items():
            for country, feature in country_vars.items():
                feature_countries.setdefault(feature, set()).add(country)
        return feature_countries

    def select_random_sections(self, available_sections: List[str]) -> List[str]:
        """Selects a random subset of sections up to max_sections."""
        num_sections = min(self.max_sections, len(available_sections))
        return random.sample(available_sections, num_sections) if num_sections > 0 else []

    def adjust_features_for_country(self, features: List[str], respondent_country: Any) -> List[str]:
        """Adjusts feature list based on respondent's country."""
        if not self.feature_to_countries:
            return features
        return [
            feature for feature in features
            if not self.feature_to_countries.get(feature) or respondent_country in self.feature_to_countries[feature]
        ]

    def select_features_in_sections(self, sections: List[str], respondent_country: Any) -> List[str]:
        """Selects a random subset of features from the given sections."""
        selected_features = []
        for section in sections:
            if self.adjusted_features_cache:
                features_in_section = self.adjusted_features_cache.get(section, [])
            else:
                features_in_section = self.adjust_features_for_country(
                    list(self.survey_mappings[section].keys()), respondent_country
                )
            num_features = min(self.max_features, len(features_in_section))
            if num_features > 0:
                selected_features.extend(random.sample(features_in_section, num_features))
        return selected_features

    def filter_valid_features(self, features: List[str], respondent: pd.Series) -> List[str]:
        """Filters out features with invalid or missing responses."""
        skip_values = {
            "not applicable", "not asked", "Not asked in this country",
            "Not asked in survey", "-3", "-4", "-5", "-specific list of codes in Annex",
            "List of codes in Annex", "Missing, Not available", "Not asked",
            "No answer ", "Missing; Not available", "Not asked "
        }
        valid_features = []
        for feature in features:
            if feature not in respondent:
                continue
            value = respondent[feature]
            if pd.isnull(value):
                continue
            section = self.feature_to_section.get(feature)
            if not section:
                continue
            feature_mapping = self.survey_mappings.get(section, {}).get(feature)
            if not feature_mapping:
                continue
            values_mapping = feature_mapping.get('values', {})
            if isinstance(value, (int, float)) and not pd.isnull(value):
                if isinstance(value, float) and value.is_integer():
                    value_key = str(int(value))
                else:
                    value_key = str(int(value)) if isinstance(value, (int, np.integer)) else str(value).strip()
            else:
                value_key = str(value).strip()
            value_text = values_mapping.get(value_key, str(value))
            if value_text.strip().lower() not in skip_values:
                valid_features.append(feature)
        return valid_features

    def create_random_profile(self, respondent: pd.Series, available_sections: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Creates a single random profile for a respondent."""
        profile = {'respondent_id': respondent[self.respondent_id]}
        respondent_country = respondent.get(self.country_field) if self.country_field else None

        # Add fixed features
        for feature in self.fixed_features:
            actual_feature = self._get_actual_feature(feature, respondent_country)
            if actual_feature and actual_feature in respondent:
                profile[actual_feature] = respondent[actual_feature]

        # Select random sections and features
        random_sections = self.select_random_sections(available_sections)
        selected_features = self.select_features_in_sections(random_sections, respondent_country)
        selected_features = [f for f in selected_features if f not in self.fixed_features]

        if not selected_features:
            return profile, random_sections

        filtered_features = self.filter_valid_features(selected_features, respondent)
        if not filtered_features:
            return profile, random_sections

        response_feature = self._select_response_feature(filtered_features, respondent_country, respondent)
        if not response_feature:
            return profile, random_sections

        # Add predictor features
        for feature in filtered_features:
            if feature != response_feature and feature in respondent:
                profile[feature] = respondent[feature]

        # Add the response feature
        if response_feature in respondent:
            profile['response_feature'] = respondent[response_feature]
            profile['response_feature_name'] = response_feature

        return profile, random_sections

    def _get_actual_feature(self, feature: str, country: Any) -> Optional[str]:
        """Retrieves the actual feature name based on country-specific variables."""
        if feature in self.country_specific_variables:
            return self.country_specific_variables[feature].get(country)
        return feature

    def _select_response_feature(
        self,
        filtered_features: List[str],
        respondent_country: Any,
        respondent: pd.Series
    ) -> Optional[str]:
        """Selects a response feature either from filtered features or all available features."""
        if random.random() < 0.5:
            return random.choice(filtered_features) if filtered_features else None

        # Select from all available features excluding filtered and fixed features
        all_features = set()
        if self.adjusted_features_cache:
            for features in self.adjusted_features_cache.values():
                all_features.update(features)
        else:
            for section in self.survey_mappings:
                features = self.adjust_features_for_country(
                    list(self.survey_mappings[section].keys()), respondent_country
                )
                all_features.update(features)

        response_pool = all_features - set(filtered_features) - set(self.fixed_features)
        response_pool = self.filter_valid_features(list(response_pool), respondent)
        if not response_pool:
            return random.choice(filtered_features) if filtered_features else None
        return random.choice(response_pool) if response_pool else None

    def generate_profiles(self, num_profiles_per_respondent: int) -> List[Dict[str, Any]]:
        """Generates multiple unique profiles for each respondent."""
        profiles = []
        seen_profiles = set()

        for _, respondent in self.data.iterrows():
            attempts = 0
            profiles_generated = 0
            max_attempts = num_profiles_per_respondent * 10  # Adjustable

            remaining_sections = list(self.survey_mappings.keys())

            while profiles_generated < num_profiles_per_respondent and attempts < max_attempts:
                if not remaining_sections:
                    remaining_sections = list(self.survey_mappings.keys())

                profile, sections_used = self.create_random_profile(respondent, remaining_sections)
                attempts += 1

                # Validate response feature
                if "response_feature" not in profile or pd.isnull(profile["response_feature"]):
                    continue

                # Create a unique signature for the profile
                signature = tuple(sorted(
                    (k, v) for k, v in profile.items()
                    if k not in {'respondent_id', 'response_feature', 'response_feature_name'}
                ))
                if signature in seen_profiles:
                    print(f"Duplicate profile encountered for respondent {respondent[self.respondent_id]}.")
                    continue

                seen_profiles.add(signature)
                profiles.append(profile)
                profiles_generated += 1

                # Remove used sections
                for section in sections_used:
                    if section in remaining_sections:
                        remaining_sections.remove(section)

        return profiles

    def profile_to_text(self, profile: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Converts a profile dictionary into a text representation.

        Returns:
            Tuple containing preamble, question, and response_text.
        """
        lines = []

        # Add predictor features
        for feature, value in profile.items():
            if feature in {'respondent_id', 'response_feature', 'response_feature_name'}:
                continue
            if pd.isnull(value):
                continue
            description, mapped_value = self.map_value(feature, value)
            lines.append(f"{description}: {mapped_value}")

        # Handle response feature
        response_feature_name = profile.get('response_feature_name')
        response_feature_value = profile.get('response_feature')

        if response_feature_name and response_feature_value is not None:
            description, mapped_value = self.map_value(response_feature_name, response_feature_value)
            section = self.feature_to_section.get(response_feature_name, {})
            feature_mapping = self.survey_mappings.get(section, {}).get(response_feature_name, {})
            question = feature_mapping.get('question', f"Please answer the following question about {description}:")
            response_text = mapped_value
        else:
            question = ""
            response_text = ""

        preamble = '\n'.join(lines)
        return preamble, question, response_text

    def map_value(self, feature_name: str, value: Any) -> Tuple[str, str]:
        """
        Maps a feature's value to its description and textual value.

        Returns:
            Tuple containing description and mapped_value.
        """
        section = self.feature_to_section.get(feature_name)
        if not section:
            return feature_name, str(value)

        feature_mapping = self.survey_mappings.get(section, {}).get(feature_name)
        if not feature_mapping:
            return feature_name, str(value)

        generic_feature_name = self.generic_to_actual_features.get(feature_name, feature_name)
        description = feature_mapping.get('description', generic_feature_name)
        values_mapping = feature_mapping.get('values', {})

        if pd.isnull(value):
            return description, "Missing"

        # Determine the key for value mapping
        if isinstance(value, float) and value.is_integer():
            value_key = str(int(value))
        elif isinstance(value, (int, np.integer)):
            value_key = str(value)
        else:
            value_key = str(value).strip()

        mapped_text = values_mapping.get(value_key, str(value))
        return description, mapped_text
