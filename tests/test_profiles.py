import pandas as pd
import pytest

from synthetic_sampling.profiles.generator import RespondentProfileGenerator
from synthetic_sampling.profiles.utils import verify_profile_nesting


@pytest.fixture
def sample_metadata():
    return {
        "demo": {
            "AGE": {
                "question": "How old are you?",
                "values": {1: "18-24", 2: "25-34"},
            },
            "GENDER": {
                "question": "What is your gender?",
                "values": {"M": "Male", "F": "Female"},
            },
        },
        "opinions": {
            "Q1": {
                "question": "Do you feel satisfied?",
                "values": {1: "Yes", 2: "No", 99: "Missing"},
            },
            "Q2": {
                "question": "Do you trust the media?",
                "values": {1: "High", 2: "Low"},
            },
            "Q3": {
                "question": "Do you use social media?",
                "values": {1: "Often", 2: "Rarely"},
            },
        },
    }


@pytest.fixture
def survey_data():
    return pd.DataFrame(
        [
            {"AGE": 1, "GENDER": "M", "Q1": 1, "Q2": 2, "Q3": 1},
            {"AGE": 2, "GENDER": "F", "Q1": 99, "Q2": 1, "Q3": 2},
        ]
    )


@pytest.fixture
def generator(sample_metadata, survey_data):
    return RespondentProfileGenerator(
        survey_data=survey_data,
        metadata=sample_metadata,
        missing_value_labels=["Missing"],
    )


def test_set_target_questions_filters_missing_options(generator):
    generator.set_target_questions(["Q1"])

    target = generator.get_target_question("Q1")
    assert target is not None
    assert target.options == ["Yes", "No"]

    pool = generator.get_available_pool()
    pooled_features = {f for features in pool.values() for f in features}
    assert "Q1" not in pooled_features


def test_generate_profile_respects_always_include_and_missing(generator):
    generator.set_target_questions(["Q1"])
    generator.add_exclusions(["Q3"])  # keep opinions pool deterministic
    generator.set_always_include(["AGE"])

    profile = generator.generate_profile(
        respondent_id=0,
        n_sections=2,
        m_features_per_section=1,
        seed=123,
    )

    assert profile.always_included == ["AGE"]
    assert list(profile.features.keys())[0] == "AGE"
    assert profile.features["AGE"]["value_label"] == "18-24"
    assert profile.n_features == 3  # 1 always-include + 2 sampled

    # ensure missing label was excluded from sampling
    sampled_values = {info["value_label"] for code, info in profile.features.items() if code != "AGE"}
    assert "Missing" not in sampled_values


def test_expand_profile_preserves_existing_features(generator):
    base = generator.generate_profile(
        respondent_id=0,
        n_sections=2,
        m_features_per_section=1,
        seed=7,
    )

    expanded = generator.expand_profile(base, add_features_per_section=1)

    assert verify_profile_nesting([base, expanded])
    assert expanded.n_features > base.n_features


def test_generate_prediction_instance_handles_missing_targets(generator):
    generator.set_target_questions(["Q1"])

    instance = generator.generate_prediction_instance(
        respondent_id=0,
        target_code="Q1",
        n_sections=2,
        m_features_per_section=1,
        seed=1,
    )

    assert instance is not None
    assert instance.answer == "Yes"
    assert "Do you feel satisfied?" == instance.target_question
    assert "Do you feel satisfied?" not in instance.features

    missing_instance = generator.generate_prediction_instance(
        respondent_id=1,
        target_code="Q1",
        n_sections=2,
        m_features_per_section=1,
        seed=1,
    )

    assert missing_instance is None

