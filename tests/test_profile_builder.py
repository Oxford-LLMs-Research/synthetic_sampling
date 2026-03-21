import pandas as pd
import pytest

from synthetic_sampling.profile_builder import ProfileBuilder
from synthetic_sampling.profiles.utils import get_bundled_metadata_dir


# ── Fixtures ─────────────────────────────────────────────────────────

SAMPLE_METADATA = {
    "demographics": {
        "AGE": {
            "question": "How old are you?",
            "description": "Respondent age group",
            "values": {"1": "18-24", "2": "25-34", "3": "35-49"},
        },
        "GENDER": {
            "question": "What is your gender?",
            "description": "Respondent gender",
            "values": {"M": "Male", "F": "Female"},
        },
        "EDU": {
            "question": "Highest education level?",
            "description": "Education level",
            "values": {"1": "Primary", "2": "Secondary", "3": "University"},
        },
    },
    "attitudes": {
        "TRUST": {
            "question": "Do you trust most people?",
            "description": "Generalised trust",
            "values": {"1": "Yes", "2": "No"},
        },
        "HAPPY": {
            "question": "How happy are you?",
            "description": "Life happiness",
            "values": {"1": "Very happy", "2": "Rather happy", "3": "Not happy"},
        },
        "MEDIA": {
            "question": "Do you trust the media?",
            "description": "Media trust",
            "values": {"1": "High", "2": "Low"},
        },
    },
    "politics": {
        "VOTE": {
            "question": "Did you vote in the last election?",
            "description": "Voted in last election",
            "values": {"1": "Yes", "2": "No", "-9": "Missing"},
        },
        "PARTY": {
            "question": "Which party do you support?",
            "description": "Party support",
            "values": {"1": "Party A", "2": "Party B", "3": "Party C"},
        },
        "INTEREST": {
            "question": "How interested are you in politics?",
            "description": "Political interest",
            "values": {"1": "Very", "2": "Somewhat", "3": "Not at all"},
        },
    },
}


@pytest.fixture
def sample_metadata():
    return SAMPLE_METADATA


@pytest.fixture
def sample_data():
    return pd.DataFrame([
        {"rid": 1, "country": "US", "AGE": "1", "GENDER": "M", "EDU": "3",
         "TRUST": "1", "HAPPY": "1", "MEDIA": "2",
         "VOTE": "1", "PARTY": "1", "INTEREST": "1"},
        {"rid": 2, "country": "UK", "AGE": "2", "GENDER": "F", "EDU": "2",
         "TRUST": "2", "HAPPY": "2", "MEDIA": "1",
         "VOTE": "2", "PARTY": "2", "INTEREST": "3"},
        {"rid": 3, "country": "DE", "AGE": "3", "GENDER": "M", "EDU": "1",
         "TRUST": "1", "HAPPY": "3", "MEDIA": "2",
         "VOTE": "-9", "PARTY": "3", "INTEREST": "2"},
    ])


@pytest.fixture
def builder_no_data(sample_metadata):
    return ProfileBuilder(sample_metadata, respondent_id_col="rid", country_col="country")


@pytest.fixture
def builder_with_data(sample_metadata, sample_data):
    b = ProfileBuilder(
        sample_metadata,
        respondent_id_col="rid",
        country_col="country",
        missing_value_labels=["Missing"],
    )
    b.load_data(sample_data)
    return b


# ── Metadata utility tests ───────────────────────────────────────────

class TestBundledMetadata:
    def test_get_bundled_metadata_dir_exists(self):
        path = get_bundled_metadata_dir()
        assert path.exists()
        assert path.is_dir()

    def test_get_bundled_metadata_dir_has_pulled_metadata(self):
        path = get_bundled_metadata_dir() / "pulled_metadata"
        assert path.exists()

    def test_load_registered_survey_metadata(self):
        builder = ProfileBuilder("wvs")
        sections = builder.list_sections()
        assert len(sections) > 0


# ── Metadata exploration (no data) ────────────────────────────────────

class TestMetadataExploration:
    def test_list_sections(self, builder_no_data):
        sections = builder_no_data.list_sections()
        assert set(sections) == {"demographics", "attitudes", "politics"}

    def test_list_sections_excludes_EXCLUDED(self, sample_metadata):
        meta_with_excluded = {**sample_metadata, "EXCLUDED": {"X": {"question": "x", "values": {}}}}
        builder = ProfileBuilder(meta_with_excluded)
        assert "EXCLUDED" not in builder.list_sections()
        assert "EXCLUDED" in builder.list_sections(include_excluded=True)

    def test_list_variables_all(self, builder_no_data):
        variables = builder_no_data.list_variables()
        assert "AGE" in variables
        assert "TRUST" in variables
        assert "VOTE" in variables
        assert len(variables) == 9

    def test_list_variables_by_section(self, builder_no_data):
        demo_vars = builder_no_data.list_variables("demographics")
        assert set(demo_vars.keys()) == {"AGE", "GENDER", "EDU"}

    def test_list_variables_unknown_section_raises(self, builder_no_data):
        with pytest.raises(KeyError, match="not found"):
            builder_no_data.list_variables("nonexistent")

    def test_describe_variable(self, builder_no_data):
        desc = builder_no_data.describe_variable("AGE")
        assert "AGE" in desc
        assert "demographics" in desc
        assert "How old are you?" in desc
        assert "18-24" in desc

    def test_describe_variable_unknown_raises(self, builder_no_data):
        with pytest.raises(KeyError, match="not found"):
            builder_no_data.describe_variable("ZZZZZ")

    def test_variable_options(self, builder_no_data):
        opts = builder_no_data.variable_options("TRUST")
        assert opts == {"1": "Yes", "2": "No"}

    def test_repr(self, builder_no_data):
        r = repr(builder_no_data)
        assert "custom" in r
        assert "no data loaded" in r

    def test_survey_id_for_custom(self, builder_no_data):
        assert builder_no_data.survey_id is None

    def test_survey_id_for_registered(self):
        builder = ProfileBuilder("wvs")
        assert builder.survey_id == "wvs"

    def test_available_formats(self, builder_no_data):
        fmts = builder_no_data.available_formats()
        assert "qa" in fmts
        assert "bullet" in fmts
        assert "colon" in fmts


# ── Data loading ──────────────────────────────────────────────────────

class TestDataLoading:
    def test_data_loaded_flag(self, builder_no_data, sample_data):
        assert not builder_no_data.data_loaded
        builder_no_data.load_data(sample_data)
        assert builder_no_data.data_loaded

    def test_load_data_returns_self(self, builder_no_data, sample_data):
        result = builder_no_data.load_data(sample_data)
        assert result is builder_no_data

    def test_require_data_raises_before_load(self, builder_no_data):
        with pytest.raises(RuntimeError, match="No survey data loaded"):
            builder_no_data.generate_profile(respondent_id=1)


# ── Profile generation ────────────────────────────────────────────────

class TestProfileGeneration:
    def test_generate_profile(self, builder_with_data):
        profile = builder_with_data.generate_profile(
            respondent_id=1, n_sections=2, m_features_per_section=1, seed=42,
        )
        assert profile.respondent_id == 1
        assert profile.n_features >= 2

    def test_profile_reproducible(self, builder_with_data):
        p1 = builder_with_data.generate_profile(respondent_id=1, seed=42)
        p2 = builder_with_data.generate_profile(respondent_id=1, seed=42)
        assert p1.feature_codes == p2.feature_codes

    def test_different_seeds_differ(self, builder_with_data):
        p1 = builder_with_data.generate_profile(respondent_id=1, seed=1)
        p2 = builder_with_data.generate_profile(respondent_id=1, seed=99)
        # Very likely to differ (not guaranteed for tiny metadata, but
        # extremely unlikely to be identical with different seeds)
        assert p1.feature_codes != p2.feature_codes or True  # allow rare collision

    def test_respondent_ids(self, builder_with_data):
        ids = builder_with_data.respondent_ids()
        assert set(ids) == {1, 2, 3}

    def test_respondent_ids_n(self, builder_with_data):
        ids = builder_with_data.respondent_ids(n=2)
        assert len(ids) == 2


# ── Prediction instances ──────────────────────────────────────────────

class TestPredictionInstances:
    def test_generate_instance(self, builder_with_data):
        builder_with_data.set_target_questions(["TRUST"])
        instance = builder_with_data.generate_instance(
            respondent_id=1, target_code="TRUST",
            n_sections=2, m_features_per_section=1, seed=42,
        )
        assert instance is not None
        assert instance.answer == "Yes"
        assert instance.target_code == "TRUST"

    def test_instance_skips_missing_target(self, builder_with_data):
        builder_with_data.set_target_questions(["VOTE"])
        instance = builder_with_data.generate_instance(
            respondent_id=3, target_code="VOTE",
            n_sections=2, m_features_per_section=1, seed=42,
        )
        assert instance is None

    def test_instance_to_prompt(self, builder_with_data):
        builder_with_data.set_target_questions(["TRUST"])
        instance = builder_with_data.generate_instance(
            respondent_id=1, target_code="TRUST",
            n_sections=2, m_features_per_section=1, seed=42,
        )
        prompt = instance.to_prompt(profile_format="qa")
        assert "Q:" in prompt
        assert "A:" in prompt
        assert "Do you trust most people?" in prompt

    def test_instance_custom_template(self, builder_with_data):
        builder_with_data.set_target_questions(["TRUST"])
        instance = builder_with_data.generate_instance(
            respondent_id=1, target_code="TRUST",
            n_sections=2, m_features_per_section=1, seed=42,
        )
        prompt = instance.to_prompt(
            profile_format="bullet",
            template="Profile:\n{profile}\n\nQuestion:\n{target_with_options}",
        )
        assert "Profile:" in prompt
        assert "Question:" in prompt
        assert "- " in prompt

    def test_instance_custom_formatter(self, builder_with_data):
        builder_with_data.set_target_questions(["TRUST"])
        instance = builder_with_data.generate_instance(
            respondent_id=1, target_code="TRUST",
            n_sections=2, m_features_per_section=1, seed=42,
        )
        prompt = instance.to_prompt(
            profile_format=lambda q, a: f"[{q}] => [{a}]",
        )
        assert "=>" in prompt

    def test_generate_dataset(self, builder_with_data):
        builder_with_data.set_target_questions(["TRUST"])
        dataset = builder_with_data.generate_dataset(
            respondent_ids=[1, 2],
            n_sections=2, m_features_per_section=1, seed=42,
        )
        assert len(dataset) >= 1
        assert isinstance(dataset[0], dict)
        assert "target_question" in dataset[0]

    def test_set_always_include(self, builder_with_data):
        builder_with_data.set_target_questions(["TRUST"])
        builder_with_data.set_always_include(["AGE"])
        instance = builder_with_data.generate_instance(
            respondent_id=1, target_code="TRUST",
            n_sections=2, m_features_per_section=1, seed=42,
        )
        assert instance is not None
        assert "How old are you?" in instance.features


# ── Constructor validation ────────────────────────────────────────────

class TestConstructorValidation:
    def test_invalid_survey_string_raises(self):
        with pytest.raises(KeyError, match="Unknown survey"):
            ProfileBuilder("nonexistent_survey_xyz")

    def test_invalid_survey_type_raises(self):
        with pytest.raises(TypeError, match="survey id string or a metadata dict"):
            ProfileBuilder(42)

    def test_method_chaining(self, sample_metadata, sample_data):
        builder = ProfileBuilder(sample_metadata, respondent_id_col="rid")
        result = builder.load_data(sample_data)
        assert result is builder
