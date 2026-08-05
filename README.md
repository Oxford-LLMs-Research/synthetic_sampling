# Synthetic Sampling

Generate configurable respondent profiles from cross-national survey metadata for LLM evaluation in social-science research.

**Synthetic Sampling** ships with curated metadata for seven major cross-national surveys and lets you:

1. **Explore** survey sections and variables without downloading any data.
2. **Load** your own survey microdata (CSV / Stata / SPSS).
3. **Generate** stratified respondent profiles with reproducible random sampling.
4. **Format** profiles for LLM prompting using 10 built-in presets or custom templates.

## Installation

```bash
pip install synthetic_sampling
```

Optional extras:

```bash
pip install "synthetic_sampling[semantic]"   # sentence-transformers for similarity filtering
pip install "synthetic_sampling[all]"        # all optional dependencies
```

## Quick Start

### 1. Explore survey metadata (no data needed)

```python
from synthetic_sampling import ProfileBuilder

builder = ProfileBuilder('wvs')

# List thematic sections
builder.list_sections()
# ['demographics', 'social_values_and_norms', 'political_attitudes', ...]

# Browse variables in a section
builder.list_variables('demographics')
# {'Q260': {'question': 'Sex', 'values': {'1': 'Male', '2': 'Female'}}, ...}

# Describe a single variable
print(builder.describe_variable('Q260'))
# Variable : Q260
# Section  : demographics
# Question : Sex
# Values   :
#   1 = Male
#   2 = Female
```

### 2. Load survey data and generate profiles

Download the survey microdata from the source (e.g. [World Values Survey](https://www.worldvaluessurvey.org/)) and point the builder at it:

```python
builder.load_data('~/data/WVS/wvs_wave7.csv')

profile = builder.generate_profile(
    respondent_id=12345,
    n_sections=3,
    m_features_per_section=2,
    seed=42,
)
print(profile.to_qa_format())
```

### 3. Generate prediction instances

Create profile + target-question bundles for LLM evaluation:

```python
builder.set_target_questions(['Q35A', 'Q46'])

instance = builder.generate_instance(
    respondent_id=12345,
    target_code='Q35A',
    n_sections=3,
    m_features_per_section=2,
    seed=42,
)

# Default prompt
print(instance.to_prompt(profile_format='qa'))

# Custom template
print(instance.to_prompt(
    profile_format='bullet',
    template=(
        "Here is a survey respondent:\n\n{profile}\n\n"
        "Based on this profile, answer:\n{target_with_options}"
    ),
))
```

### 4. Batch generation

```python
ids = builder.respondent_ids(n=100)
dataset = builder.generate_dataset(
    respondent_ids=ids,
    n_sections=3,
    m_features_per_section=2,
    seed=42,
)
# dataset is a list of dicts ready for serialisation
```

## Available Surveys

| ID | Survey | Regions |
|----|--------|---------|
| `wvs` | World Values Survey (Wave 7) | Global |
| `ess_wave_10` | European Social Survey (Wave 10) | Europe |
| `ess_wave_11` | European Social Survey (Wave 11) | Europe |
| `afrobarometer` | Afrobarometer | Africa |
| `arabbarometer` | Arab Barometer | Middle East / North Africa |
| `asianbarometer` | Asian Barometer | Asia |
| `latinobarometer` | Latinobarometro | Latin America |

```python
from synthetic_sampling import list_surveys
list_surveys()
# ['wvs', 'afrobarometer', 'arabbarometer', 'asianbarometer',
#  'latinobarometer', 'ess_wave_10', 'ess_wave_11']
```

## Profile Formats

Built-in presets for `profile_format` / `format_profile()`:

| Preset | Example output |
|--------|---------------|
| `qa` | `Q: How old are you?\nA: 30-45` |
| `bullet` | `- How old are you?: 30-45` |
| `colon` | `How old are you?: 30-45` |
| `arrow` | `How old are you? -> 30-45` |
| `narrative` | `When asked "How old are you?", they answered "30-45".` |
| `interview` | `Interviewer: How old are you?\nRespondent: 30-45` |
| `xml` | `<question>How old are you?</question>\n<answer>30-45</answer>` |
| `json` | `{"q": "How old are you?", "a": "30-45"}` |
| `brackets` | `[How old are you?] 30-45` |
| `card` | `How old are you? \| 30-45` |

Pass a custom callable for any format not listed:

```python
instance.to_prompt(
    profile_format=lambda q, a: f"** {q} ** => {a}",
)
```

## Using Custom Survey Metadata

You can use `ProfileBuilder` with your own metadata (no registry entry required):

```python
my_metadata = {
    "demographics": {
        "age": {
            "question": "How old are you?",
            "description": "Respondent age bracket",
            "values": {"1": "18-29", "2": "30-49", "3": "50+"},
        },
    },
    "attitudes": {
        "trust": {
            "question": "Do you trust most people?",
            "description": "Generalised trust",
            "values": {"1": "Yes", "2": "No"},
        },
    },
}

builder = ProfileBuilder(
    my_metadata,
    respondent_id_col='respondent_id',
    country_col='country',
)
builder.load_data(my_dataframe)
```

## License

MIT
