=====
Usage
=====

``ProfileBuilder`` is the recommended entry point for using this package.
It wraps the internal profile generation machinery in a simple three-step
workflow: **explore metadata**, **load data**, **generate profiles**.


Exploring Survey Metadata
=========================

You can browse the bundled metadata for any supported survey without
downloading any data files::

    from synthetic_sampling import ProfileBuilder

    builder = ProfileBuilder('wvs')

    # Available thematic sections
    builder.list_sections()
    # ['demographics', 'social_values_and_norms', 'political_attitudes', ...]

    # Variables inside a section (returns dict of var_code -> info)
    demographics = builder.list_variables('demographics')

    # Human-readable description of a single variable
    print(builder.describe_variable('Q260'))

    # Raw value mapping (code -> label)
    builder.variable_options('Q260')
    # {'1': 'Male', '2': 'Female'}

To list all supported surveys::

    from synthetic_sampling import list_surveys
    list_surveys()
    # ['wvs', 'afrobarometer', 'arabbarometer', 'asianbarometer',
    #  'latinobarometer', 'ess_wave_10', 'ess_wave_11']


Loading Survey Data
===================

Download the survey microdata from the original source (e.g. the
`World Values Survey <https://www.worldvaluessurvey.org/>`_ website) and
point the builder at it.  CSV, Stata (``.dta``), and SPSS (``.sav``)
formats are supported::

    builder.load_data('~/data/WVS/wvs_wave7.csv')

You can also pass a pre-loaded DataFrame::

    import pandas as pd
    df = pd.read_csv('~/data/WVS/wvs_wave7.csv')
    builder.load_data(df)

``load_data`` returns the builder itself so you can chain calls::

    builder = ProfileBuilder('wvs').load_data('~/data/WVS/wvs.csv')


Generating Profiles
===================

Once data is loaded you can generate a respondent profile::

    profile = builder.generate_profile(
        respondent_id=12345,
        n_sections=3,            # sample from 3 thematic sections
        m_features_per_section=2, # pick 2 features per section
        seed=42,                  # reproducible sampling
    )

    print(profile.to_qa_format())

The profile type is described by the shorthand ``s{n}m{m}`` --
for example ``s3m2`` means 3 sections with 2 features each (6 total
features, excluding always-include variables).


Profile Parameters
------------------

``n_sections``
    Number of thematic sections to sample from.

``m_features_per_section``
    Number of features to draw from each selected section.

``seed``
    Random seed.  The same ``(respondent_id, seed)`` pair always
    produces the same profile.

``shuffle_features``
    If ``True``, feature order is randomised instead of being grouped
    by section.


Prediction Instances
====================

A **prediction instance** pairs a respondent profile with a target
question and its ground-truth answer::

    builder.set_target_questions(['Q35A', 'Q46'])

    instance = builder.generate_instance(
        respondent_id=12345,
        target_code='Q35A',
        n_sections=3,
        m_features_per_section=2,
        seed=42,
    )

    # Full prompt ready for an LLM
    print(instance.to_prompt())

    # Convert to a serialisable dict
    instance.to_dict()


Batch Generation
----------------

Generate instances for many respondents at once::

    ids = builder.respondent_ids(n=100)
    dataset = builder.generate_dataset(
        respondent_ids=ids,
        n_sections=3,
        m_features_per_section=2,
        seed=42,
    )
    # list of dicts


Profile Formats
===============

The ``profile_format`` argument (or ``format_profile`` method) controls
how profile features are rendered as text.  Ten built-in presets are
available:

=========  ==========================================
Preset     Example output
=========  ==========================================
``qa``     ``Q: How old are you?\nA: 30-45``
``bullet`` ``- How old are you?: 30-45``
``colon``  ``How old are you?: 30-45``
``arrow``  ``How old are you? → 30-45``
``narrative`` ``When asked "How old are you?", they answered "30-45".``
``interview`` ``Interviewer: How old are you?\nRespondent: 30-45``
``xml``    ``<question>…</question>\n<answer>…</answer>``
``json``   ``{"q": "…", "a": "…"}``
``brackets`` ``[How old are you?] 30-45``
``card``   ``How old are you? | 30-45``
=========  ==========================================

List them programmatically::

    from synthetic_sampling import list_profile_formats
    list_profile_formats()

Pass a custom callable for any format not listed::

    instance.to_prompt(
        profile_format=lambda q, a: f"** {q} ** => {a}",
    )


Custom Templates
================

Override the full prompt layout with the ``template`` argument::

    instance.to_prompt(
        profile_format='bullet',
        template=(
            "Here is a survey respondent:\n\n"
            "{profile}\n\n"
            "Based on this profile, answer:\n"
            "{target_with_options}"
        ),
    )

Available placeholders:

- ``{profile}`` -- formatted profile features
- ``{target}`` -- target question text (without options)
- ``{options}`` -- numbered list of answer options
- ``{target_with_options}`` -- target question text followed by options


Custom Survey Metadata
======================

You can use ``ProfileBuilder`` with your own metadata (no registry
entry needed).  The metadata dict must follow this schema::

    my_metadata = {
        "section_name": {
            "var_code": {
                "question": "Full question text",
                "description": "Short description",
                "values": {"1": "Option A", "2": "Option B"},
            },
        },
    }

    builder = ProfileBuilder(
        my_metadata,
        respondent_id_col='id',
        country_col='country',
    )
    builder.load_data(my_dataframe)


Advanced: Semantic Filtering
============================

Enable sentence-transformer similarity filtering to exclude profile
features whose question text is too similar to the target question.
This prevents information leakage::

    builder = ProfileBuilder(
        'wvs',
        use_semantic_filtering=True,
        similarity_model='all-MiniLM-L6-v2',
        similarity_threshold=0.7,
    )

This requires the ``sentence-transformers`` package::

    pip install "synthetic_sampling[semantic]"


Advanced: Missing Value Configuration
======================================

By default, common survey artefacts (``Missing``, ``No answer``,
``Refused``, etc.) are excluded from both profile features and target
options.  Customise the exclusion lists via::

    builder = ProfileBuilder(
        'wvs',
        missing_value_labels=['Missing', 'No answer'],
        missing_value_patterns=['not asked', 'refused'],
    )
