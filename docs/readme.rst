==================
Synthetic Sampling
==================

Generate configurable respondent profiles from cross-national survey
metadata for LLM evaluation in social-science research.

**Synthetic Sampling** ships with curated metadata for seven major
cross-national surveys and lets you:

1. **Explore** survey sections and variables without downloading any data.
2. **Load** your own survey microdata (CSV / Stata / SPSS).
3. **Generate** stratified respondent profiles with reproducible random
   sampling.
4. **Format** profiles for LLM prompting using 10 built-in presets or
   custom templates.


Quick Start
-----------

::

    from synthetic_sampling import ProfileBuilder

    # Explore metadata (no data needed)
    builder = ProfileBuilder('wvs')
    builder.list_sections()
    builder.list_variables('demographics')

    # Load survey data
    builder.load_data('~/data/WVS/wvs.csv')

    # Generate a profile
    profile = builder.generate_profile(respondent_id=12345, seed=42)
    print(profile.to_qa_format())

See :doc:`usage` for the full guide.


Available Surveys
-----------------

====================  =======================================  ==================
ID                    Survey                                   Regions
====================  =======================================  ==================
``wvs``               World Values Survey (Wave 7)             Global
``ess_wave_10``       European Social Survey (Wave 10)         Europe
``ess_wave_11``       European Social Survey (Wave 11)         Europe
``afrobarometer``     Afrobarometer                            Africa
``arabbarometer``     Arab Barometer                           Middle East / N. Africa
``asianbarometer``    Asian Barometer                          Asia
``latinobarometer``   Latinobarometro                          Latin America
====================  =======================================  ==================


License
-------

MIT
