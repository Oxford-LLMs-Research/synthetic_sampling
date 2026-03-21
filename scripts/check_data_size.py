import pandas as pd

df = pd.read_csv('analysis/mixed_effects/mixed_effects_data.csv')
print(f'Rows: {len(df):,}')
print(f'Unique models: {df["model"].nunique()}')
print(f'Unique respondents: {df["respondent"].nunique():,}')
print(f'Unique questions: {df["question"].nunique()}')
print(f'Unique regions: {df["region"].nunique()}')
print(f'Unique sections: {df["topic_section"].nunique()}')

# Estimate memory needed for design matrix
# With crossed random effects, statsmodels creates a matrix with:
# rows = n_obs, cols = sum of levels for each random effect
total_levels = (df["model"].nunique() + 
                df["respondent"].nunique() + 
                df["question"].nunique() + 
                df["region"].nunique() + 
                df["topic_section"].nunique())
print(f'\nTotal random effect levels: {total_levels:,}')
print(f'Estimated design matrix size: {len(df):,} x {total_levels:,}')
print(f'Estimated memory (float64): {(len(df) * total_levels * 8) / (1024**3):.1f} GiB')
