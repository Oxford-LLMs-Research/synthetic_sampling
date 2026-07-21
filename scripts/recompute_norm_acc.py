"""
Recompute per_question_norm_acc.csv and majority_class_norm_acc.csv
by re-parsing Arab Barometer and WVS JSONL files with correct target_code extraction.

The original aggregation script used rfind('_') on example_id, which truncated
codes like Q725_5 to just '5'. This script fixes that by correctly parsing
{survey}_{respondent_id}_{target_code}_{profile} format.
"""

import json
import re
import os
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = r'C:\Users\maksimz\Desktop\emnlp\synthetic_sampling\results'
ANALYSIS_DIR = r'C:\Users\maksimz\Desktop\emnlp\synthetic_sampling\analysis\normalized_accuracy'

PROFILE_RE = re.compile(r'_(s\dm\d)$')

# Map result directory name -> model name used in existing CSV
MODEL_NAME_MAP = {
    'deepseek-v3p1-terminus': 'deepseek',
    'gemma-3-27b-instruct':   'gemma3-27b',
    'gpt_oss':                'gpt-oss',
    'llama3.1-70b-base':      'llama3.1_70b_base',
    'llama3.1-70b-instruct':  'llama3.1_70b_instruct',
    'llama3.1-8b-base':       'llama3.1_8b_base',
    'llama3.1-8b-instruct':   'llama3.1_8b_instruct',
    'olmo3-32b-base':         'olmo3_32b_base',
    'olmo3-32b-dpo':          'olmo3_32b_dpo',
    'olmo3-7b-base':          'olmo3_7b_base',
    'olmo3-7b-dpo':           'olmo3_7b_dpo',
    'qwen3-32b':              'qwen3-32b',
    'qwen3-4b':               'qwen3-4b',
}

# Only reprocess surveys that have underscore-containing Q-codes
REPROCESS_SURVEYS = {'arabbarometer', 'wvs'}


def parse_example_id(example_id, survey):
    """
    Extract (target_code, profile_type) from example_id.
    Format: {survey}_{respondent_id}_{target_code}_{profile}
    Respondent ID is always the first field after the survey prefix.
    """
    m = PROFILE_RE.search(example_id)
    profile_type = m.group(1)
    without_profile = example_id[:m.start()]
    after_survey = without_profile[len(survey) + 1:]  # strip 'survey_'
    idx = after_survey.index('_')
    target_code = after_survey[idx + 1:]
    return target_code, profile_type


def find_survey_file(dir_name, survey):
    pattern = os.path.join(RESULTS_DIR, dir_name, f'*_survey_{survey}_results.jsonl')
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_survey_data(dir_name, model_name, survey):
    path = find_survey_file(dir_name, survey)
    if not path:
        print(f'  WARNING: no file for {dir_name}/{survey}')
        return []

    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            target_code, profile_type = parse_example_id(rec['example_id'], survey)
            rows.append({
                'survey': survey,
                'target_code': target_code,
                'profile_type': profile_type,
                'ground_truth': rec['ground_truth'],
                'correct': int(rec['correct']),
                'n_options': len(rec['options']),
                'model': model_name,
            })
    return rows


def compute_norm_acc(raw_acc, n_options):
    if n_options <= 1:
        return np.nan
    chance = 1.0 / n_options
    return (raw_acc - chance) / (1.0 - chance)


def main():
    norm_path = os.path.join(ANALYSIS_DIR, 'per_question_norm_acc.csv')
    maj_path  = os.path.join(ANALYSIS_DIR, 'majority_class_norm_acc.csv')

    df_norm = pd.read_csv(norm_path)
    df_maj  = pd.read_csv(maj_path)

    print(f'Existing norm_acc: {len(df_norm)} rows, '
          f'{df_norm.groupby(["survey","target_code"]).ngroups} unique (survey,target)')
    print(f'Existing majority: {len(df_maj)} rows, '
          f'{df_maj.groupby(["survey","target_code"]).ngroups} unique (survey,target)')

    # Collect raw records for reprocess surveys
    all_rows = []
    for dir_name, model_name in MODEL_NAME_MAP.items():
        for survey in REPROCESS_SURVEYS:
            rows = load_survey_data(dir_name, model_name, survey)
            if rows:
                codes = set(r['target_code'] for r in rows)
                print(f'  {dir_name}/{survey}: {len(rows)} records, {len(codes)} unique codes')
            all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)

    # Compute per-question norm_acc
    records_norm = []
    for (survey, target_code, profile_type, model), g in df_raw.groupby(
            ['survey', 'target_code', 'profile_type', 'model']):
        n = len(g)
        raw_acc = g['correct'].sum() / n
        n_options = g['n_options'].iloc[0]
        records_norm.append({
            'survey': survey,
            'target_code': target_code,
            'profile_type': profile_type,
            'n_respondents': n,
            'raw_acc': raw_acc,
            'n_options': float(n_options),
            'norm_acc': compute_norm_acc(raw_acc, n_options),
            'model': model,
        })
    df_new_norm = pd.DataFrame(records_norm)

    # Compute majority class — use one model (all have same ground_truth)
    df_one_model = df_raw[df_raw['model'] == df_raw['model'].iloc[0]]
    records_maj = []
    for (survey, target_code), g in df_one_model.groupby(['survey', 'target_code']):
        g_p = g[g['profile_type'] == g['profile_type'].iloc[0]]
        maj_class = g_p['ground_truth'].mode()[0]
        maj_acc = (g_p['ground_truth'] == maj_class).mean()
        n_options = g_p['n_options'].iloc[0]
        n_resp = len(g_p)
        maj_norm = compute_norm_acc(maj_acc, n_options)
        for profile_type in sorted(g['profile_type'].unique()):
            records_maj.append({
                'survey': survey,
                'target_code': target_code,
                'profile_type': profile_type,
                'n_respondents': n_resp,
                'majority_acc': maj_acc,
                'n_options': float(n_options),
                'majority_norm_acc': maj_norm,
            })
    df_new_maj = pd.DataFrame(records_maj)

    print(f'\nNew norm_acc: {len(df_new_norm)} rows, '
          f'{df_new_norm.groupby(["survey","target_code"]).ngroups} unique (survey,target)')
    print(f'New majority: {len(df_new_maj)} rows, '
          f'{df_new_maj.groupby(["survey","target_code"]).ngroups} unique (survey,target)')

    print('\nArab Barometer target codes (new):')
    print(sorted(df_new_norm[df_new_norm['survey']=='arabbarometer']['target_code'].unique()))
    print('\nWVS target codes containing underscore (new):')
    wvs_codes = sorted(df_new_norm[df_new_norm['survey']=='wvs']['target_code'].unique())
    print([c for c in wvs_codes if '_' in c])

    # Merge: drop old rows for reprocess surveys, add new
    df_norm_fixed = df_norm[~df_norm['survey'].isin(REPROCESS_SURVEYS)]
    df_norm_fixed = pd.concat([df_norm_fixed, df_new_norm], ignore_index=True)
    df_norm_fixed = df_norm_fixed.sort_values(
        ['survey', 'target_code', 'profile_type', 'model']).reset_index(drop=True)

    df_maj_fixed = df_maj[~df_maj['survey'].isin(REPROCESS_SURVEYS)]
    df_maj_fixed = pd.concat([df_maj_fixed, df_new_maj], ignore_index=True)
    df_maj_fixed = df_maj_fixed.sort_values(
        ['survey', 'target_code', 'profile_type']).reset_index(drop=True)

    print(f'\nFinal norm_acc: {len(df_norm_fixed)} rows, '
          f'{df_norm_fixed.groupby(["survey","target_code"]).ngroups} unique (survey,target)')
    print(f'Final majority: {len(df_maj_fixed)} rows, '
          f'{df_maj_fixed.groupby(["survey","target_code"]).ngroups} unique (survey,target)')

    out_norm = os.path.join(ANALYSIS_DIR, 'per_question_norm_acc_fixed.csv')
    out_maj  = os.path.join(ANALYSIS_DIR, 'majority_class_norm_acc_fixed.csv')
    df_norm_fixed.to_csv(out_norm, index=False)
    df_maj_fixed.to_csv(out_maj, index=False)
    print(f'\nSaved:\n  {out_norm}\n  {out_maj}')


if __name__ == '__main__':
    main()
