# -------------------------------------------------------------------
# Script: consolidate ESS variables
# -------------------------------------------------------------------

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import unicodedata

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ESS10_META_PATH_IN = Path("ess10_meta_enriched_clean.json")
ESS11_META_PATH_IN = Path("ess11_meta_enriched_clean.json")   

ESS10_DATA_PATH_IN = Path("../../../../../data/ess/ESS10.csv")
ESS11_DATA_PATH_IN = Path("../../../../../data/ess/ESS11.csv")

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output names
ESS10_DATA_PATH_OUT = OUT_DIR / "ESS10_with_consolidations.csv"
ESS11_DATA_PATH_OUT = OUT_DIR / "ESS11_with_consolidations.csv"
ESS10_META_PATH_OUT = OUT_DIR / "ess10_meta_enriched_clean_with_consolidations.json"
ESS11_META_PATH_OUT = OUT_DIR / "ess11_meta_enriched_clean_with_consolidations.json"

# New variable name (same for ESS10 and ESS11)
NEW_VAR = "prtcl_closest_party_name"
NEW_VAR_SRC = "prtcl_closest_party_source_var"
NEW_VAR_VOTED = "prtvtd_party_last_voted_name"
NEW_VAR_VOTED_SRC = "prtvtd_party_last_voted_source_var"
NEW_VAR_FLAGS = {
    "multi_nonmissing_flag": "prtcl_closest_party_multi_nonmissing_flag",
    "unmapped_code_flag": "prtcl_closest_party_unmapped_code_flag",
    "nonmissing_count": "prtcl_closest_party_nonmissing_count",
}

MISSING_LABEL = "missing"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def code_to_key(x):
    """
    Convert a dataframe cell to a metadata key (ESS codes are stored as strings in metadata dicts).
    Handles numeric-like strings, ints, floats (e.g., 2.0 -> "2").
    """
    if pd.isna(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        # normalize "2.0" -> "2"
        if re.fullmatch(r"-?\d+(\.0+)?", s):
            return str(int(float(s)))
        return s

    # numeric types
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isfinite(x):
            if float(x).is_integer():
                return str(int(x))
            return str(x)
        return None

    # fallback
    return str(x)


def normalize_question_text(s):
    return (s or "").strip().lower()


def clean_label_for_encoding(x: str) -> str:
    """
    Normalize/clean strings to reduce encoding headaches downstream:
      - Unicode normalize (NFKC)
      - Replace curly quotes/apostrophes/dashes with ASCII equivalents
      - Replace non-breaking spaces
      - Strip/control-whitespace cleanup
    """
    if x is None or pd.isna(x):
        return MISSING_LABEL

    s = str(x)

    # Normalize unicode forms (compatibility composition)
    s = unicodedata.normalize("NFKC", s)

    # Replace common “problem” punctuation with ASCII
    replacements = {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote (curly apostrophe)
        "\u201B": "'",  # single high-reversed-9 quote
        "\u2032": "'",  # prime
        "\u2035": "'",  # reversed prime
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote
        "\u201E": '"',  # low double quote
        "\u00AB": '"',  # left angle quote
        "\u00BB": '"',  # right angle quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u00A0": " ",  # non-breaking space
        "\u2007": " ",  # figure space
        "\u202F": " ",  # narrow no-break space
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Drop remaining control chars (keep normal printable text)
    s = "".join(ch for ch in s if (ch.isprintable() or ch in "\t\n\r"))

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Guard empty -> missing
    return s if s else MISSING_LABEL

def find_party_closeness_vars(meta_records):
    """
    Find all variable_names whose question_text_raw indicates party closeness.
    Primary rule (your request): question_text_raw starts with 'Which party feel closer to'
    Also includes a robust fallback for slight wording variants.
    """
    vars_found = []
    for rec in meta_records:
        if not isinstance(rec, dict):
            continue
        q_raw = normalize_question_text(rec.get("question_text_raw"))
        if not q_raw:
            continue

        # Strict startswith criterion (as requested)
        if q_raw.startswith("which party feel closer to"):
            vars_found.append(rec.get("variable_name"))
            continue

        # Robust fallback (wording variants across rounds/languages/editions)
        # e.g., "Which party do you feel closer to", "Which party do you feel closest to"
        if (
            "which party" in q_raw
            and ("feel closer" in q_raw or "feel closest" in q_raw)
        ):
            vars_found.append(rec.get("variable_name"))

    # remove Nones / duplicates while preserving order
    seen = set()
    out = []
    for v in vars_found:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_value_map_for_var(meta_records, var_name):
    """
    Build mapping code(str) -> label(str) for a given variable_name.
    Prefer values_raw (typically contains the actual party list for that country variable).
    Fall back to adjusted_values if values_raw missing.
    """
    rec = next((r for r in meta_records if isinstance(r, dict) and r.get("variable_name") == var_name), None)
    if rec is None:
        return {}

    values_raw = rec.get("values_raw") or {}
    adjusted = rec.get("adjusted_values") or {}

    # For party lists, values_raw is usually the correct (full) mapping.
    # If it's empty, use adjusted_values.
    mapping = values_raw if isinstance(values_raw, dict) and len(values_raw) > 0 else adjusted
    if not isinstance(mapping, dict):
        return {}

    # ensure keys are strings
    return {str(k): ("" if v is None else str(v)) for k, v in mapping.items()}


def add_party_closeness_column(df, meta_records, new_var=NEW_VAR, ignore_not_applicable=False, na_category="No answer"):
    """
    Adds:
      - new_var: consolidated party closeness label
      - target_var_src: which country-specific variable supplied the value
    """
    party_vars = find_party_closeness_vars(meta_records)

    # Keep only vars that are actually present in the data
    party_vars_in_df = [v for v in party_vars if v in df.columns]

    if not party_vars_in_df:
        raise ValueError(
            "No party closeness variables found in the dataframe based on metadata. "
            "Check that metadata question_text_raw contains the expected wording and matches column names."
        )

    # Build per-variable mappings
    value_maps = {v: build_value_map_for_var(meta_records, v) for v in party_vars_in_df}

    sub = df[party_vars_in_df]
    if ignore_not_applicable:
    # Replace "Not applicable" category with missing
        for v in party_vars_in_df:
            s = sub[v].astype("string").str.strip()
            mask = s == str(na_category)
            sub.loc[mask, v] = pd.NA
    nonnull = sub.notna()
    nonmissing_count = nonnull.sum(axis=1)
    
    check = df[nonmissing_count > 1]
    check2 = check[party_vars_in_df]

    multi_flag = nonmissing_count > 1
    any_flag = nonmissing_count > 0

    # Identify the first non-missing column for each row (vectorized)
    # idxmax returns the first True column because booleans cast to 0/1
    src_col = nonnull.idxmax(axis=1)
    src_col = src_col.where(any_flag, other=pd.NA)

    # Pull the corresponding value (vectorized via numpy)
    arr = sub.to_numpy()
    mask = ~pd.isna(arr)
    first_idx = mask.argmax(axis=1)  # returns 0 if all False; we'll mask those out using any_flag
    first_val = arr[np.arange(arr.shape[0]), first_idx]
    first_val = pd.Series(first_val, index=df.index)
    first_val = first_val.where(any_flag, other=pd.NA)

    
     # Initialize result as 'missing'
    out = pd.Series(MISSING_LABEL, index=df.index, dtype="object")
    unmapped_flag = pd.Series(False, index=df.index)

    for col in party_vars_in_df:
        idx = src_col == col
        if not idx.any():
            continue

        codes = first_val.loc[idx].map(code_to_key)
        mapping = value_maps.get(col, {}) or {}

        mapped = codes.map(mapping)
        is_unmapped = codes.notna() & mapped.isna()
        unmapped_flag.loc[idx] = is_unmapped.fillna(False)

        # use mapping if present; if not, keep MISSING_LABEL
        out.loc[idx] = mapped.fillna(MISSING_LABEL)

    # clean strings for encoding issues
    out = out.map(clean_label_for_encoding)

    # Add columns to df
    df[new_var] = out
    df[NEW_VAR_SRC] = src_col.astype("object").fillna(MISSING_LABEL)
    # df[NEW_VAR_FLAGS["nonmissing_count"]] = nonmissing_count.astype("Int64")
    # df[NEW_VAR_FLAGS["multi_nonmissing_flag"]] = multi_flag
    # df[NEW_VAR_FLAGS["unmapped_code_flag"]] = unmapped_flag
    

    # Basic checks / reporting
    n = len(df)
    n_none = int((nonmissing_count == 0).sum())
    n_multi = int(multi_flag.sum())
    n_unmapped = int(unmapped_flag.sum())

    print("------------------------------------------------------------")
    print(f"Created {new_var}")
    print(f"Party-closeness columns detected (in df): {len(party_vars_in_df)}")
    print(f"Rows: {n}")
    print(f"Rows with no non-missing party variable -> '{MISSING_LABEL}': {n_none}")
    print(f"Rows with >1 non-missing party variable (DATA ERROR FLAG): {n_multi}")
    print(f"Rows with unmapped code (CHECK METADATA/VALUES): {n_unmapped}")
    print("------------------------------------------------------------")

    # Save a small diagnostic file for problematic rows (optional but useful)
    # diag = df.loc[multi_flag | unmapped_flag, ["idno", "cntry"] + party_vars_in_df + [new_var, NEW_VAR_SRC,
    #                                                                                 NEW_VAR_FLAGS["nonmissing_count"],
    #                                                                                 NEW_VAR_FLAGS["multi_nonmissing_flag"],
    #                                                                                 NEW_VAR_FLAGS["unmapped_code_flag"]]]
    # if len(diag) > 0:
    #     diag_path = OUT_DIR / f"diagnostics_{new_var}.csv"
    #     diag.to_csv(diag_path, index=False)
    #     print(f"Diagnostics written to: {diag_path}")

    return party_vars_in_df


def add_party_voted_column(df, meta_records, new_var=NEW_VAR_VOTED, ignore_not_applicable=False, na_category="No answer"):
    """
    Adds:
      - NEW_VAR: consolidated party closeness label
      - NEW_VAR_SRC: which country-specific variable supplied the value
    """
    party_vars = ['prtvtebe', 'prtvtebg', 'prtvthch', 'prtvtbhr', 'prtvtecz', 
                  'prtvtdat', 'prtvtfbg',
                  'prtvthee', 'prtvtefi', 'prtvtefr', 'prtvtdgr', 'prtvtghu', 
                  'prtvtdis', 'prtvtdie', 'prtvtdit', 'prtvclt1', #'prtvclt2', # removed second and third ballots for Lithuania and second from Germany
                  # 'prtvclt3',  'prtvgde2',
                  'prtvtame', 'prtvthnl', 'prtvtmk', 'prtvtbno', 'prtvtdpt', 
                  'prtvtfsi', 'prtvtesk', 'prtvtdgb', 'prtvtchr', 'prtvtccy',
                  'prtvtiee', 'prtvtffi', 'prtvtffr', 'prtvgde1', 'prtvtegr', 'prtvthhu', 
                  'prtvteis', 'prtvteie', 'prtvteil', 'prtvteit', 'prtvtblv', 'prtvtbme', 
                  'prtvtinl', 'prtvtcno', 'prtvtfpl', 'prtvtept', 'prtvtbrs', 'prtvtgsi', 
                  'prtvtges', 'prtvtdse', 'prtvtdua']

    # Keep only vars that are actually present in the data
    party_vars_in_df = [v for v in party_vars if v in df.columns]

    # Build per-variable mappings
    value_maps = {v: build_value_map_for_var(meta_records, v) for v in party_vars_in_df}

    sub = df[party_vars_in_df]
    if ignore_not_applicable:
        # Replace "Not applicable" category with missing
        for v in party_vars_in_df:
            s = sub[v].astype("string").str.strip()
            mask = s == str(na_category)
            sub.loc[mask, v] = pd.NA
    nonnull = sub.notna()
    nonmissing_count = nonnull.sum(axis=1)
    
    # Checking - element with all rows where nonmissing_count >1
    dups = df[nonmissing_count > 1]        
    dups2 = dups[party_vars_in_df]
    
    missings  = df[nonmissing_count == 0] 
    missings2 = missings[party_vars_in_df]

    multi_flag = nonmissing_count > 1
    any_flag = nonmissing_count > 0

    # Identify the first non-missing column for each row (vectorized)
    # idxmax returns the first True column because booleans cast to 0/1
    src_col = nonnull.idxmax(axis=1)
    src_col = src_col.where(any_flag, other=pd.NA)

    # Pull the corresponding value (vectorized via numpy)
    arr = sub.to_numpy()
    mask = ~pd.isna(arr)
    first_idx = mask.argmax(axis=1)  # returns 0 if all False; we'll mask those out using any_flag
    first_val = arr[np.arange(arr.shape[0]), first_idx]
    first_val = pd.Series(first_val, index=df.index)
    first_val = first_val.where(any_flag, other=pd.NA)

    
     # Initialize result as 'missing'
    out = pd.Series(MISSING_LABEL, index=df.index, dtype="object")
    unmapped_flag = pd.Series(False, index=df.index)

    for col in party_vars_in_df:
        idx = src_col == col
        if not idx.any():
            continue

        codes = first_val.loc[idx].map(code_to_key)
        mapping = value_maps.get(col, {}) or {}

        mapped = codes.map(mapping)
        is_unmapped = codes.notna() & mapped.isna()
        unmapped_flag.loc[idx] = is_unmapped.fillna(False)

        # use mapping if present; if not, keep MISSING_LABEL
        out.loc[idx] = mapped.fillna(MISSING_LABEL)

    # clean strings for encoding issues
    out = out.map(clean_label_for_encoding)

    # Add columns to df
    df[new_var] = out
    df[NEW_VAR_VOTED_SRC] = src_col.astype("object").fillna(MISSING_LABEL)
    

    # Basic checks / reporting
    n = len(df)
    n_none = int((nonmissing_count == 0).sum())
    n_multi = int(multi_flag.sum())
    n_unmapped = int(unmapped_flag.sum())

    print("------------------------------------------------------------")
    print(f"Created {new_var}")
    print(f"Party-vote columns detected (in df): {len(party_vars_in_df)}")
    print(f"Rows: {n}")
    print(f"Rows with no non-missing party variable -> '{MISSING_LABEL}': {n_none}")
    print(f"Rows with >1 non-missing party variable (DATA ERROR FLAG): {n_multi}")
    print(f"Rows with unmapped code (CHECK METADATA/VALUES): {n_unmapped}")
    print("------------------------------------------------------------")

    return party_vars_in_df


def add_consolidated_column(df, meta_records, new_var, target_vars, target_var_src, ignore_not_applicable=False, na_category="Not applicable"):
    """
    Adds:
      - new_var: consolidated values of target variable list
      - target_var_src: which specific variable supplied the value
    """

    # Keep only vars that are actually present in the data
    target_vars_in_df = [v for v in target_vars if v in df.columns]



    # Build per-variable mappings
    value_maps = {v: build_value_map_for_var(meta_records, v) for v in target_vars_in_df}

    sub = df[target_vars_in_df]
    if ignore_not_applicable:
        # Replace "Not applicable" category with missing
        for v in target_vars_in_df:
            s = sub[v].astype("string").str.strip()
            mask = s == str(na_category)
            sub.loc[mask, v] = pd.NA
    nonnull = sub.notna()
    nonmissing_count = nonnull.sum(axis=1)

    multi_flag = nonmissing_count > 1
    any_flag = nonmissing_count > 0

    # Identify the first non-missing column for each row (vectorized)
    # idxmax returns the first True column because booleans cast to 0/1
    src_col = nonnull.idxmax(axis=1)
    src_col = src_col.where(any_flag, other=pd.NA)

    # Pull the corresponding value (vectorized via numpy)
    arr = sub.to_numpy()
    mask = ~pd.isna(arr)
    first_idx = mask.argmax(axis=1)  # returns 0 if all False; we'll mask those out using any_flag
    first_val = arr[np.arange(arr.shape[0]), first_idx]
    first_val = pd.Series(first_val, index=df.index)
    first_val = first_val.where(any_flag, other=pd.NA)

    
     # Initialize result as 'missing'
    out = pd.Series(MISSING_LABEL, index=df.index, dtype="object")
    unmapped_flag = pd.Series(False, index=df.index)

    for col in target_vars_in_df:
        idx = src_col == col
        if not idx.any():
            continue

        codes = first_val.loc[idx].map(code_to_key)
        mapping = value_maps.get(col, {}) or {}

        mapped = codes.map(mapping)
        is_unmapped = codes.notna() & mapped.isna()
        unmapped_flag.loc[idx] = is_unmapped.fillna(False)

        # use mapping if present; if not, keep MISSING_LABEL
        out.loc[idx] = mapped.fillna(MISSING_LABEL)

    # clean strings for encoding issues
    out = out.map(clean_label_for_encoding)

    # Add columns to df
    df[new_var] = out
    df[target_var_src] = src_col.astype("object").fillna(MISSING_LABEL)
    

    # Basic checks / reporting
    n = len(df)
    n_none = int((nonmissing_count == 0).sum())
    n_multi = int(multi_flag.sum())
    n_unmapped = int(unmapped_flag.sum())

    print("------------------------------------------------------------")
    print(f"Created {new_var}")
    print(f"Target columns detected (in df): {len(target_vars_in_df)}")
    print(f"Rows: {n}")
    print(f"Rows with no non-missing target variable value -> '{MISSING_LABEL}': {n_none}")
    print(f"Rows with >1 non-missing target variable value (DATA ERROR FLAG): {n_multi}")
    print(f"Rows with unmapped code (CHECK METADATA/VALUES): {n_unmapped}")
    print("------------------------------------------------------------")

    return target_vars_in_df



def add_metadata_entry_closeness(meta_records, df, source_vars, ess_round_label):
    """
    Append metadata entry for the derived variable, including adjusted_values as an
    identity mapping over all response categories observed in the data file.
    """
    # avoid duplicates if re-running
    if any(isinstance(r, dict) and r.get("variable_name") == NEW_VAR for r in meta_records):
        return meta_records

    # Collect ALL observed categories in the data, and CLEAN them
    observed = (
        df[NEW_VAR]
        .astype("string")
        .fillna(MISSING_LABEL)
        .map(clean_label_for_encoding)
        .unique()
        .tolist()
    )

    observed_clean = [x for x in observed if x is not None]
    observed_sorted = sorted(set(observed_clean), key=lambda x: (x != MISSING_LABEL, str(x).lower()))
    identity_map = {str(v): str(v) for v in observed_sorted}

    rec = {
        "variable_name": NEW_VAR,
        "question_text_raw": (
            "Derived variable: consolidated party closeness across countries.\n"
            "Constructed by selecting the (single) non-missing country-specific 'Which party ... feel closer to' variable "
            "and replacing the numeric code with the corresponding party label from the metadata."
        ),
        "values_raw": {
            "": "Free text labels derived from country-specific party lists; see adjusted_values for observed categories."
        },
        "description": "Party respondent feels closest to (consolidated)",
        "question_cleaned": "Which political party do you feel closest to?",
        "adjusted_values": identity_map,   
        "notes": (
            f"Derived for {ess_round_label}. Source variables: {', '.join(source_vars)}. "
            "For each respondent, the script identifies the only non-missing source variable among these; "
            f"if multiple are non-missing, {NEW_VAR_FLAGS['multi_nonmissing_flag']} is set to True. "
            f"If none are non-missing, {NEW_VAR} is set to '{MISSING_LABEL}'. "
            "Numeric response codes are decoded to party names using each source variable's values_raw in the metadata. "
            # f"If a code is not found in the mapping, {NEW_VAR_FLAGS['unmapped_code_flag']} is set to True and the value is set to '{MISSING_LABEL}'. "
            "adjusted_values for this derived variable is an identity mapping over all categories observed in the exported data file."
        ),
        "question_category": "politics",
        "notes_2": "This variable consolidates country-specific party closeness items into a single harmonized string variable for cross-national analysis."
    }
    meta_records.append(rec)
    return meta_records



def add_metadata_entry_generic(meta_records, df, source_vars, ess_round_label, var_name, target_vars, question_category, question_wording, description="see source variables"):
    """
    Append metadata entry for the derived variable, including adjusted_values as an
    identity mapping over all response categories observed in the data file.
    """
    # avoid duplicates if re-running
    if any(isinstance(r, dict) and r.get("variable_name") == var_name for r in meta_records):
        return meta_records

    # Collect ALL observed categories in the data, and CLEAN them
    observed = (
        df[var_name]
        .astype("string")
        .fillna(MISSING_LABEL)
        .map(clean_label_for_encoding)
        .unique()
        .tolist()
    )

    observed_clean = [x for x in observed if x is not None]
    observed_sorted = sorted(set(observed_clean), key=lambda x: (x != MISSING_LABEL, str(x).lower()))
    identity_map = {str(v): str(v) for v in observed_sorted}

    rec = {
        "variable_name": var_name,
        "question_text_raw": (
            f"Derived variable: consolidated across countries. Source variables: {', '.join(source_vars)}.\n"
        ),
        "values_raw": {
            "": "see adjusted_values for observed categories."
        },
        "description": description,
        "question_cleaned": question_wording,
        "adjusted_values": identity_map,   
        "notes": (
            f"Derived for {ess_round_label}. Source variables: {', '.join(source_vars)}. "
            "For each respondent, the script identifies the only non-missing source variable among these; "
            f"If none are non-missing, {var_name} is set to '{MISSING_LABEL}'. "
            "Numeric response codes are replaced with clear text labels using each source variable's values_raw in the metadata."
            "adjusted_values for this derived variable is an identity mapping over all categories observed in the exported data file."
        ),
        "question_category": question_category,
        "notes_2": ""
    }
    meta_records.append(rec)
    return meta_records




# Run the process - ESS 10
meta = load_json(ESS10_META_PATH_IN)
df = pd.read_csv(ESS10_DATA_PATH_IN, low_memory=False)
# Party closeness
source_vars = add_party_closeness_column(df, meta_records=meta, new_var=NEW_VAR)
meta = add_metadata_entry_closeness(meta, df=df, source_vars=source_vars, ess_round_label="ESS Round 10")
# Party last voted for (taking first vote if multiple options in country)
source_vars = add_party_voted_column(df, meta_records=meta, new_var=NEW_VAR_VOTED)
meta = add_metadata_entry_generic(meta,  df=df, source_vars=source_vars, ess_round_label="ESS Round 10", var_name=NEW_VAR_VOTED, target_vars = 'prtvteXX', question_category = 'politics', question_wording = 'What party did you vote for in the last election?', description="see source variables")
# Importance for democracy variable
target_vars = ['impdema', 'impdemb', 'impdemc', 'impdemd', 'impdeme']
source_vars = add_consolidated_column(df, meta_records=meta, new_var='impdem_consolidated', target_vars=target_vars, target_var_src='impdem_consolidated_source_var', ignore_not_applicable=True, na_category='6')
meta = add_metadata_entry_generic(meta,  df=df, source_vars=source_vars, ess_round_label="ESS Round 10", var_name='impdem_consolidated', target_vars = target_vars, question_category = 'politics', question_wording = 'What do you think is the most important for democracy in general?', description="see source variables")
# Own education variable
target_vars = ['edlvebe', 'edlvebg', 'edlvdch', 'edlvehr', 'edlvdcz', 
               'edlvdee', 'edlvdfi', 'edlvdfr', 'edlvegr', 'edlvdahu',
               'edlvdis', 'edlvdie', 'edlveit', 'edlvdlt', 'edlvdme', 
               'edlvenl', 'edlvdmk', 'edlveno', 'edlvdpt', 'edlvesi', 
               'edlvdsk', 'edubgb2' #,  'educgb1' # removed due to large redundancy with edubgb2
               ]
source_vars = add_consolidated_column(df, meta_records=meta, new_var='edlv_consolidated', target_vars=target_vars, target_var_src='edlv_consolidated_source_var', ignore_not_applicable=False)
meta = add_metadata_entry_generic(meta,  df=df, source_vars=source_vars, ess_round_label="ESS Round 10", var_name='edlv_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = 'What is the highest level of education you have completed?', description="see source variables")
# Partner education variable
target_vars = ['edlvpebe', 'edlvpebg', 'edlvpdch', 'edlvpehr', 'edlvpdcz', 
               'edlvpdee', 'edlvpdfi', 'edlvpdfr', 'edlvpegr', 'edlvpdahu',
               'edlvpdis', 'edlvpdie', 'edlvpeit', 'edlvpdlt', 'edlvpdme',
               'edlvpenl', 'edlvpdmk', 'edlvpeno', 'edlvpdpt', 'edlvpesi',
               'edlvpdsk', 'edupbgb2' #,  'edupcgb1' # removed due to large redundancy with edubgb2
               ]
source_vars = add_consolidated_column(df, meta_records=meta, new_var='edlvp_consolidated', target_vars=target_vars, target_var_src='edlvp_consolidated_source_var', ignore_not_applicable=False)
meta = add_metadata_entry_generic(meta,  df=df, source_vars=source_vars, ess_round_label="ESS Round 10", var_name='edlvp_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = "What is the highest level of education your spouse or partner has completed?", description="see source variables")
# Father education variable 
target_vars = ['edlvfebe', 'edlvfebg', 'edlvfdch', 'edlvfehr', 'edlvfdcz',
               'edlvfdee', 'edlvfdfi', 'edlvfdfr', 'edlvfegr', 'edlvfdahu',
               'edlvfdis', 'edlvfdie', 'edlvfeit', 'edlvfdlt', 'edlvfdme',
               'edlvfenl', 'edlvfdmk', 'edlvfeno', 'edlvfdpt', 'edlvfesi',
               'edlvfdsk', 'edufbgb2' #,  'edupcgb1' # removed due to large redundancy with edubgb2
               ]           
source_vars = add_consolidated_column(df, meta_records=meta, new_var='edlvf_consolidated', target_vars=target_vars, target_var_src='edlvf_consolidated_source_var', ignore_not_applicable=False)
meta = add_metadata_entry_generic(meta,  df=df, source_vars=source_vars, ess_round_label="ESS Round 10", var_name='edlvf_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = "What is the highest level of education your father has completed?", description="see source variables")
# Mother education variable 
target_vars = ['edlvmebe', 'edlvmebg', 'edlvmdch', 'edlvmehr', 'edlvmdcz',
               'edlvmdee', 'edlvmdfi', 'edlvmdfr', 'edlvmegr', 'edlvmdahu',
               'edlvmdis', 'edlvmdie', 'edlvmeit', 'edlvmdlt', 'edlvmdme',
               'edlvmenl', 'edlvmdmk', 'edlvmeno', 'edlvmdpt', 'edlvmesi',
               'edlvmdsk', 'edumbgb2' #,  'edumcgb1' # removed due to large redundancy with edubgb2
               ]           
source_vars = add_consolidated_column(df, meta_records=meta, new_var='edlvm_consolidated', target_vars=target_vars, target_var_src='edlvm_consolidated_source_var', ignore_not_applicable=False)
meta = add_metadata_entry_generic(meta,  df=df, source_vars=source_vars, ess_round_label="ESS Round 10", var_name='edlvm_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = "What is the highest level of education your mother has completed?", description="see source variables")
# Export
df.to_csv(ESS10_DATA_PATH_OUT, index=False)
save_json(meta, ESS10_META_PATH_OUT)
print(f"[DONE] Wrote data:     {ESS10_DATA_PATH_OUT}")
print(f"[DONE] Wrote metadata: {ESS10_META_PATH_OUT}")



# Run the process - ESS 11
meta_11 = load_json(ESS11_META_PATH_IN)
df_11 = pd.read_csv(ESS11_DATA_PATH_IN, low_memory=False)
# Party closeness
source_vars = add_party_closeness_column(df_11, meta_records=meta_11, new_var=NEW_VAR, ignore_not_applicable=True, na_category='99.0')
meta_11 = add_metadata_entry_closeness(meta_11, df=df_11, source_vars=source_vars, ess_round_label="ESS Round 11")
# Party last voted for (taking first vote if multiple options in country)
source_vars = add_party_voted_column(df_11, meta_records=meta_11, new_var=NEW_VAR_VOTED, ignore_not_applicable=True, na_category='99.0')
meta_11 = add_metadata_entry_generic(meta_11,  df=df_11, source_vars=source_vars, ess_round_label="ESS Round 11", var_name=NEW_VAR_VOTED, target_vars = 'prtvteXX', question_category = 'politics', question_wording = 'What party did you vote for in the last election?', description="see source variables")
# Own education variable
target_vars = ['edlveat', 'edlvebe', 'edlvebg', 'edlvehr', 'edlvgcy',
               'edlvdee', 'edlvdfi', 'edlvdfr', 'educde2', # 'edudde1', 
               'edlvegr', 'edlvdahu', 'edlvdis', 'edlvdie', 'edubil1',
               'edlvfit', 'edlvelv', 'edlvdlt', 'edlveme', # 'eduail2',
               'edlvenl', 'edlveno', 'edlvipl', 'edlvept', 'edlvdrs',
               'edlvdsk', 'edlvesi', 'edlvies', 'edlvdse', 'edlvdch',
               'edlvdua', 'edubgb2' # , 'educgb1' removed due to large redundancy with edubgb2
               ]
source_vars = add_consolidated_column(df_11, meta_records=meta_11, new_var='edlv_consolidated', target_vars=target_vars, target_var_src='edlv_consolidated_source_var', ignore_not_applicable=False)
meta_11 = add_metadata_entry_generic(meta_11,  df=df_11, source_vars=source_vars, ess_round_label="ESS Round 11", var_name='edlv_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = 'What is the highest level of education you have completed?', description="see source variables")
# Partner education variable
target_vars = ['edlvpfat', 'edlvpebe', 'edlvpebg', 'edlvpehr', 'edlvpgcy',
               'edlvpdee', 'edlvpdfi', 'edlvpdfr',  'edupcde2', #'edupdde1',
               'edlvpegr', 'edlvpdahu', 'edlvpdis', 'edlvpdie', # 'edupail2',
               'edupbil1', 'edlvpfit', 'edlvpelv', 'edlvpdlt', 'edlvpeme',
               'edlvpenl', 'edlvpeno', 'edlvphpl', 'edlvpept', 'edlvpdrs',
               'edlvpdsk', 'edlvpesi', 'edlvphes', 'edlvpdse', 'edlvpdch',
               'edlvpdua', 'edupbgb2' #,  'edupcgb1', # removed due to large redundancy with edubgb2
               ]
source_vars = add_consolidated_column(df_11, meta_records=meta_11, new_var='edlvp_consolidated', target_vars=target_vars, target_var_src='edlvp_consolidated_source_var', ignore_not_applicable=False)
meta_11 = add_metadata_entry_generic(meta_11,  df=df_11, source_vars=source_vars, ess_round_label="ESS Round 11", var_name='edlvp_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = "What is the highest level of education your spouse or partner has completed?", description="see source variables")
# Father education variable 
target_vars = ['edlvfeat', 'edlvfebe', 'edlvfebg', 'edlvfehr', 'edlvfgcy',
               'edlvfdee', 'edlvfdfi', 'edlvfdfr',  'edufbde2', # 'edufcde1',
               'edlvfegr', 'edlvfdahu', 'edlvfdis', 'edlvfdie', 'edufbil1',
               'edlvffit', 'edlvfelv', 'edlvfdlt', 'edlvfeme', # 'edufail2', 
               'edlvfenl', 'edlvfeno', 'edlvfgpl', 'edlvfept', 'edlvfdrs',
               'edlvfdsk', 'edlvfesi', 'edlvfges', 'edlvfdse', 'edlvfdch',
               'edlvfdua', 'edufbgb2' #,  'edufcgb1', # removed due to large redundancy with edubgb2
               ]           
source_vars = add_consolidated_column(df_11, meta_records=meta_11, new_var='edlvf_consolidated', target_vars=target_vars, target_var_src='edlvf_consolidated_source_var', ignore_not_applicable=False)
meta_11 = add_metadata_entry_generic(meta_11,  df=df_11, source_vars=source_vars, ess_round_label="ESS Round 11", var_name='edlvf_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = "What is the highest level of education your father has completed?", description="see source variables")
# Mother education variable 
target_vars = ['edlvmeat', 'edlvmebe', 'edlvmebg', 'edlvmehr', 'edlvmgcy',
               'edlvmdee', 'edlvmdfi', 'edlvmdfr', 'edumbde2', #  'edumcde1', # removed due to large redundancy with edumbde2
               'edlvmegr', 'edlvmdahu', 'edlvmdis', 'edlvmdie', 'edumbil1',
               'edlvmfit', 'edlvmelv', 'edlvmdlt', 'edlvmeme', # 'edumail2', 
               'edlvmenl', 'edlvmeno', 'edlvmgpl', 'edlvmept', 'edlvmdrs',
               'edlvmdsk', 'edlvmesi', 'edlvmges', 'edlvmdse', 'edlvmdch',
               'edlvmdua', 'edumbgb2' #,   'edumcgb1', # removed due to large redundancy with edubgb2
               ]           
source_vars = add_consolidated_column(df_11, meta_records=meta_11, new_var='edlvm_consolidated', target_vars=target_vars, target_var_src='edlvm_consolidated_source_var', ignore_not_applicable=False)
meta_11 = add_metadata_entry_generic(meta_11,  df=df_11, source_vars=source_vars, ess_round_label="ESS Round 11", var_name='edlvm_consolidated', target_vars = target_vars, question_category = 'demographics', question_wording = "What is the highest level of education your mother has completed?", description="see source variables")


# Export
df_11.to_csv(ESS11_DATA_PATH_OUT, index=False)
save_json(meta_11, ESS11_META_PATH_OUT)
print(f"[DONE] Wrote data:     {ESS11_DATA_PATH_OUT}")
print(f"[DONE] Wrote metadata: {ESS11_META_PATH_OUT}")





# ------------------------------------ CHECKS ---------------------------------------

# Crosstable
a = df_11["edubil1"].map(code_to_key).fillna(MISSING_LABEL)
b = df_11["eduail2"].map(code_to_key).fillna(MISSING_LABEL)
ct = pd.crosstab(a, b, dropna=False, margins=True)
print("Cross-tabulation: edubil1 (rows) x eduail2 (cols)")
print(ct.to_string())

# Simple freq table
freq_table = df_11["eduail2"].value_counts(dropna=False).sort_index()
# ensure full output (no truncation)
print(freq_table.to_string())



# Simple freq table
freq_table = df_11["testji6"].value_counts(dropna=False).sort_index()
# ensure full output (no truncation)
print(freq_table.to_string())

# Crosstable
a = df_11["testji6"].map(code_to_key).fillna(MISSING_LABEL)
b = df_11["testji3"].map(code_to_key).fillna(MISSING_LABEL)
ct = pd.crosstab(a, b, dropna=False, margins=True)
print("Cross-tabulation: testji3 (rows) x eduail2 (cols)")
print(ct.to_string())


# frequency table for the new variable
print("Frequency table for ESS Round 10 party closeness variable:")
freq_table = df[NEW_VAR].value_counts(dropna=False).sort_index()
# ensure full output (no truncation)
print(freq_table.to_string())

# checking frequencies against constituent variables, merged with metadata info on categories
vars = source_vars[0:4]  # first four are country-specific party closeness vars
print("\nFrequencies for constituent source variables (labels from metadata):") 
for var in vars:
    print(f"Frequencies for source variable {var}:")
    vm = build_value_map_for_var(meta, var)  # code(str) -> label(str)

    # Normalize codes from the dataframe, then map to metadata labels.
    codes = df[var].map(code_to_key)
    mapped_labels = codes.map(lambda c: vm.get(c) if c is not None and vm.get(c) is not None else None)
    # Clean labels and ensure unmapped/missing show as MISSING_LABEL
    mapped_labels = mapped_labels.map(clean_label_for_encoding).fillna(MISSING_LABEL)

    freq_table = mapped_labels.value_counts(dropna=False).sort_index()
    print(freq_table.to_string())
    print()



print("Frequency table for ESS Round 11 party vote variable:")
freq_table = df_11[NEW_VAR_VOTED].value_counts(dropna=False).sort_index()
# ensure full output (no truncation)
print(freq_table.to_string())


vars = source_vars[0:4]  # first four are country-specific party vote vars
print("\nFrequencies for constituent source variables (labels from metadata):") 
for var in vars:
    print(f"Frequencies for source variable {var}:")
    vm = build_value_map_for_var(meta_11, var)  # code(str) -> label(str)

    # Normalize codes from the dataframe, then map to metadata labels.
    codes = df_11[var].map(code_to_key)
    mapped_labels = codes.map(lambda c: vm.get(c) if c is not None and vm.get(c) is not None else None)
    # Clean labels and ensure unmapped/missing show as MISSING_LABEL
    mapped_labels = mapped_labels.map(clean_label_for_encoding).fillna(MISSING_LABEL)

    freq_table = mapped_labels.value_counts(dropna=False).sort_index()
    print(freq_table.to_string())
    print()



freq_table = df_11['prtvtfbg'].value_counts(dropna=False).sort_index()
# ensure full output (no truncation)
print(freq_table.to_string())

for var in source_vars[4]:
    print(f"Frequencies for source variable {var}:")
    freq_table = df[var].value_counts(dropna=False).sort_index()
    print(freq_table.to_string())
    print()

source_vars = target_vars
for var in source_vars:
    print(f"Frequencies for source variable {var}:")
    freq_table = df[var].value_counts(dropna=False).sort_index()
    print(freq_table.to_string())
    print()

