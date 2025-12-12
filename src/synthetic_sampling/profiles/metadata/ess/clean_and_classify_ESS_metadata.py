import json
from pathlib import Path
from copy import deepcopy
from openai import OpenAI
import os
from tqdm import tqdm

key_path = "../../../../../_keys/openai_key.txt"
with open(key_path, "r", encoding="utf-8") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()


# Additional TO DO/observations: 
# - Replace [country] with 'your country' in questions?
# - Manually inspect scale mapping JSON for any needed corrections?
# - Consider collapsing some non-response categories?  

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ESS10_PATH_IN = Path("ess10_meta_enriched_raw.json")
ESS11_PATH_IN = Path("ess11_meta_enriched_raw.json")

ESS10_PATH_OUT = Path("ess10_meta_enriched_clean.json")
ESS11_PATH_OUT = Path("ess11_meta_enriched_clean.json")
MAPPING_PATH_OUT = Path("ess_meta_scale_mapping.json")

POOLED_META_DIR = Path("../../metadata/pulled_metadata")
ESS10_POOLED_OUT = POOLED_META_DIR / "ess10_profiles_metadata.json"
ESS11_POOLED_OUT = POOLED_META_DIR / "ess11_profiles_metadata.json"

# Check if output paths exist (false may flag path issues)
ESS10_POOLED_OUT.parent.exists()


# -------------------------------------------------------------------
# OPENAI CLIENT AND QUESTION CLEANING AND CATEGORIZATION
# -------------------------------------------------------------------
client = OpenAI()


def clean_question_for_dialog(question_text: str, response_scale: dict):
    """
    Use GPT-model to clean the question wording for natural language dialog,
    *and verify that the question wording is compatible with the response scale.*

    The model receives both the question text and the response scale, and must ensure:
        • The question naturally elicits responses consistent with the scale.
        • Placeholders like [country] are replaced.
        • Numeric/Likert scale instructions are removed and rephrased.
        • Binary categorical questions get reformulated into natural yes/no format.
        • If the question is already conversational and fits the scale, keep it mostly unchanged.

    Returns:
      (question_cleaned, notes_2)
    """
    if not question_text or not isinstance(question_text, str):
        return question_text, ""

    # Format the response scale into a readable block for GPT
    scale_lines = "\n".join([f"{code}: {label}" for code, label in response_scale.items()])
    response_scale_text = f"Response scale provided by the survey:\n{scale_lines}\n\n"

    system_prompt = f"""
You are helping to rewrite survey questions so they sound like natural questions in ordinary human conversation.
Imagine you are a friendly interviewer asking these questions to a respondent, and the respondent answers in natural language.

You will receive:
1. A survey question.
2. The full set of response categories associated with this question.

Your task is to rewrite the question ONLY if necessary, making sure the wording matches what a human would naturally ask,
while ensuring the question remains consistent with the type of responses implied by the scale.

FOLLOW THESE RULES:

1. PLACEHOLDER CLEANING
   If the question contains placeholders like [country], replace them with appropriate natural phrases
   such as "this country" or "in this country".

2. NUMERIC / LIKERT SCALES
   If the question explicitly refers to numeric or Likert instructions (e.g., 
   "from 0 to 10", "on a scale from 1 to 5", "0 means X and 10 means Y"):
       — REMOVE these instructions entirely.
       — REPHRASE the question so that it elicits a natural-language response *appropriate for the scale*.
         For example, if the scale is 0–10 levels of trust, rephrase to:
         "How much do you trust…?"

3. BINARY / APPLICABILITY QUESTIONS
   Some questions are actually about whether something applied (e.g. "Was 'Not applicable' marked...?").
   Rewrite them into natural yes/no questions such as:
       "Is this question not applicable in your case?"

4. SCALE CONSISTENCY CHECK
   Do NOT rewrite the question into a form that contradicts the response scale.
   For example:
       • If the scale has gradations (0–10 trust, 1–5 likelihood), do NOT rewrite the question as a yes/no question.
       • If the scale is yes/no (or 'applicable'/'not applicable'), ensure the question elicits such answers.
       • If the scale includes categories like “very likely”, “somewhat likely”, ensure the question invites such responses.

5. PRESERVE IF ALREADY GOOD
   If the question already reads like natural conversational language AND matches the response scale,
   leave the wording mostly unchanged.

OUTPUT FORMAT:
Return ONLY a JSON object in this form:
{{
  "question_cleaned": "<final natural-language question>",
  "notes_2": "<short description of changes made, or empty string if no change>"
}}
No additional commentary.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                # User message includes BOTH question and scale text
                {"role": "user",
                 "content": f"{response_scale_text}\nOriginal question:\n{question_text}"},
            ],
            # temperature=0,
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        cleaned = data.get("question_cleaned", "").strip() or question_text
        notes = (data.get("notes_2", "") or "").strip()

        # Normalize "no change" style outputs
        if notes.lower() in {"", "none", "no change", "no changes"}:
            notes = ""

        return cleaned, notes

    except Exception as e:
        print(f"OpenAI API error while cleaning question: {e}")
        return question_text, ""


QUESTION_CATEGORIES = [
    "civic engagement",
    "climate change",
    "covid",
    "demographics",
    "economic outlook",
    "governance",
    "media/information sources",
    "political affiliation",
    "politics",
    "trust in institutions",
    "trust in social groups",
    "other"
]

def categorize_question(question_text):
    """
    Send the raw question to GPT-4o-mini and classify it into one of the categories.
    Falls back to 'other' on any API error.
    """
    if not question_text or not isinstance(question_text, str):
        return "other"

    system_prompt = f"""
You are a classifier for European Social Survey questions. 
Assign the question to one and only one category from the list below:

{QUESTION_CATEGORIES}

Respond with only the category text, no explanation.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text}
            ],
            temperature=0
        )

        category = response.choices[0].message.content.strip().lower()

        # Enforce valid output
        if category not in QUESTION_CATEGORIES:
            return "other"

        return category

    except Exception as e:
        print(f"OpenAI API error while categorizing question: {e}")
        return "other"


def clean_all_questions(records):
    """
    Apply OpenAI cleaning to all questions, with a tqdm progress bar.
    Overwrites 'question_cleaned' and adds 'notes_2'.
    """
    print("\nCleaning question wording for natural dialog (OpenAI)...")

    for rec in tqdm(records, desc="Cleaning questions", ncols=100):
        src_q = rec.get("question_cleaned") or rec.get("question_text_raw")
        src_r = rec.get("adjusted_values") or rec.get("values_raw")
        cleaned_q, notes2 = clean_question_for_dialog(src_q, src_r)
        rec["question_cleaned"] = cleaned_q
        rec["notes_2"] = notes2

    return records


def categorize_all_questions(records):
    """
    Run OpenAI categorization on every record, using a tqdm progress bar.
    Adds 'question_category' to each record.
    """
    print("\nCategorizing questions with GPT-4o-mini...")

    for rec in tqdm(records, desc="Classifying questions", ncols=100):
        qtext = rec.get("question_cleaned") or rec.get("question_text_raw")
        category = categorize_question(qtext)
        rec["question_category"] = category

    return records


# -------------------------------------------------------------------
# SPECIAL LABEL NORMALIZATION
# -------------------------------------------------------------------
STAR_LABELS = {
    "Refusal*": "Refusal",
    "Not applicable*": "Not applicable",
    "No answer*": "No answer",
    "Don't know*": "Don't know",
    "Not available*": "Not available",
}

def normalize_label(label):
    """Normalize special missing-data labels by removing the trailing '*'."""
    if not isinstance(label, str):
        return label
    cleaned = label.strip()
    return STAR_LABELS.get(cleaned, cleaned)

def normalize_dict_labels(d):
    """Normalize all labels in a {code: label} dict."""
    return {code: normalize_label(lbl) for code, lbl in (d or {}).items()}

def normalize_record(rec):
    """Normalize values_raw and adjusted_values for a single record."""
    rec = dict(rec)  # shallow copy
    rec["values_raw"] = normalize_dict_labels(rec.get("values_raw") or {})
    rec["adjusted_values"] = normalize_dict_labels(rec.get("adjusted_values") or {})
    return rec

# -------------------------------------------------------------------
# JSON  HELPERS
# -------------------------------------------------------------------
def load_json_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json_records(records, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
        
        
def category_to_key(category: str) -> str:
    """
    Convert a category label like 'civic engagement' or
    'media/information sources' into a JSON-friendly key
    like 'civic_engagement' or 'media_information_sources'.
    """
    if not category:
        return "other"
    key = category.strip().lower()
    key = key.replace("/", "_")
    key = key.replace(" ", "_")
    return key


def build_profiles_structure(records):
    """
    Reformat records into the nested structure:

    {
      "<category_key>": {
        "<variable_name>": {
          "description": ...,
          "question": ...,
          "values": {...}
        },
        ...
      },
      ...
    }
    """
    profiles = {}

    for rec in records:
        raw_cat = rec.get("question_category") or "other"
        cat_key = category_to_key(raw_cat)

        var_name = rec.get("variable_name")
        if not var_name:
            continue

        description = rec.get("description") or ""
        question = rec.get("question_cleaned") or rec.get("question_text_raw") or ""
        values = rec.get("adjusted_values") or {}

        if cat_key not in profiles:
            profiles[cat_key] = {}

        profiles[cat_key][var_name] = {
            "description": description,
            "question": question,
            "values": values
        }

    return profiles

# -------------------------------------------------------------------
# SCALE-LEVEL MAPPING
# -------------------------------------------------------------------
def make_scale_key(values_raw_dict):
    """
    Create a canonical, hashable representation of a response scale.

    We use a JSON dump with sorted keys so that logically identical dicts
    (same codes and labels) produce the same key, regardless of input order.
    """
    return json.dumps(values_raw_dict, sort_keys=True, ensure_ascii=False)

def build_scale_mapping(records, verbose=True):
    """
    Build a mapping from an entire values_raw dict to an entire adjusted_values dict.

    mapping: { scale_key (canonical string) -> adjusted_values_dict }

    If the same values_raw pattern (scale_key) appears with a different adjusted_values
    dict, we keep the first and report a warning.
    """
    mapping = {}
    conflicts = []

    for rec in records:
        var_name = rec.get("variable_name", "<unknown>")
        values_raw = rec.get("values_raw") or {}
        adjusted_values = rec.get("adjusted_values") or {}

        # Only build mapping if there ARE response categories
        if not values_raw:
            continue

        # Canonical key for the response scale
        scale_key = make_scale_key(values_raw)

        # Canonical adjusted_values for comparison
        # (sorted keys to make equality checks reliable)
        canonical_adj = json.loads(
            json.dumps(adjusted_values, sort_keys=True, ensure_ascii=False)
        )

        if scale_key in mapping:
            existing_canonical_adj = json.loads(
                json.dumps(mapping[scale_key], sort_keys=True, ensure_ascii=False)
            )
            if existing_canonical_adj != canonical_adj:
                conflicts.append({
                    "variable_name": var_name,
                    "values_raw": values_raw,
                    "existing_adjusted": mapping[scale_key],
                    "new_adjusted": adjusted_values,
                })
                # Keep the FIRST mapping; do not overwrite.
        else:
            # Store a deep copy so later mutations don't affect the mapping
            mapping[scale_key] = deepcopy(adjusted_values)

    if verbose:
        if conflicts:
            print("\n*** WARNING: Conflicting scale mappings detected ***")
            for c in conflicts:
                print(
                    f"Scale used in variable '{c['variable_name']}' has already "
                    f"been mapped once but appears with a different adjusted_values. "
                    f"Keeping FIRST mapping for that scale."
                )
            print("*** End of conflict report ***\n")
        else:
            print("No conflicting scale mappings detected.")

    return mapping

def apply_scale_mapping(records, mapping, verbose=True):
    """
    For each record, replace its adjusted_values with the canonical adjusted_values
    dict corresponding to its values_raw scale (if present in mapping).

    If a scale is not in the mapping, we keep its existing adjusted_values as-is.
    """
    missing_scales = 0

    for rec in records:
        values_raw = rec.get("values_raw") or {}
        if not values_raw:
            continue

        scale_key = make_scale_key(values_raw)
        if scale_key in mapping:
            rec["adjusted_values"] = deepcopy(mapping[scale_key])
        else:
            missing_scales += 1

    if verbose:
        if missing_scales:
            print(
                f"{missing_scales} records had values_raw scales that were not in "
                f"the mapping; their adjusted_values were left unchanged."
            )
        else:
            print("All records' scales were found in the mapping.")

    return records

# -------------------------------------------------------------------
# Run the main process
# -------------------------------------------------------------------

print(f"Loading ESS10 from {ESS10_PATH_IN}")
ess10_raw = load_json_records(ESS10_PATH_IN)

print(f"Loading ESS11 from {ESS11_PATH_IN}")
ess11_raw = load_json_records(ESS11_PATH_IN)

# Normalize labels (remove '*' for the specified categories) before mapping
ess10 = [normalize_record(r) for r in ess10_raw]
ess11 = [normalize_record(r) for r in ess11_raw]

all_records = ess10 + ess11
print(f"Total records (ESS10 + ESS11): {len(all_records)}")

# Build scale-level mapping
print("Building unified scale-level values_raw -> adjusted_values mapping...")
scale_mapping = build_scale_mapping(all_records, verbose=True)

# Save scale mapping for inspection / manual editing - ONLY NEEDED ONCE
# Note: keys are canonical JSON strings of values_raw dicts
# print(f"Saving scale mapping to {MAPPING_PATH_OUT}")
# with MAPPING_PATH_OUT.open("w", encoding="utf-8") as f:
#     json.dump(scale_mapping, f, ensure_ascii=False, indent=2)

# Load scale mapping from file (if needed)
print(f"Loading scale mapping from {MAPPING_PATH_OUT}")
with MAPPING_PATH_OUT.open("r", encoding="utf-8") as f:
    scale_mapping = json.load(f)

# Apply mapping back to each dataset
print("Applying scale mapping to ESS10...")
ess10_clean = apply_scale_mapping(ess10, scale_mapping, verbose=True)

print("Applying scale mapping to ESS11...")
ess11_clean = apply_scale_mapping(ess11, scale_mapping, verbose=True)


# Classify question topic based on question text
all_clean = ess10_clean + ess11_clean

# Additional question cleanding with OpenAI api
all_clean = clean_all_questions(all_clean)

all_clean = categorize_all_questions(all_clean)

# Split back into separate datasets
ess10_final = all_clean[: len(ess10_clean)]
ess11_final = all_clean[len(ess10_clean):]

# Remove admin variables

admin_vars_10 = {
    "name", "essround", "edition", "proddate",  "idno", "essround", "pweight", "anweight", "pspwght", "dweight",
    "admit", "showcv", "regunit", "vdcond", "vdtype", "vdtpsvre", "vdtpitre", "vdtpscre", "vdtpaure",
    "vdtpvire", "vdtpoire", "vdtpntre", "vdtpapre", "vdtprere", "vdtpdkre", "vdtpnare", "inwds", "ainws",
    "ainwe", "binwe", "cinwe", "dinwe", "finwe", "ginwe", "hinwe", "iinwe", "kinwe", "recon", "vinwe", "inwde", 
    "jinws", "jinwe", "inwtm", "mode", "domain", "prob", "stratum", "psu", "vdovexre"}
ess10_final = [r for r in ess10_final if r.get("variable_name") not in admin_vars_10]

admin_vars_11 = {
    "name", "essround", "edition", "proddate",  "idno", "essround", "pweight", "anweight", "pspwght", "dweight",
    "admit", "showcv", "regunit", "vdcond", "vdtype", "vdtpsvre", "vdtpitre", "vdtpscre", "vdtpaure",
    "vdtpvire", "vdtpoire", "vdtpntre", "vdtpapre", "vdtprere", "vdtpdkre", "vdtpnare", "inwds", "ainws",
    "ainwe", "binwe", "cinwe", "dinwe", "finwe", "ginwe", "hinwe", "iinwe", "einwe", "recon", "kinwe", "rinwe", "vinwe", "inwde", 
    "jinws", "jinwe", "inwtm", "mode", "domain", "prob", "stratum", "psu"}
ess11_final = [r for r in ess11_final if r.get("variable_name") not in admin_vars_11]


# Save cleaned metadata
print(f"Saving cleaned ESS10 metadata to: {ESS10_PATH_OUT}")
save_json_records(ess10_final, ESS10_PATH_OUT)

print(f"Saving cleaned ESS11 metadata to: {ESS11_PATH_OUT}")
save_json_records(ess11_final, ESS11_PATH_OUT)

# Load cleaned data for verification
ess10_final = load_json_records(ESS10_PATH_OUT)
ess11_final = load_json_records(ESS11_PATH_OUT)


# Reformat and export into standardized format used for further analysis

POOLED_META_DIR.mkdir(parents=True, exist_ok=True)

ess10_profiles = build_profiles_structure(ess10_final)
ess11_profiles = build_profiles_structure(ess11_final)

print(f"Saving ESS10 profiles metadata → {ESS10_POOLED_OUT}")
with ESS10_POOLED_OUT.open("w", encoding="utf-8") as f:
    json.dump(ess10_profiles, f, ensure_ascii=False, indent=2)

print(f"Saving ESS11 profiles metadata → {ESS11_POOLED_OUT}")
with ESS11_POOLED_OUT.open("w", encoding="utf-8") as f:
    json.dump(ess11_profiles, f, ensure_ascii=False, indent=2)


print("Done.")




