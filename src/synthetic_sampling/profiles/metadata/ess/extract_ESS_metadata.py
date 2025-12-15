import pandas as pd
from bs4 import BeautifulSoup
import os
from openai import OpenAI
from tqdm import tqdm
tqdm.pandas()
import json

# MAKE SURE API KEY IS AVAILABLE BEFORE RUNNING
key_path = "../../../../../_keys/openai_key.txt"
with open(key_path, "r", encoding="utf-8") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()


# ---------- PART 1: Parse ESS codebook html ----------

def parse_ess_codebook(html_path: str) -> pd.DataFrame:
    """
    Parse the ESS codebook HTML file and return a DataFrame with:
        - variable_name
        - question_text_raw
        - values (dict)
    """

    # ---- Ensure UTF-8 reading ----
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    records = []

    # All variable sections correspond to <h3 id="varname">
    for h3 in soup.find_all("h3"):
        var_id = h3.get("id")
        if not var_id:
            continue  # skip any h3 without an id

        container_div = h3.find_parent("div")
        if container_div is None:
            continue

        # ---- Extract question text ----
        question_text_parts = []
        for sibling in h3.next_siblings:
            # stop once we hit the answer options table wrapper
            if getattr(sibling, "name", None) == "div" and \
               "data-table" in (sibling.get("class") or []):
                break

            if getattr(sibling, "name", None):
                text = sibling.get_text(" - ", strip=True)
            else:
                text = str(sibling).strip()

            if text:
                question_text_parts.append(text)

        question_text_raw = "\n".join(question_text_parts)

        # ---- Extract value labels ----
        values = {}
        data_table_div = container_div.find("div", class_="data-table")
        if data_table_div:
            table = data_table_div.find("table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        code = cells[0].get_text(strip=True)
                        label = cells[1].get_text(" ", strip=True)
                        if code:
                            values[code] = label

        # ---- Append record ----
        records.append({
            "variable_name": var_id,
            "question_text_raw": question_text_raw,
            "values_raw": values
        })

    # ---- Construct DataFrame with explicit columns ----
    df = pd.DataFrame(records)
    df = df[["variable_name", "question_text_raw", "values_raw"]]  # ensures correct column order

    return df


# ---------- PART 2: Call OpenAI (gpt-5-mini) to enrich each row ----------

# Initialize OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI()

SYSTEM_PROMPT = """
You are a careful data transformation assistant working with survey codebooks.
For each survey variable, you must produce structured JSON with:
- description: brief description (2-6 words) of what the question measures.
- question_cleaned: a natural, conversational version of the question text.
- adjusted_values: a possibly simplified mapping of response codes to labels.
- notes: short description of any modifications you made (e.g., removed artifacts,
         collapsed scales, consolidated categories). If nothing changed, say "None".
Follow these rules:

1. Description:
   - Keep it concise, 2-6 words.
   - Describe the underlying construct, e.g. "trust in government", "life satisfaction", "party identification".
   - If not provided, infer a brief, descriptive summary from the question (e.g., "respondent's gender", "political party preference")

2. Question wording (question_cleaned):
   - Remove survey artifacts and procedural language like:
     "Looking at this card", "Please use the card", "I will now read out", "Interviewer: read out", etc.
   - Rephrase into natural, direct question format as if asked in person-to-person conversation
   - Remove interviewer-only instructions and routing info.
   - Keep the core meaning and reference period (e.g. "in the last 12 months").
   - Document significant changes in the "notes" field

3. Values (adjusted_values):
   - Start from the original code-to-label mapping.
   - Preserve special codes for missing data or non-substantive responses (e.g. "Refusal", "Don't know",
     "No answer", "Not applicable", "Missing") such that all original variable values can be mapped to new labels.
   - For long numerical scales (e.g., 0–10 or 1–11 with only endpoints labelled or clearly monotone):
       * Collapse to 5-point scale natural language labels like
         "Strongly disagree", "Somewhat disagree", "Neither agree nor disagree",
         "Somewhat agree", "Strongly agree", etc.
       * Assign contiguous ranges of codes to each category.
       * The final mapping MUST cover all original codes, only the labels are collapsed.
   - Example for 0-10 scale with endpoints 'Strongly oppose' and 'Strongly support':
     • 0-1 → "Strongly oppose"
     • 2-3 → "Somewhat oppose"  
     • 4-6 → "Neither oppose nor support"
     • 7-8 → "Somewhat support"
     • 9-10 → "Strongly support"
   - Do not collapse scales/categories that are clearly non-monotone or non-ordinal.
   - Document the collapse in "notes": "Collapsed from 11-point to 5-point scale"

4. Notes:
   - Briefly describe any important transformations (significant rewording of question text, 
   scale collapses or category modifications, any other transformations made for clarity), such as:
     "Removed card instructions and interviewer notes."
     "Collapsed 11-point scale into 5 categories."
     "Simplified technical phrasing of response labels."
   - Add any other transformations made for clarity
   - If you made no meaningful changes, return "None".
   
   
# ---------- Stylized Examples for clarity ----------

Example 1 - Standard categorical question:

Input:
Variable: GENDER
Question: Respondent's gender
Values: 1, 2
Value Labels: 1=Male, 2=Female

Output:
{
  "GENDER": {
    "description": "Respondent's gender",
    "question_cleaned": "What is your gender?",
    "adjusted_values": {
      "1": "Male",
      "2": "Female"
    }
  }
}

---

Example 2 - Question with survey artifacts:

Input:
Variable: pxmltn
Question: Looking at this card, can you tell me which political party you identify with most?
Values: 1-5, 99
Value Labels: 1=Democrat, 2=Republican, 3=Independent, 4=Other party, 5=No party, 99=Refused

{
    "description": "Political party identification",
    "question_cleaned": "Which political party do you identify with most?",
    "adjusted_values": {
      "1": "Democrat",
      "2": "Republican",
      "3": "Independent",
      "4": "Other party",
      "5": "No party",
      "99": "Refused"
    },
    "notes": "Removed survey artifact: 'Looking at this card'"
}

---

Example 3 - Likert scale requiring collapse:

Input:
Variable: TRUST_GOV
Question: On a scale from 0 to 10, how much do you trust the government?
Values: 0-10
Value Labels: 0=Do not trust at all, 10=Trust completely

Output:
{
    "description": "Trust in government",
    "question_cleaned": "How much do you trust the government?",
    "adjusted_values": {
      "0": "Do not trust at all",
      "1": "Do not trust at all",
      "2": "Slightly trust",
      "3": "Slightly trust",
      "4": "Somewhat trust",
      "5": "Somewhat trust",
      "6": "Somewhat trust",
      "7": "Mostly trust",
      "8": "Mostly trust",
      "9": "Trust completely",
      "10": "Trust completely"
    },
    "notes": "Collapsed 11-point scale to 5 meaningful categories for natural language labeling"
}

---

Example 4 - Likert scale requiring collapse II:

Variable: testjc35
Question: "How likely, large numbers of people limit energy use
STILL CARD 30
How likely do you think it is that large numbers of people will actually limit their energy use to try to reduce climate change?"
Values: 0-10
Value Labels: 0=Not at all likely, 10=Extremely likely


Output:
{
    "description": "Limit energy use for climate change",
    "question_cleaned": "How likely do you think it is that large numbers of people will actually limit their energy use to try to reduce climate change?",
    "adjusted_values": {
      "0": "Not at all likely",
      "1": "Not at all likely",
      "2": "Not likely",
      "3": "Not likely",
      "4": "Somewhat likely",
      "5": "Somewhat likely",
      "6": "Somewhat likely",
      "7": "Very likely",
      "8": "Very likely",
      "9": "Extremely likely",
      "10": "Extremely likely"
    },
    "notes": "Collapsed 11-point scale to 5 meaningful categories for natural language labeling"
}   
---

Example 5 - Question wording with interviewer instructions:

Variable: jbexeref
Question: "In any job, ever exposed to: refusal
CARD 67
And in any of the jobs you have ever had, which of the things on this card were you exposed to? Refusal
INTERVIEWER PROBE: Which others? CODE ALL THAT APPLY"
Values: 0-1
Value Labels: 0=Not marked, 1=Marked

Output:
{
    "description": "Job exposure: refusal",
    "question_cleaned": "And in any of the jobs you have ever had, which of the things on this card were you exposed to? Refusal",
    "adjusted_values": {
      "0": "Not marked",
      "1": "Marked"
    },
    "notes": "Removed interviewer instructions and card references."
}
   
"""

def build_row_prompt(variable_name: str, question_text_raw: str, values: dict) -> str:
    """
    Build the user prompt for a single row to send to the model.
    """
    values_json = json.dumps(values, ensure_ascii=False, indent=2)
    question_text_raw = question_text_raw or ""
    return f"""
You are given a single survey variable from a codebook.

Variable name:
{variable_name}

Raw question text (may contain survey artifacts, card instructions, interviewer notes, etc.):
\"\"\"{question_text_raw}\"\"\"

Original value mapping (code -> label) as JSON:
{values_json}

Your task: Return a JSON object with exactly these keys:
- "description": string
- "question_cleaned": string
- "adjusted_values": object (mapping code string -> label string)
- "notes": string

Do not wrap the result in an outer object keyed by the variable name.
Return only a single JSON object with those four keys.
"""

def transform_row_with_openai(row: pd.Series) -> pd.Series:
    """
    Call gpt-5-mini for a single row and return the four new columns.
    """
    variable_name = str(row["variable_name"])
    question_text_raw = row.get("question_text_raw") or ""
    values = row.get("values_raw") or {}

    user_prompt = build_row_prompt(variable_name, question_text_raw, values)

    response = client.chat.completions.create(
        model="gpt-5-mini",
        # temperature=0, # 5 mini does not support temperature param
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if model output is not strict JSON; keep things non-crashy.
        parsed = {
            "description": None,
            "question_cleaned": None,
            "adjusted_values": {},
            "notes": f"Failed to parse JSON from model output: {content[:200]}...",
        }

    # Ensure all keys exist
    description = parsed.get("description")
    question_cleaned = parsed.get("question_cleaned")
    adjusted_values = parsed.get("adjusted_values", {})
    notes = parsed.get("notes")

    return pd.Series(
        {
            "description": description,
            "question_cleaned": question_cleaned,
            "adjusted_values": adjusted_values,
            "notes": notes,
        }
    )
    
# Safer wrapper so every row returns something, even on API failure
def safe_transform_row_with_openai(row: pd.Series) -> pd.Series:
    try:
        return transform_row_with_openai(row)
    except Exception as e:
        # Fallback: keep alignment, mark failure in notes
        return pd.Series(
            {
                "description": None,
                "question_cleaned": None,
                "adjusted_values": {},
                "notes": f"OpenAI API error: {e}"
            }
        )

def enrich_with_openai(df: pd.DataFrame, skip_n: int = 0, max_retries: int = 1) -> pd.DataFrame:
    """
    Enrich df with OpenAI-derived columns.
    - `skip_n` allows skipping the first n rows (placeholders added automatically).
    - `max_retries` controls how many times rows with OpenAI errors are retried.
    """

    # --- 1. Placeholder rows for skipped observations ---
    if skip_n > 0:
        placeholder_rows = pd.DataFrame(
            [
                {
                    "description": None,
                    "question_cleaned": None,
                    "adjusted_values": {},
                    "notes": f"Skipped (skip_n={skip_n})"
                }
                for _ in range(skip_n)
            ],
            index=df.index[:skip_n]
        )
    else:
        placeholder_rows = pd.DataFrame(
            columns=["description", "question_cleaned", "adjusted_values", "notes"]
        )

    # --- 2. Apply OpenAI enrichment for rows after skip_n (first pass) ---
    enriched_rows = []
    if skip_n < len(df):
        for idx, row in tqdm(
            df.iloc[skip_n:].iterrows(),
            total=len(df) - skip_n,
            desc="OpenAI enrichment (first pass)"
        ):
            enriched = safe_transform_row_with_openai(row)
            enriched_rows.append(enriched)

        enriched_rows_df = pd.DataFrame(enriched_rows, index=df.index[skip_n:])
    else:
        enriched_rows_df = pd.DataFrame(
            columns=["description", "question_cleaned", "adjusted_values", "notes"]
        )

    # --- 3. Combine skipped + enriched results into one table of new cols ---
    new_cols = pd.concat([placeholder_rows, enriched_rows_df], axis=0).sort_index()

    # --- 4. Retry rows with OpenAI errors / parse failures, if requested ---
    for attempt in range(max_retries):
        # Identify rows to retry: notes contain OpenAI error or parse failure
        if "notes" not in new_cols.columns:
            break

        error_mask = new_cols["notes"].astype(str).str.contains(
            "OpenAI API error|Failed to parse JSON", case=False, na=False
        )
        # Optional: avoid retrying the same rows indefinitely if notes already mention "Retry"
        error_mask &= ~new_cols["notes"].astype(str).str.contains(
            "Retry", case=False, na=False
        )

        idx_to_retry = new_cols.index[error_mask]
        if len(idx_to_retry) == 0:
            break  # nothing left to retry

        tqdm_desc = f"Retrying failed rows (attempt {attempt + 1})"
        for idx in tqdm(idx_to_retry, desc=tqdm_desc):
            original_row = df.loc[idx]
            try:
                # Fresh call without the safe wrapper so we can distinguish this retry
                retry_result = transform_row_with_openai(original_row)
                # Overwrite previous error result
                for col in ["description", "question_cleaned", "adjusted_values", "notes"]:
                    new_cols.at[idx, col] = retry_result.get(col)
            except Exception as e:
                # If retry also fails, append info to notes but keep existing values
                prev_notes = new_cols.at[idx, "notes"]
                if prev_notes is None:
                    prev_notes = ""
                new_cols.at[idx, "notes"] = (
                    f"{prev_notes} | Retry {attempt + 1} failed: OpenAI API error: {e}"
                )

    # --- 5. Merge safely with original df (preserving row alignment) ---
    df_enriched = pd.concat(
        [df.reset_index(drop=True), new_cols.reset_index(drop=True)],
        axis=1
    )

    return df_enriched



# ESS 11 - pot. TO DO: grab html from Google Drive directly
html_file = "../../../data/ess/ESS11_codebook.html"
df_base = parse_ess_codebook(html_file)


# Enrich with OpenAI-derived columns
# (Make sure OPENAI_API_KEY is set in your environment.)
df_enriched = enrich_with_openai(df_base, skip_n=5, max_retries=3)

# Inspect and/or save; UTF-8 ensured on write
print(df_enriched.head())

df_enriched.to_json(
    "ess11_meta_enriched_raw.json",
    orient="records",
    force_ascii=False,
    indent=2,
)


# ESS 10 - pot. TO DO: grab html from Google Drive directly
html_file_10 = "../../../data/ess/ESS10_codebook.html"
df_base_10 = parse_ess_codebook(html_file_10)

df_enriched_10 = enrich_with_openai(df_base_10, skip_n=5, max_retries=3)

print(df_enriched_10.head())

df_enriched_10.to_json(
    "ess10_meta_enriched_raw.json",
    orient="records",
    force_ascii=False,
    indent=2,
)


