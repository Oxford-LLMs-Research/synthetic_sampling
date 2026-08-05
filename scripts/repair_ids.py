"""Repair corrupted respondent_id / target_code in analysis/<model>/results_data.csv.

The pipeline that wrote results_data.csv split example_id, whose format is

    {survey}_{respondent_id}_{target_code}_{profile_type}

on underscores at a fixed position. That is wrong whenever the target code
itself contains an underscore, which is the case for 23 of the 36 Arab
Barometer targets (Q725_5, Q201B_13, Q2061A_KUW, ...) and a couple of WVS
targets. For those rows the trailing fragment became target_code and the
leading part was absorbed into respondent_id:

    example_id  arabbarometer_700512_Q725_5_s6m4
    stored      respondent_id "700512_Q725", target_code "5"
    truth       respondent_id "700512",      target_code "Q725_5"

Two consequences: the rows fail every join on respondent_id or target_code, and
several distinct questions collapse onto the same fragment ("2" is Q204_2,
Q277_2, Q534_2 and Q550A_2 at once), so any group-by on target_code pools
unrelated questions.

example_id is intact, so the damage is fully recoverable. We resolve each row
against the known target list, longest code first, which is unambiguous because
no valid target code is a suffix of another after the underscore separator.

mixed_effects_data.csv is NOT affected and needs no repair.

Usage:
    from repair_ids import repair
    df = repair(pd.read_csv(..., usecols=[..., "example_id"]))
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
TARGET_SOURCE = ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv"


def load_valid_targets() -> dict[str, list[str]]:
    """survey -> target codes, longest first."""
    q = pd.read_csv(TARGET_SOURCE, usecols=["survey", "target_code"], dtype=str)
    out: dict[str, list[str]] = {}
    for survey, g in q.groupby("survey"):
        out[survey] = sorted(g["target_code"].unique(), key=len, reverse=True)
    return out


def repair(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fix target_code and respondent_id in place from example_id.

    Requires columns example_id, survey, profile_type. Rows whose stored
    target_code is already valid are left untouched.
    """
    valid = load_valid_targets()
    df = df.copy()

    ok = pd.Series(False, index=df.index)
    for survey, codes in valid.items():
        m = df["survey"] == survey
        ok |= m & df["target_code"].isin(codes)

    bad = ~ok
    n_bad = int(bad.sum())
    if n_bad == 0:
        if verbose:
            print("  repair_ids: nothing to fix")
        return df

    fixed_t, fixed_r, unresolved = [], [], 0
    for idx in df.index[bad]:
        eid = df.at[idx, "example_id"]
        survey = df.at[idx, "survey"]
        profile = df.at[idx, "profile_type"]
        stem = eid[: -(len(profile) + 1)] if eid.endswith(f"_{profile}") else eid
        target = respondent = None
        for code in valid.get(survey, []):
            suffix = f"_{code}"
            if stem.endswith(suffix):
                target = code
                respondent = stem[: -len(suffix)]
                if respondent.startswith(f"{survey}_"):
                    respondent = respondent[len(survey) + 1:]
                break
        if target is None:
            unresolved += 1
            fixed_t.append(df.at[idx, "target_code"])
            fixed_r.append(df.at[idx, "respondent_id"])
        else:
            fixed_t.append(target)
            fixed_r.append(respondent)

    df.loc[bad, "target_code"] = fixed_t
    if "respondent_id" in df.columns:
        df.loc[bad, "respondent_id"] = fixed_r

    if verbose:
        print(f"  repair_ids: repaired {n_bad - unresolved:,} of {n_bad:,} rows "
              f"({unresolved:,} unresolved)")
    return df


if __name__ == "__main__":
    # Self-check on one model: every target code must become valid.
    df = pd.read_csv(ANALYSIS / "qwen3-32b" / "results_data.csv",
                     usecols=["example_id", "survey", "respondent_id",
                              "target_code", "profile_type"], dtype=str)
    print(f"loaded {len(df):,} rows")
    out = repair(df)
    valid = load_valid_targets()
    still_bad = 0
    for survey, codes in valid.items():
        m = out["survey"] == survey
        still_bad += int((m & ~out["target_code"].isin(codes)).sum())
    print(f"rows with an invalid target code after repair: {still_bad:,}")
    arab = out[(out["survey"] == "arabbarometer") & (out["profile_type"] == "s6m4")]
    print(f"arabbarometer rich rows: {len(arab):,}, "
          f"distinct targets: {arab['target_code'].nunique()}")
    print(arab[["example_id", "respondent_id", "target_code"]].head(4).to_string(index=False))
