"""Nested stratified respondent draw for the second main run.

Every respondent gets a rank inside their source-country cell from a seeded
hash of (seed, survey, respondent id). A floor of n is the respondents with
rank < n in each cell, so a smaller floor is always a strict prefix of a
larger one: scoring the first 50 per cell and extending to 100 later reuses
every scored instance, and the union is still a simple random sample within
each cell. The rank depends on nothing but the three hashed values, so it
does not move with row order, with the other respondents in the file, or
with the floor asked for.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import pandas as pd


def draw_key(seed: int, survey_id: str, respondent_id: Any) -> str:
    return hashlib.sha256(
        f"{seed}_{survey_id}_{respondent_id}".encode()).hexdigest()


def nested_cell_ranks(
    respondent_ids: Iterable[Any],
    countries: Iterable[Any],
    survey_id: str,
    seed: int,
) -> pd.DataFrame:
    """Rank every respondent within their country cell.

    Respondents with an ambiguous (duplicated) id or no country leave the
    pool, the convention the cell census and the XGB full fit use. Returns
    columns survey, respondent_id, country, rank (0-based within cell),
    sorted by country then rank.
    """
    df = pd.DataFrame({"respondent_id": list(respondent_ids),
                       "country": list(countries)})
    df = df[df["country"].notna()]
    df = df[~df["respondent_id"].astype(str).duplicated(keep=False)]
    df["country"] = df["country"].astype(str)
    df["_key"] = [draw_key(seed, survey_id, r) for r in df["respondent_id"]]
    df = df.sort_values(["country", "_key"], kind="mergesort")
    df["rank"] = df.groupby("country").cumcount()
    df.insert(0, "survey", survey_id)
    return df.drop(columns="_key").reset_index(drop=True)


def take_floor(ranks: pd.DataFrame, floor: int,
               start: int = 0) -> pd.DataFrame:
    """Respondents with start <= rank < floor in every cell (start > 0 gives
    the extension tranche on top of an already-scored smaller floor)."""
    return ranks[(ranks["rank"] >= start) & (ranks["rank"] < floor)]
