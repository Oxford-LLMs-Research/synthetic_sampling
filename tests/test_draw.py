"""The nested per-country draw: prefix property, determinism, hygiene."""

import numpy as np
import pandas as pd

from synthetic_sampling.surveys.draw import nested_cell_ranks, take_floor


def _frame():
    rng = np.random.RandomState(0)
    countries = ["A"] * 300 + ["B"] * 120 + ["C"] * 60
    ids = [f"r{i}" for i in range(len(countries))]
    order = rng.permutation(len(ids))
    return pd.DataFrame({"id": np.array(ids)[order],
                         "country": np.array(countries)[order]})


def test_floor_is_exact_per_cell_and_nested():
    df = _frame()
    ranks = nested_cell_ranks(df["id"], df["country"], "wvs", 42)
    f50, f100 = take_floor(ranks, 50), take_floor(ranks, 100)
    assert f50.groupby("country").size().to_dict() == {"A": 50, "B": 50, "C": 50}
    # C has 60 respondents: a floor above the cell takes the whole cell.
    assert f100.groupby("country").size().to_dict() == {"A": 100, "B": 100, "C": 60}
    assert set(f50["respondent_id"]) < set(f100["respondent_id"])
    ext = take_floor(ranks, 100, start=50)
    assert not set(ext["respondent_id"]) & set(f50["respondent_id"])
    assert set(ext["respondent_id"]) | set(f50["respondent_id"]) == set(
        f100["respondent_id"])


def test_rank_ignores_row_order_and_other_respondents():
    df = _frame()
    a = nested_cell_ranks(df["id"], df["country"], "wvs", 42)
    shuffled = df.sample(frac=1.0, random_state=1)
    b = nested_cell_ranks(shuffled["id"], shuffled["country"], "wvs", 42)
    assert a.equals(b)
    # Dropping respondents never reorders the ones that remain.
    kept = df[df["id"] != a["respondent_id"].iloc[0]]
    c = nested_cell_ranks(kept["id"], kept["country"], "wvs", 42)
    assert (a["respondent_id"].iloc[1:].tolist()
            == c["respondent_id"].tolist())
    # Seed and survey both move the draw.
    d = nested_cell_ranks(df["id"], df["country"], "wvs", 7)
    e = nested_cell_ranks(df["id"], df["country"], "afrobarometer", 42)
    assert a["respondent_id"].tolist() != d["respondent_id"].tolist()
    assert a["respondent_id"].tolist() != e["respondent_id"].tolist()


def test_duplicate_ids_and_missing_countries_leave_the_pool():
    ranks = nested_cell_ranks(["x", "x", "y", "z"], ["A", "A", "A", None],
                              "wvs", 42)
    assert ranks["respondent_id"].tolist() == ["y"]
