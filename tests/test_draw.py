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


def test_asian_barometer_loads_the_combined_csv(tmp_path):
    """Per-country .dta files in the folder must not shadow the combined CSV."""
    from synthetic_sampling.surveys.file_io import find_data_files
    from synthetic_sampling.surveys.registry import get_survey_config

    (tmp_path / "asiabarom_combined_datasets.csv").write_text("a\n1\n")
    (tmp_path / "australia.dta").write_bytes(b"")
    cfg = get_survey_config("asianbarometer")
    files = find_data_files(tmp_path, cfg.get_file_patterns(),
                            prefer_numeric=cfg.prefer_numeric)
    assert [f.name for f in files] == ["asiabarom_combined_datasets.csv"]
    assert get_survey_config("wvs").prefer_numeric is True


def test_case_variant_columns_are_coalesced():
    """Asian Barometer: q1 (six country files) and Q1 (three) are one item."""
    import pytest
    from synthetic_sampling.surveys import DataPaths
    from synthetic_sampling.surveys.loaders import SurveyLoader

    loader = SurveyLoader(DataPaths.default_bundled(".", "./outputs"),
                          verbose=False)
    df = pd.DataFrame({"q1": ["Good", None, None], "Q1": [None, "Bad", None],
                       "x": [1, 2, 3]})
    out = loader._merge_case_variants(df.copy())
    assert out["Q1"].tolist() == out["q1"].tolist() == ["Good", "Bad", None]
    assert out["x"].tolist() == [1, 2, 3]

    clash = pd.DataFrame({"q1": ["Good"], "Q1": ["Bad"]})
    with pytest.raises(ValueError):
        loader._merge_case_variants(clash)


def test_other_specify_variables_leave_the_metadata():
    """Afrobarometer "...OTHER" verbatim appendages are not questions; ESS
    *oth checkbox items are, and stay."""
    from synthetic_sampling.profiles.utils import load_survey_metadata
    from synthetic_sampling.surveys.harmonise import apply_harmonisation

    def codes(meta):
        return {v for b in meta.values() if isinstance(b, dict) for v in b}

    import json
    from pathlib import Path
    from synthetic_sampling.surveys import paths as _p
    meta_dir = Path(_p.__file__).parent / "metadata"

    def pulled(name):
        return json.loads((meta_dir / name).read_text(encoding="utf-8"))

    raw = pulled("pulled_metadata_afrobarometer.json")
    raw_other = {v for v in codes(raw) if v.upper().endswith("OTHER")}
    assert "Q84AOTHER" in raw_other and len(raw_other) == 8
    clean = codes(apply_harmonisation(raw, "afrobarometer"))
    assert not {v for v in clean if v.upper().endswith("OTHER")}
    assert "Q84A" in clean
    assert len(codes(raw)) - len(clean) == 8

    ess = pulled("pulled_metadata_ess10.json")
    assert codes(load_survey_metadata("afrobarometer")) == clean
    assert codes(apply_harmonisation(ess, "ess_wave_10")) >= {
        v for v in codes(ess) if v in ("dscroth", "dngoth", "medtroth")}
