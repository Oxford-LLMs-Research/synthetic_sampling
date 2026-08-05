"""Whose majority do the models follow?

The dissenter analysis establishes that predictions track the modal answer of a
respondent's own country. It does not ask whether some other population's
majority is doing the work. The natural suspicion is a Western one: that a model
asked to speak for a Nigerian or Thai respondent reproduces what a British or
American sample would have said.

This tests that directly, on the World Values Survey, the only one of our six
surveys that covers Western and non-Western countries with the same instrument.
For every target question we take the modal recorded answer of a Western bloc
(US, CA, GB, DE, NL, AU, NZ, AD) and the modal recorded answer of each
non-Western country. Restricting to the cells where those two disagree, the
question becomes which of them the model's prediction matches:

    pull = P(prediction = Western mode) - P(prediction = the country's own mode)

A positive pull is Western anchoring. It is not interpretable on its own,
because the Western mode is often simply the common answer everywhere, so the
statistic is calibrated against a null built from placebo blocs: random
eight-country blocs drawn from the non-Western countries, and one substantive
non-Western bloc (East and Southeast Asia). If the Western pull sits inside that
null, the finding is that models follow the local majority and no particular
population's.

Most profiles never name the respondent's country, so the statistic is also
split by whether the randomly drawn profile happened to contain a country or
region item, using the same flag as the country-conditioning analysis. Western
anchoring, if it existed, would have to act most strongly where the model has
been told the country.

Outputs to analysis/western_anchoring/:
  pull_by_model.csv       per model: Western pull, the pull on the subset where
                          the Western mode differs from the pooled global mode,
                          and the pull split by whether the profile named the
                          respondent's country
  pull_by_region.csv      Western pull by region of the respondent's country
  placebo_blocs.csv       one row per placebo bloc: bloc pull (13-model mean)
  cells.csv               the analysis cells, for reuse
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from repair_ids import repair

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
MR = ANALYSIS / "marginal_recovery"
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
# Which instances happened to draw a country or region item into the profile.
# Same flag the rest of the paper uses (Appendix: country conditioning).
CUE_FILE = REPO / "analysis" / "mixed_effects" / "mixed_effects_data_country_in_profile.csv"
OUT = ANALYSIS / "western_anchoring"
OUT.mkdir(exist_ok=True)

SURVEY = "wvs"
PROFILE = "s6m4"
MIN_CELL_N = 10     # respondents behind a country's modal answer
MIN_BLOC_N = 30     # respondent-answers behind a bloc's modal answer
BLOC_SIZE = 8
N_PLACEBO = 200
SEED = 20260726

WEST = ["US", "CA", "GB", "DE", "NL", "AU", "NZ", "AD"]

MODELS = [
    "deepseek", "gemma3-27b", "gpt-oss",
    "llama3.1_70b_base", "llama3.1_70b_instruct",
    "llama3.1_8b_base", "llama3.1_8b_instruct",
    "olmo3_32b_base", "olmo3_32b_dpo",
    "olmo3_7b_base", "olmo3_7b_dpo",
    "qwen3-32b", "qwen3-4b",
]

DISPLAY = {
    "qwen3-32b": "Qwen 3 32B", "deepseek": "DeepSeek-V3.1",
    "olmo3_7b_dpo": "OLMo 3 7B inst.", "qwen3-4b": "Qwen 3 4B",
    "llama3.1_8b_instruct": "Llama 3.1 8B inst.", "gemma3-27b": "Gemma 3 27B",
    "gpt-oss": "GPT-OSS 120B", "olmo3_7b_base": "OLMo 3 7B base",
    "llama3.1_70b_instruct": "Llama 3.1 70B inst.",
    "olmo3_32b_base": "OLMo 3 32B base", "olmo3_32b_dpo": "OLMo 3 32B inst.",
    "llama3.1_70b_base": "Llama 3.1 70B base",
    "llama3.1_8b_base": "Llama 3.1 8B base",
}

ASIAN_BLOC_REGIONS = ("East Asia", "Southeast Asia")


def load_cue_ids() -> set[str]:
    """example_ids whose randomly drawn profile named a country or region."""
    ids = pd.read_csv(CUE_FILE, usecols=["example_id"], dtype=str,
                      encoding="latin-1")["example_id"]
    return set(ids.unique())


def load_predictions() -> pd.DataFrame:
    """One row per (question, respondent) with the 13 models' predictions."""
    resp_country = pd.read_csv(MR / "respondent_country.csv", dtype=str, keep_default_na=False)
    base = None
    for model in MODELS:
        df = pd.read_csv(
            ANALYSIS / model / "results_data.csv",
            usecols=["example_id", "survey", "respondent_id", "target_code",
                     "profile_type", "ground_truth", "predicted"],
            dtype=str,
        )
        df = df[(df["profile_type"] == PROFILE) & (df["survey"] == SURVEY)]
        df = repair(df, verbose=False)
        df = df[["example_id", "respondent_id", "target_code",
                 "ground_truth", "predicted"]]
        df = df.rename(columns={"predicted": model})
        if base is None:
            base = df
        else:
            # ground_truth is model-independent; keep the first copy only.
            base = base.merge(df.drop(columns=["ground_truth", "example_id"]),
                              on=["respondent_id", "target_code"], how="inner")
    base["survey"] = SURVEY
    base["country_cue"] = base["example_id"].isin(load_cue_ids())
    base = base.merge(resp_country, on=["survey", "respondent_id"], how="left")
    base = base[base["country"].notna()]
    region_map = json.load(open(SCRIPTS / "country_to_region.json"))
    base["region"] = base["country"].map(region_map)
    return base


def modes_by_question(df: pd.DataFrame, min_n: int) -> pd.Series:
    """target_code -> modal recorded answer, where enough answers back it."""
    g = df.groupby("target_code")["ground_truth"]
    counts = g.size()
    modes = g.agg(lambda s: s.mode().iloc[0])
    return modes[counts >= min_n]


def local_modes(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["target_code", "country"])["ground_truth"]
    out = pd.DataFrame({"n": g.size(),
                        "local_mode": g.agg(lambda s: s.mode().iloc[0])})
    return out[out["n"] >= MIN_CELL_N].reset_index()


def pull_for_bloc(preds: pd.DataFrame, locals_: pd.DataFrame, bloc: list[str],
                  pooled: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analysis frame and per-model pull statistics for one reference bloc.

    Cells enter when the country is outside the bloc and its own modal answer
    differs from the bloc's. `pull_strict` repeats the statistic on the cells
    where the bloc's mode is also not the pooled global mode, so that matching
    it cannot be explained by the answer being common everywhere.
    """
    bloc_modes = modes_by_question(preds[preds["country"].isin(bloc)], MIN_BLOC_N)
    if bloc_modes.empty:
        return pd.DataFrame(), pd.DataFrame()

    cells = locals_[~locals_["country"].isin(bloc)].copy()
    cells["bloc_mode"] = cells["target_code"].map(bloc_modes)
    cells = cells[cells["bloc_mode"].notna()]
    cells = cells[cells["bloc_mode"] != cells["local_mode"]]
    if cells.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame = preds.merge(cells[["target_code", "country", "local_mode", "bloc_mode"]],
                        on=["target_code", "country"], how="inner")
    frame["pooled_mode"] = frame["target_code"].map(pooled)

    stats = pull_stats(frame)
    strict = frame[frame["pooled_mode"] != frame["bloc_mode"]]
    if len(strict):
        stats = stats.merge(pull_stats(strict)[["model", "pull"]]
                            .rename(columns={"pull": "pull_strict"}), on="model")
    else:
        stats["pull_strict"] = np.nan
    for name, sub in (("cue", frame[frame["country_cue"]]),
                      ("nocue", frame[~frame["country_cue"]])):
        col = f"pull_{name}"
        if len(sub):
            stats = stats.merge(pull_stats(sub)[["model", "pull"]]
                                .rename(columns={"pull": col}), on="model")
        else:
            stats[col] = np.nan
    return frame, stats


def pull_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Per model: how often the prediction matches each of the two modes."""
    rows = []
    for model in MODELS:
        hit_bloc = (frame[model] == frame["bloc_mode"]).mean()
        hit_local = (frame[model] == frame["local_mode"]).mean()
        rows.append({"model": model, "p_bloc": hit_bloc, "p_local": hit_local,
                     "pull": hit_bloc - hit_local})
    return pd.DataFrame(rows)


def cluster_bootstrap(frame: pd.DataFrame, n_boot: int = 2000,
                      seed: int = SEED) -> tuple[float, float]:
    """95% CI for the 13-model mean pull, resampling countries."""
    per_country = []
    for country, g in frame.groupby("country"):
        pulls = [(g[m] == g["bloc_mode"]).mean() - (g[m] == g["local_mode"]).mean()
                 for m in MODELS]
        per_country.append((np.mean(pulls), len(g)))
    vals = np.array([p for p, _ in per_country])
    wts = np.array([n for _, n in per_country], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boots = (vals[idx] * wts[idx]).sum(axis=1) / wts[idx].sum(axis=1)
    return tuple(np.percentile(boots, [2.5, 97.5]))


def main() -> None:
    preds = load_predictions()
    print(f"WVS rich instances: {len(preds):,}  "
          f"questions: {preds['target_code'].nunique()}  "
          f"countries: {preds['country'].nunique()}")

    locals_ = local_modes(preds)
    pooled = modes_by_question(preds, MIN_BLOC_N)

    # --- the Western bloc -------------------------------------------------
    frame, by_model = pull_for_bloc(preds, locals_, WEST, pooled)
    n_cells = frame.groupby(["target_code", "country"]).ngroups
    strict = frame[frame["pooled_mode"] != frame["bloc_mode"]]
    print(f"\nWestern bloc {WEST}")
    print(f"cells where the local mode differs from the Western mode: {n_cells} "
          f"({len(frame):,} instances, {frame['country'].nunique()} countries)")
    by_model["display"] = by_model["model"].map(DISPLAY)
    by_model = by_model.sort_values("pull", ascending=False)
    by_model.to_csv(OUT / "pull_by_model.csv", index=False)

    lo, hi = cluster_bootstrap(frame)
    mean_pull = by_model["pull"].mean()
    print(f"\n13-model mean Western pull: {mean_pull:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}] (country-clustered)")
    print(f"strict subset ({len(strict):,} instances, Western mode != pooled mode): "
          f"{by_model['pull_strict'].mean():+.4f}")

    cue = frame[frame["country_cue"]]
    nocue = frame[~frame["country_cue"]]
    lo_c, hi_c = cluster_bootstrap(cue)
    lo_n, hi_n = cluster_bootstrap(nocue)
    print(f"\ncountry named in the profile: {len(cue):,} instances "
          f"({len(cue) / len(frame):.1%}), "
          f"{cue.groupby(['target_code', 'country']).ngroups} cells, "
          f"{cue['country'].nunique()} countries; "
          f"pull {by_model['pull_cue'].mean():+.4f} [{lo_c:+.4f}, {hi_c:+.4f}]")
    print(f"country not named:           {len(nocue):,} instances; "
          f"pull {by_model['pull_nocue'].mean():+.4f} [{lo_n:+.4f}, {hi_n:+.4f}]")

    print("\nper model:")
    print(by_model[["display", "p_bloc", "p_local", "pull", "pull_strict",
                    "pull_cue", "pull_nocue"]].round(4).to_string(index=False))

    # --- by region --------------------------------------------------------
    reg_rows = []
    for region, g in frame.groupby("region"):
        pulls = [(g[m] == g["bloc_mode"]).mean() - (g[m] == g["local_mode"]).mean()
                 for m in MODELS]
        reg_rows.append({"region": region, "n_instances": len(g),
                         "n_countries": g["country"].nunique(),
                         "pull": float(np.mean(pulls))})
    by_region = pd.DataFrame(reg_rows).sort_values("pull", ascending=False)
    by_region.to_csv(OUT / "pull_by_region.csv", index=False)
    print("\nby region:")
    print(by_region.round(4).to_string(index=False))

    # --- placebo blocs ----------------------------------------------------
    non_west = sorted(set(preds["country"].unique()) - set(WEST))
    rng = np.random.default_rng(SEED)
    placebo = []
    for i in range(N_PLACEBO):
        bloc = list(rng.choice(non_west, size=BLOC_SIZE, replace=False))
        _, stats = pull_for_bloc(preds, locals_, bloc, pooled)
        if stats.empty:
            continue
        placebo.append({"bloc": "|".join(bloc), "kind": "random",
                        "pull": stats["pull"].mean(),
                        "pull_strict": stats["pull_strict"].mean(),
                        "pull_cue": stats["pull_cue"].mean(),
                        "pull_nocue": stats["pull_nocue"].mean()})

    asian = sorted(preds.loc[preds["region"].isin(ASIAN_BLOC_REGIONS),
                             "country"].unique())
    _, asian_stats = pull_for_bloc(preds, locals_, asian, pooled)
    if not asian_stats.empty:
        placebo.append({"bloc": "|".join(asian), "kind": "east_southeast_asia",
                        "pull": asian_stats["pull"].mean(),
                        "pull_strict": asian_stats["pull_strict"].mean(),
                        "pull_cue": asian_stats["pull_cue"].mean(),
                        "pull_nocue": asian_stats["pull_nocue"].mean()})

    pl = pd.DataFrame(placebo)
    pl.to_csv(OUT / "placebo_blocs.csv", index=False)
    rand = pl[pl["kind"] == "random"]
    for col, obs in (("pull", mean_pull),
                     ("pull_strict", by_model["pull_strict"].mean()),
                     ("pull_cue", by_model["pull_cue"].mean()),
                     ("pull_nocue", by_model["pull_nocue"].mean())):
        v = rand[col].dropna()
        pct = float((v < obs).mean())
        print(f"\nplacebo ({col}): {len(v)} random {BLOC_SIZE}-country blocs, "
              f"mean {v.mean():+.4f}, 2.5-97.5 pct "
              f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]; "
              f"Western observed {obs:+.4f} at the {pct:.0%} percentile")
    if not asian_stats.empty:
        print(f"\nEast/Southeast Asian bloc ({len(asian)} countries): "
              f"pull {asian_stats['pull'].mean():+.4f}, "
              f"strict {asian_stats['pull_strict'].mean():+.4f}")

    frame.drop(columns=MODELS).to_csv(OUT / "cells.csv", index=False)


if __name__ == "__main__":
    main()
