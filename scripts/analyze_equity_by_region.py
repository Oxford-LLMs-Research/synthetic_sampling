"""Does the demographic null in the equity audit hold region by region?

analyze_equity_audit.py estimates each demographic contrast within
(survey, question, country) cells and finds every one within 0.005 of zero.
That is an average of country-specific contrasts, so a gap that favours men in
one part of the world and women in another would cancel and still report zero.

This re-estimates the same within-cell contrasts separately by world region and
asks whether their dispersion exceeds sampling noise. The comparison of
interest is the spread across regions, not any single region, so the printed
summary reports the range, how many regional intervals exclude zero, and how
many would be expected to by chance.

Outputs to analysis/equity_audit/:
  subgroup_adjusted_by_region.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_equity_audit import ANALYSIS, DIMS, MIN_N, OUT, build_instance_table

SCRIPTS = Path(__file__).resolve().parent
N_BOOT = 1000
SEED = 42
MIN_REGION_RESPONDENTS = 40   # respondents contributing to a region's contrast


def cell_demeaned(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["country"].notna()].copy()
    cell = ["survey", "target_code", "country"]
    df["cell_mean"] = df.groupby(cell)["norm_acc"].transform("mean")
    df["cell_n"] = df.groupby(cell)["norm_acc"].transform("size")
    df = df[df["cell_n"] >= MIN_N]
    df["dev"] = df["norm_acc"] - df["cell_mean"]
    region_map = json.load(open(SCRIPTS / "country_to_region.json"))
    df["region"] = df["country"].map(region_map)
    return df[df["region"].notna() & (df["region"] != "Unknown")]


def contrast(sub: pd.DataFrame, dim: str, ref: str, lv: str,
             rng: np.random.Generator) -> tuple[float, float, float, int]:
    """Within-cell contrast lv - ref, with a respondent-cluster bootstrap."""
    agg = (sub.groupby(["survey", "respondent_id", dim], observed=True)["dev"]
           .agg(["sum", "size"]).reset_index())
    lv_arr = agg[dim].to_numpy()
    S = agg["sum"].to_numpy()
    n = agg["size"].to_numpy()
    if (lv_arr == ref).sum() == 0 or (lv_arr == lv).sum() == 0:
        return np.nan, np.nan, np.nan, len(agg)

    R = len(agg)
    W = rng.multinomial(R, np.full(R, 1 / R), size=N_BOOT)

    def level_mean(weights, level):
        mask = lv_arr == level
        return (weights[:, mask] @ S[mask]) / (weights[:, mask] @ n[mask])

    est = (S[lv_arr == lv].sum() / n[lv_arr == lv].sum()
           - S[lv_arr == ref].sum() / n[lv_arr == ref].sum())
    boots = level_mean(W, lv) - level_mean(W, ref)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi, R


def main() -> None:
    df = cell_demeaned(build_instance_table())
    rng = np.random.default_rng(SEED)
    print(f"\ninstances in qualifying cells: {len(df):,}  "
          f"regions: {df['region'].nunique()}")

    rows = []
    for dim, levels in DIMS.items():
        ref = levels[0]
        for lv in levels[1:]:
            for region, g in df[df[dim].notna()].groupby("region"):
                est, lo, hi, n_resp = contrast(g, dim, ref, lv, rng)
                if n_resp < MIN_REGION_RESPONDENTS or np.isnan(est):
                    continue
                rows.append({"dim": dim, "contrast": f"{lv} - {ref}",
                             "region": region, "n_respondents": n_resp,
                             "estimate": est, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "subgroup_adjusted_by_region.csv", index=False)

    print("\nregional dispersion of each within-cell contrast:")
    for (dim, con), g in out.groupby(["dim", "contrast"]):
        sig = ((g["ci_lo"] > 0) | (g["ci_hi"] < 0)).sum()
        print(f"  {dim:9s} {con:16s} regions {len(g):2d}  "
              f"range [{g['estimate'].min():+.3f}, {g['estimate'].max():+.3f}]  "
              f"sd {g['estimate'].std():.3f}  "
              f"intervals excluding zero: {sig}")

    n_tests = len(out)
    n_sig = int(((out["ci_lo"] > 0) | (out["ci_hi"] < 0)).sum())
    print(f"\nacross all {n_tests} regional contrasts, {n_sig} intervals exclude "
          f"zero ({n_sig / n_tests:.1%}; {0.05 * n_tests:.0f} expected by chance)")
    print("\nlargest absolute regional contrasts:")
    print(out.reindex(out["estimate"].abs().sort_values(ascending=False).index)
          .head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
