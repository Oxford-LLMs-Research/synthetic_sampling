"""Analyze the controlled country-injection experiment (paired within-instance).

Each rich-profile instance whose profile and target never name the respondent's
country is scored under three prompts differing by one appended item:

    baseline              no country stated
    with_country          "In which country do you live?" -> true country
    with_country_placebo  the same item, a country from a different region

Contrasts, and what each is for:

    with_country - baseline           does adding the country help at all?
                                      Confounded: adding any item lengthens the
                                      prompt and signals geography is relevant.
    placebo - baseline                measures that confound directly.
    with_country - placebo            isolates the only remaining difference,
                                      whether the country is the right one.

The placebo is not an inert control. Naming a country from another region
injects wrong information rather than none, so the third contrast is an active
comparison. That makes it *more* sensitive than the first if the model uses
country information correctly, but it also means the contrast is only
informative if the manipulation reaches the model's output at all. We therefore
report the diagnostic alongside it: how often naming a different country changes
the prediction, and, among the instances where it does, whether the true country
is right more often than the wrong one (McNemar on the discordant pairs).

Normalized accuracy uses M = distinct answer labels, matching
recompute_norm_acc_distinct_m.py.

Outputs to analysis/country_injection/:
  summary.csv     overall accuracies, contrasts and CIs
  by_region.csv   per-region contrasts against the observational effects
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
COLAB = ROOT / "synthetic_sampling" / "outputs" / "colab_experiments"
JSONL_DIR = (ROOT / "synthetic_sampling" / "outputs"
             / "main_data_smaller_20_jan_26" / "main_data")
OUT = ROOT / "analysis" / "country_injection"

CONDITIONS = ["baseline", "with_country", "with_country_placebo"]
SEED = 20260728
N_BOOT = 10000

# 13-model mean observational effect (explicit minus implicit country item),
# from the country-conditioning analysis. Used only for the comparison table.
OBSERVATIONAL = {
    "Central Asia": +0.059, "Southern Africa": +0.018, "South America": +0.014,
    "Central America": +0.010, "Middle East": +0.008, "East Asia": +0.007,
    "Caribbean": +0.006, "North America": +0.003, "Northern Europe": +0.003,
    "Central Africa": -0.000, "North Africa": -0.001, "Southeast Asia": -0.002,
    "West Africa": -0.002, "East Africa": -0.006, "South Asia": -0.020,
    "Oceania": -0.021, "Southern Europe": -0.029, "Western Europe": -0.040,
    "Eastern Europe": -0.049,
}


def distinct_m() -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for path in sorted(JSONL_DIR.glob("*_instances.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                key = (d["survey"], d["target_code"])
                if key not in out:
                    out[key] = len(set(d["options"]))
    return out


def paired_ci(d: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return tuple(np.percentile(boots, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=str(COLAB / "country_injection_instances.jsonl"))
    ap.add_argument("--results", default=str(COLAB / "country_results.jsonl"))
    args = ap.parse_args()

    inst = {}
    with open(args.instances, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            inst[d["example_id"]] = d

    by: dict[str, dict] = defaultdict(dict)
    with open(args.results, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by[r["example_id"]][r["condition"]] = r

    m_map = distinct_m()
    rows = []
    for eid, conds in by.items():
        if not all(c in conds for c in CONDITIONS) or eid not in inst:
            continue
        d = inst[eid]
        m = m_map.get((d["survey"], d["target_code"]))
        rec = {"example_id": eid, "survey": d["survey"], "region": d["region"],
               "country": d["country_iso2"], "n_options": m}
        for c in CONDITIONS:
            rec[f"correct_{c}"] = bool(conds[c]["correct"])
            rec[f"pred_{c}"] = conds[c]["predicted"]
            rec[f"norm_{c}"] = (np.nan if not m or m < 2 else
                                (float(conds[c]["correct"]) - 1 / m) / (1 - 1 / m))
        rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"complete triples: {len(df):,}  regions: {df['region'].nunique()}  "
          f"countries: {df['country'].nunique()}")

    # ---- overall accuracy and contrasts -------------------------------
    print("\n=== accuracy ===")
    summary = []
    for scale, pre in (("raw", "correct_"), ("normalized", "norm_")):
        vals = {c: df[f"{pre}{c}"].astype(float) for c in CONDITIONS}
        base, true, plac = (vals["baseline"], vals["with_country"],
                            vals["with_country_placebo"])
        print(f"  {scale:<11} baseline {base.mean():.4f} | true {true.mean():.4f} "
              f"| placebo {plac.mean():.4f}")
        for label, d in (("true - baseline", true - base),
                         ("placebo - baseline", plac - base),
                         ("contentful (true - placebo)", true - plac)):
            arr = d.dropna().to_numpy()
            lo, hi = paired_ci(arr)
            print(f"      {label:<28} {arr.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
            summary.append({"scale": scale, "contrast": label,
                            "estimate": arr.mean(), "ci_lo": lo, "ci_hi": hi})

    # ---- does the manipulation reach the output? ----------------------
    print("\n=== does the placebo bite? ===")
    flip_t = (df["pred_with_country"] != df["pred_baseline"]).mean()
    flip_p = (df["pred_with_country_placebo"] != df["pred_baseline"]).mean()
    diverge = df["pred_with_country"] != df["pred_with_country_placebo"]
    print(f"  true vs baseline flips     {flip_t:.4f}")
    print(f"  placebo vs baseline flips  {flip_p:.4f}")
    print(f"  true vs PLACEBO divergence {diverge.mean():.4f}  "
          f"({int(diverge.sum())} instances)")
    moved = ((df["pred_with_country"] != df["pred_baseline"])
             | (df["pred_with_country_placebo"] != df["pred_baseline"]))
    print(f"  either cue moved the prediction: {int(moved.sum())} "
          f"({moved.mean():.1%}); of those, true and placebo differ on "
          f"{diverge[moved].mean():.1%}")

    # McNemar on the discordant pairs: conditional on the country identity
    # changing the answer, is the true country right more often?
    dis = df[diverge]
    b = int((dis["correct_with_country"] & ~dis["correct_with_country_placebo"]).sum())
    c = int((~dis["correct_with_country"] & dis["correct_with_country_placebo"]).sum())
    d_arr = (dis["correct_with_country"].astype(float)
             - dis["correct_with_country_placebo"].astype(float)).to_numpy()
    lo, hi = paired_ci(d_arr)
    try:
        from scipy.stats import binomtest
        p = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
    except ImportError:
        p = float("nan")
    print(f"  discordant pairs: true right {b}, placebo right {c}; "
          f"difference {d_arr.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], McNemar p = {p:.4f}")
    summary.append({"scale": "diagnostic", "contrast": "true vs placebo divergence",
                    "estimate": diverge.mean(), "ci_lo": np.nan, "ci_hi": np.nan})
    summary.append({"scale": "diagnostic", "contrast": "discordant McNemar p",
                    "estimate": p, "ci_lo": np.nan, "ci_hi": np.nan})

    # ---- by region, against the observational effects -----------------
    print("\n=== by region (normalized contentful effect) ===")
    reg_rows = []
    for region, g in df.groupby("region"):
        d = (g["norm_with_country"] - g["norm_with_country_placebo"]).dropna().to_numpy()
        lo, hi = paired_ci(d)
        obs = OBSERVATIONAL.get(region, np.nan)
        reg_rows.append({"region": region, "n": len(g), "experimental": d.mean(),
                         "ci_lo": lo, "ci_hi": hi, "observational": obs,
                         "ci_excludes_observational":
                             bool(not np.isnan(obs) and (obs < lo or obs > hi))})
    reg = pd.DataFrame(reg_rows).sort_values("observational")
    print(reg.round(4).to_string(index=False))
    ok = reg["observational"].notna()
    r = np.corrcoef(reg.loc[ok, "experimental"], reg.loc[ok, "observational"])[0, 1]
    print(f"\ncorrelation experimental vs observational across "
          f"{int(ok.sum())} regions: r = {r:.3f}")
    print(f"regions whose interval excludes the observational value: "
          f"{int(reg['ci_excludes_observational'].sum())}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    reg.to_csv(OUT / "by_region.csv", index=False)
    print(f"\nwrote {OUT / 'summary.csv'} and {OUT / 'by_region.csv'}")


if __name__ == "__main__":
    main()
