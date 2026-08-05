"""Refit the V2 geographic model without the modal-share control.

The main text claims that controlling for how skewed a question's answers are
shrinks the unexplained regional spread. The controlled model is stored in
analysis/mixed_effects/mixed_effects_model_v2_full_sample.json (regional fixed
effects span 3.2 percentage points), but the uncontrolled comparison it is
measured against was never saved, so the "before" number had no source. This
refits the identical specification with modal_share dropped and prints the
regional spread, so the shrinkage can be stated from an artifact on disk.

Writes analysis/mixed_effects/mem_v2_no_modalshare.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
MEM = ROOT / "synthetic_sampling" / "analysis" / "mixed_effects"
DATA = MEM / "mixed_effects_data.csv"
OUT = MEM / "mem_v2_no_modalshare.json"

# Identical to the stored V2 model except that modal_share is dropped.
FORMULA = "correct ~ n_options + model + region + topic_section"
GROUPS = "survey"


def region_span(params: pd.Series) -> tuple[float, dict[str, float]]:
    coefs = {k.split("T.", 1)[1].rstrip("]"): float(v)
             for k, v in params.items() if k.startswith("region[")}
    # The reference region is the omitted level, at 0 by construction.
    coefs["(reference)"] = 0.0
    return max(coefs.values()) - min(coefs.values()), coefs


def main() -> None:
    df = pd.read_csv(DATA, encoding="latin-1",
                     usecols=["correct", "n_options", "modal_share", "model",
                              "region", "topic_section", "survey"])
    df = df[df["region"] != "Unknown"]
    for c in ("correct", "n_options", "modal_share"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    print(f"rows: {len(df):,}  regions: {df['region'].nunique()}")

    fit = smf.mixedlm(FORMULA, df, groups=df[GROUPS]).fit()
    span, coefs = region_span(fit.params)
    print(f"\nwithout modal_share: regional spread {span * 100:.1f} pp")
    for k, v in sorted(coefs.items(), key=lambda kv: kv[1]):
        print(f"  {k:<18} {v:+.4f}")

    stored = json.load(open(MEM / "mixed_effects_model_v2_full_sample.json"))
    ctrl = {k.split("region", 1)[1]: v["coef"] if isinstance(v, dict) else v
            for k, v in stored["fixed_effects"].items() if k.startswith("region")}
    ctrl_span = max(ctrl.values()) - min(ctrl.values())
    print(f"\nwith modal_share (stored): regional spread {ctrl_span * 100:.1f} pp")
    print(f"shrinkage: {100 * (1 - ctrl_span / span):.0f}%")

    json.dump({"formula": FORMULA + f" + (1|{GROUPS})",
               "n_obs": int(len(df)),
               "region_coefs": coefs,
               "region_span_pp": span * 100,
               "region_span_pp_with_modal_share": ctrl_span * 100,
               "shrinkage_pct": 100 * (1 - ctrl_span / span)},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
