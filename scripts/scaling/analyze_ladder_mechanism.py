#!/usr/bin/env python
r"""Two follow-ups to the three-arm ladder.

1. Is the informative arm's advantage only battery siblings?

   The oracle's rank-1 feature is usually an adjacent item from the same
   battery, which raises the obvious worry that the arm measures the model
   restating a nearby answer rather than inferring anything. If so, the
   advantage should vanish on the targets whose best predictor is NOT a near
   neighbour. Closeness is measured the same way the generator's leakage filter
   measures it, cosine between question wordings under all-MiniLM-L6-v2, so the
   split uses the project's own notion of similarity rather than a new one.
   Every feature here is already below the 0.85 exclusion threshold; the
   question is where within that range the effect lives.

2. Does accuracy peak and then fall as weaker features accumulate?

   The informative arm spends its best features first, so if dilution hurts,
   its curve should turn down while the anti arm's keeps rising. Level-to-level
   changes are taken within arm as per-pair differences, which is the only way
   to see a decline of a few points against pair-level variance that is an
   order of magnitude larger. The all-features level, shared by all three arms
   at a median of 196 features, is the endpoint of the same test.

    python .../analyze_ladder_mechanism.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
SCALE = REPO / "outputs" / "scaling_experiment"
SHARDS = SCALE / "ladder_shards"
LAD = REPO.parent / "analysis" / "ladder"
IMP = REPO.parent / "analysis" / "feature_importance"
LEVELS = [0, 1, 2, 4, 8, 16, 32, 64, 96]
ARMS = ("informative", "random", "anti")


def boot_ci(values: np.ndarray, clusters: np.ndarray, n: int = 2000,
            seed: int = 42) -> tuple[float, float]:
    """Question-clustered bootstrap, averaging within question before across."""
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(clusters, return_inverse=True)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    means = np.array([values[inv == i].mean() for i in range(len(keys))])
    draws = means[rng.integers(0, len(keys), size=(n, len(keys)))].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def qmean(g: pd.DataFrame, col: str) -> float:
    return g.groupby(["survey", "target_code"])[col].mean().mean()


def norm_acc(g: pd.DataFrame, col: str = "correct_mean") -> float:
    v = (g[col] - 1 / g["n_options"]) / (1 - 1 / g["n_options"])
    return qmean(g.assign(_v=v), "_v")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--split", type=float, default=None,
                    help="cosine cut for near vs far; default is the median cell")
    args = ap.parse_args()

    d = pd.read_csv(LAD / "ladder_per_instance.csv")

    # The per-instance file has no country, and the oracle ranking is per
    # (target, country) cell, so recover the mapping from the shard inputs.
    pair_country = {}
    for p in sorted(SHARDS.glob("ladder_shard_*.jsonl")):
        for line in open(p, encoding="utf-8", errors="replace"):
            r = json.loads(line)
            pair_country[f"{r['survey']}|{r['id']}|{r['target_code']}"] = r["country"]
    d["country"] = d.pair.map(pair_country)
    print(f"{len(d):,} rows | country resolved for {d.country.notna().mean():.1%}")

    # Restrict to the fixed pair set, as in the main analysis.
    want = {(a, k) for a in ARMS for k in LEVELS if k > 0} | {("all", 0)}
    have = d.groupby("pair").apply(
        lambda g: want <= set(zip(g.arm, g.k)), include_groups=False)
    d = d[d.pair.isin(set(have[have].index))].copy()
    zero = d[(d.arm == "all") & (d.k == 0)]
    lad = pd.concat([d[d.arm.isin(ARMS)]] +
                    [zero.assign(arm=a) for a in ARMS], ignore_index=True)
    lad["q"] = lad.survey + "|" + lad.target_code
    print(f"fixed pair set: {lad.pair.nunique():,} pairs, "
          f"{lad.groupby(['survey','target_code']).ngroups} targets")

    # ---- 1. how close is each cell's best feature to the target? -------------
    from sentence_transformers import SentenceTransformer
    from synthetic_sampling.config import DataPaths
    from synthetic_sampling.loaders.survey_loader import SurveyLoader

    paths = DataPaths.from_yaml(str(REPO / "configs" / "local.yaml"))
    loader = SurveyLoader(paths, verbose=False)
    text = {}
    for s in sorted(lad.survey.unique()):
        _, meta = loader.load_survey(s)
        text[s] = {c: v.get("question", "")
                   for sec in meta.values() if isinstance(sec, dict)
                   for c, v in sec.items() if isinstance(v, dict)}

    # rank is 0-based in feature_importance.csv: rank 0 is the best feature, and
    # cell_diagnostics.top_feature agrees with it. Using rank == 1 would silently
    # profile the SECOND-best feature.
    imp = pd.read_csv(IMP / "feature_importance.csv", dtype={"country": str})
    top1 = imp[imp["rank"] == 0]
    diag = pd.read_csv(IMP / "cell_diagnostics.csv", dtype={"country": str})
    bad = top1.merge(diag, on=["survey", "target_code", "country"])
    bad = bad[bad.feature != bad.top_feature]
    if len(bad):
        raise SystemExit(f"rank 0 disagrees with top_feature on {len(bad)} cells")
    cells = []
    for _, r in top1.iterrows():
        tt, ft = text.get(r.survey, {}).get(r.target_code), text.get(r.survey, {}).get(r.feature)
        if tt and ft:
            cells.append({"survey": r.survey, "target_code": r.target_code,
                          "country": str(r.country), "feature": r.feature,
                          "t_text": tt, "f_text": ft})
    cells = pd.DataFrame(cells)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    uniq = sorted(set(cells.t_text) | set(cells.f_text))
    emb = model.encode(uniq, normalize_embeddings=True, show_progress_bar=False)
    vec = dict(zip(uniq, emb))
    cells["cos"] = [float(vec[a] @ vec[b]) for a, b in zip(cells.t_text, cells.f_text)]

    per_t = (cells.groupby(["survey", "target_code"])["cos"].mean()
             .reset_index().rename(columns={"cos": "top1_cos"}))
    # The oracle's own permutation importance for that feature says whether the
    # cell has non-adjacent signal at all. A "far" target where the oracle also
    # finds nothing is a data limitation; one where the oracle finds plenty and
    # the model still does not use it is a model limitation.
    orac = (diag.groupby(["survey", "target_code"])
            [["top_importance", "majority_baseline"]].mean().reset_index())
    per_t = per_t.merge(orac, on=["survey", "target_code"], how="left")
    cut = args.split if args.split is not None else float(per_t.top1_cos.median())
    per_t["stratum"] = np.where(per_t.top1_cos >= cut, "near", "far")
    print(f"\n=== 1. is the advantage only battery siblings? ===")
    print(f"cosine between each target and its best feature, averaged over its "
          f"country cells\nsplit at the median, {cut:.3f}\n")
    print(f"{'target':<34}{'cos':>7}{'oracle imp':>12}  stratum   best feature")
    for _, r in per_t.sort_values("top1_cos", ascending=False).iterrows():
        f = cells[(cells.survey == r.survey) &
                  (cells.target_code == r.target_code)].feature.mode()
        print(f"{r.survey + ' ' + r.target_code:<34}{r.top1_cos:>7.3f}"
              f"{r.top_importance:>12.4f}  {r.stratum:<8}  "
              f"{f.iloc[0] if len(f) else '?'}")
    for st in ("near", "far"):
        g = per_t[per_t.stratum == st]
        print(f"  {st:<5} n={len(g):>2}  mean cosine {g.top1_cos.mean():.3f}  "
              f"mean oracle importance of the best feature "
              f"{g.top_importance.mean():.4f}")

    lad = lad.merge(per_t[["survey", "target_code", "top1_cos", "stratum"]],
                    on=["survey", "target_code"], how="left")

    print(f"\ninformative minus random, NLL of the true answer, by stratum")
    print(f"{'k':>4}{'near (siblings)':>26}{'far (no near item)':>28}")
    wide = lad.pivot_table(index=["pair", "survey", "target_code", "k", "stratum"],
                           columns="arm", values="nll_true").reset_index()
    wide["q"] = wide.survey + "|" + wide.target_code
    strat_rows = []
    for k in LEVELS:
        if k == 0:
            continue
        cells_out = []
        for st in ("near", "far"):
            g = wide[(wide.k == k) & (wide.stratum == st)].dropna(
                subset=["informative", "random"])
            if g.empty:
                cells_out.append(f"{'-':>26}")
                continue
            dd = (g["informative"] - g["random"]).to_numpy()
            m = pd.Series(dd).groupby(g.q.to_numpy()).mean().mean()
            lo, hi = boot_ci(dd, g.q.to_numpy(), args.boot)
            sig = "*" if (lo > 0 or hi < 0) else " "
            cells_out.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}".rjust(26))
            strat_rows.append({"k": k, "stratum": st, "delta": m, "lo": lo,
                               "hi": hi, "n_targets": g.q.nunique()})
        print(f"{k:>4}" + "".join(cells_out))

    # A median split is coarse and the two strata differ in oracle signal as well
    # as in wording similarity, so relate the advantage to both continuously.
    print(f"\nper-target advantage against how close its best feature is")
    adv = []
    for k in (4, 8, 16):
        g = wide[wide.k == k].dropna(subset=["informative", "random"])
        per_q = (g.assign(d=g["informative"] - g["random"])
                 .groupby(["survey", "target_code"])["d"].mean().reset_index())
        per_q["k"] = k
        adv.append(per_q)
    adv = pd.concat(adv).merge(
        per_t[["survey", "target_code", "top1_cos", "top_importance"]],
        on=["survey", "target_code"], how="left")
    for k in (4, 8, 16):
        s = adv[adv.k == k]
        rc = s[["d", "top1_cos"]].corr(method="spearman").iloc[0, 1]
        ri = s[["d", "top_importance"]].corr(method="spearman").iloc[0, 1]
        n_neg = int((s.d < 0).sum())
        print(f"  k={k:>2}  Spearman with cosine {rc:+.3f}   "
              f"with oracle importance {ri:+.3f}   "
              f"advantage negative for {n_neg}/{len(s)} targets")
    print("  a correlation near zero means wording similarity does not explain "
          "which\n  targets benefit from ordering; a negative one with importance "
          "means the\n  advantage is larger where the oracle found more signal")
    adv.to_csv(LAD / "ladder_target_advantage.csv", index=False)

    print(f"\nnormalized accuracy by arm and stratum")
    print(f"{'k':>4}" + "".join(f"{a[:4] + '/' + s:>16}"
                                for s in ("near", "far") for a in ARMS))
    for k in LEVELS:
        row = []
        for st in ("near", "far"):
            for a in ARMS:
                g = lad[(lad.arm == a) & (lad.k == k) & (lad.stratum == st)]
                row.append(f"{norm_acc(g):>16.4f}" if not g.empty else f"{'-':>16}")
        print(f"{k:>4}" + "".join(row))

    # ---- 2. does the curve peak and turn down? -------------------------------
    print(f"\n=== 2. does accuracy peak, then fall as weaker features arrive? ===")
    print("per-pair change from one level to the next, within arm; "
          "positive means the extra features helped")
    steps = list(zip(LEVELS, LEVELS[1:]))
    peak_rows = []
    for a in ARMS:
        print(f"\n  {a}")
        print(f"    {'step':>12}{'norm acc':>12}{'NLL true':>26}")
        for lo_k, hi_k in steps:
            g = lad[(lad.arm == a) & (lad.k.isin([lo_k, hi_k]))]
            w = g.pivot_table(index=["pair", "q", "n_options"], columns="k",
                              values=["correct_mean", "nll_true"]).dropna()
            if w.empty:
                continue
            M = w.index.get_level_values("n_options").to_numpy()
            dacc = ((w[("correct_mean", hi_k)] - w[("correct_mean", lo_k)])
                    / (1 - 1 / M)).to_numpy()
            dnll = (w[("nll_true", hi_k)] - w[("nll_true", lo_k)]).to_numpy()
            qq = w.index.get_level_values("q").to_numpy()
            ma = pd.Series(dacc).groupby(qq).mean().mean()
            mn = pd.Series(dnll).groupby(qq).mean().mean()
            lo, hi = boot_ci(dnll, qq, args.boot)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"    {f'{lo_k}->{hi_k}':>12}{ma:>+12.4f}"
                  f"{f'{mn:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}':>26}")
            peak_rows.append({"arm": a, "from": lo_k, "to": hi_k, "d_norm_acc": ma,
                              "d_nll": mn, "lo": lo, "hi": hi})

        # k=96 to the all-features level, which is shared across arms.
        end = d[(d.arm == "all") & (d.k > 0)][["pair", "correct_mean", "nll_true",
                                               "n_options", "survey", "target_code"]]
        end = end.assign(q=end.survey + "|" + end.target_code)
        g96 = lad[(lad.arm == a) & (lad.k == 96)][["pair", "correct_mean", "nll_true"]]
        m = end.merge(g96, on="pair", suffixes=("_all", "_96"))
        if m.empty:
            continue
        dacc = ((m.correct_mean_all - m.correct_mean_96) / (1 - 1 / m.n_options)).to_numpy()
        dnll = (m.nll_true_all - m.nll_true_96).to_numpy()
        ma = pd.Series(dacc).groupby(m.q.to_numpy()).mean().mean()
        mn = pd.Series(dnll).groupby(m.q.to_numpy()).mean().mean()
        lo, hi = boot_ci(dnll, m.q.to_numpy(), args.boot)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"    {'96->all':>12}{ma:>+12.4f}"
              f"{f'{mn:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}':>26}")
        peak_rows.append({"arm": a, "from": 96, "to": -1, "d_norm_acc": ma,
                          "d_nll": mn, "lo": lo, "hi": hi})

    LAD.mkdir(parents=True, exist_ok=True)
    per_t.to_csv(LAD / "ladder_top1_similarity.csv", index=False)
    pd.DataFrame(strat_rows).to_csv(LAD / "ladder_by_stratum.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(LAD / "ladder_level_steps.csv", index=False)
    print(f"\nwrote ladder_top1_similarity.csv, ladder_by_stratum.csv and "
          f"ladder_level_steps.csv to {LAD}")


if __name__ == "__main__":
    main()
