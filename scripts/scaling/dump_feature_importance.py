#!/usr/bin/env python
r"""Rank features by how well they predict each target, on the full survey data.

The by-feature ladder needs, per target, an ordering of features from most to
least predictive, so a profile can be grown by informativeness rather than at
random.

Basis. Importance is a property of the SURVEY, not of the profiles we happened
to draw, so this fits on the raw microdata through the project's own
SurveyLoader: every respondent, every legitimate feature. An earlier version of
this script estimated on the k=96 profile instances instead, which was wrong on
two counts. It threw away most respondents (Asian Barometer tops out at 131 per
target against thousands in the raw file) and most of each respondent's answers
(96 of a pool running to 600), leaving permutation importance to be read off
roughly 90 held-out cases over hundreds of features.

Selection leakage. Ranking features on the same respondents the language model
is later evaluated on gives the informative arm a head start that will not
replicate, because the top of the ranking is top partly through noise in those
people. Respondents are split three ways and the third split is written out
untouched, for the language model alone.

Construct leakage. A feature can give the answer away without repeating the
target's code. Two screens, both FLAGGED rather than dropped, because whether a
strong correlate counts as leakage is a judgement to make with the numbers
visible:

  conditional  the feature's missingness rate differs sharply across target
               classes, the signature of a follow-up asked only of respondents
               who answered one way. This screen only works on raw microdata,
               where missingness means "not asked" rather than "not drawn".
  lexical      the feature's wording overlaps the target's

    python .../dump_feature_importance.py --survey wvs
    python .../dump_feature_importance.py --all --n-repeats 5
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from xgboost_baseline import SURVEY_CONFIG, make_clf                   # noqa: E402
from synthetic_sampling.config import DataPaths                        # noqa: E402
from synthetic_sampling.config.surveys import get_survey_config        # noqa: E402
from synthetic_sampling.loaders.survey_loader import SurveyLoader      # noqa: E402

OUT = REPO.parent / "analysis" / "feature_importance"
CONFIG = REPO / "configs" / "local.yaml"

MIN_RESPONDENTS = 300
MIN_CLASSES = 2
MIN_COVERAGE = 0.30          # a feature answered by fewer than this is unusable
SPLIT = (0.50, 0.25, 0.25)   # fit, score, reserved for the language model
MISSING_RANGE = 0.7          # conditional-leakage threshold, as in features_project
STOP = set("the a an of to in for do you your and or is are was were be been "
           "how what which who whom that this these those on at by with as it "
           "not no yes about would will can could should if then than there "
           "very much more most less least please tell me say think".split())


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower())
            if w not in STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def targets_for(survey: str) -> list[str]:
    """The AAAI target questions for this survey."""
    q = pd.read_csv(REPO.parent / "analysis" / "normalized_accuracy" /
                    "per_question_norm_acc.csv")
    return sorted(q[q["survey"] == survey]["target_code"].unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default=None, choices=list(SURVEY_CONFIG))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=400)
    args = ap.parse_args()
    surveys = list(SURVEY_CONFIG) if args.all else [args.survey]
    if not surveys or surveys == [None]:
        raise SystemExit("pass --survey <name> or --all")

    paths = DataPaths.from_yaml(str(CONFIG))
    loader = SurveyLoader(paths, verbose=False)

    rows, reserved = [], []
    for survey in surveys:
        cfg = get_survey_config(survey)
        df, meta = loader.load_survey(survey)
        id_col = cfg.respondent_id_col
        # The feature universe is the metadata's variables, which is the same
        # pool the profile generator draws from.
        code2text = {c: v.get("question", "")
                     for sec in meta.values() if isinstance(sec, dict)
                     for c, v in sec.items() if isinstance(v, dict)}
        pool_all = [c for c in code2text if c in df.columns]
        print(f"\n{survey}: {len(df):,} respondents, "
              f"{len(pool_all):,} metadata variables present in the data")

        tgts = [t for t in targets_for(survey) if t in df.columns]
        print(f"  {len(tgts)} AAAI targets found in the file")
        rng = np.random.default_rng(42)

        for target in tgts:
            y_raw = df[target]
            ok = y_raw.notna()
            if ok.sum() < MIN_RESPONDENTS:
                continue
            sub = df.loc[ok]
            y_lab = sub[target].astype(str)
            classes = sorted(y_lab.unique())
            if len(classes) < MIN_CLASSES:
                continue

            pool = [c for c in pool_all if c != target]
            X = sub[pool].apply(pd.to_numeric, errors="coerce")
            # Survey codes use negatives for refused/missing; treat as missing.
            X = X.mask(X < 0)
            cover = X.notna().mean()
            X = X.loc[:, cover >= MIN_COVERAGE]
            if X.shape[1] > args.max_features:
                X = X[cover[X.columns].sort_values(ascending=False)
                      .head(args.max_features).index]
            if X.shape[1] < 10:
                continue

            # Conditional leakage: missingness that tracks the target's classes.
            big = [c for c in classes if (y_lab == c).sum() >= 20]
            cond = set()
            if len(big) >= 2:
                rates = pd.DataFrame(
                    {c: X.loc[(y_lab == c).values].isna().mean() for c in big})
                rng_span = rates.max(axis=1) - rates.min(axis=1)
                cond = set(rng_span[rng_span >= MISSING_RANGE].index)

            y = pd.Categorical(y_lab, categories=classes).codes
            ids = sub[id_col].astype(str).to_numpy()
            uniq = np.array(sorted(set(ids)))
            perm = rng.permutation(len(uniq))
            n_fit = int(SPLIT[0] * len(uniq))
            n_sc = int(SPLIT[1] * len(uniq))
            grp = {}
            for j, p in enumerate(perm):
                grp[uniq[p]] = ("fit" if j < n_fit
                                else "score" if j < n_fit + n_sc else "reserved")
            where = np.array([grp[i] for i in ids])
            if (where == "fit").sum() < 100 or (where == "score").sum() < 60:
                continue
            if len(np.unique(y[where == "fit"])) < MIN_CLASSES:
                continue

            clf = make_clf()
            clf.fit(X[where == "fit"], y[where == "fit"])
            acc = float((clf.predict(X[where == "score"])
                         == y[where == "score"]).mean())
            pi = permutation_importance(
                clf, X[where == "score"], y[where == "score"],
                n_repeats=args.n_repeats, random_state=42, scoring="accuracy")

            tgt_tok = tokens(code2text.get(target, ""))
            order = np.argsort(-pi.importances_mean)
            for rank, j in enumerate(order):
                code = X.columns[j]
                rows.append({
                    "survey": survey, "target_code": target, "feature": code,
                    "rank": rank,
                    "importance_mean": float(pi.importances_mean[j]),
                    "importance_std": float(pi.importances_std[j]),
                    "coverage": round(float(cover[code]), 3),
                    "lexical_overlap": round(
                        jaccard(tokens(code2text.get(code, "")), tgt_tok), 3),
                    "conditional_flag": code in cond,
                    "model_acc": round(acc, 4),
                    "n_fit": int((where == "fit").sum()),
                    "n_score": int((where == "score").sum()),
                })
            for rid in sorted({i for i, w in zip(ids, where) if w == "reserved"}):
                reserved.append({"survey": survey, "target_code": target,
                                 "respondent_id": rid})
            top = X.columns[order[0]]
            print(f"  {target:<12}{X.shape[1]:>4} feats  fit {int((where=='fit').sum()):>5}"
                  f"  score {int((where=='score').sum()):>5}  acc {acc:.3f}"
                  f"  top {top} ({pi.importances_mean[order[0]]:+.4f})"
                  f"{'  [cond-flagged: ' + str(len(cond)) + ']' if cond else ''}")

    if not rows:
        raise SystemExit("nothing produced")
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "feature_importance.csv", index=False)
    pd.DataFrame(reserved).to_csv(OUT / "reserved_respondents.csv", index=False)

    top = d[d["rank"] == 0]
    print(f"\n{d.groupby(['survey','target_code']).ngroups} targets, "
          f"{len(d):,} feature rows")
    print(f"top feature's mean accuracy drop when permuted: "
          f"{top['importance_mean'].mean():+.4f} "
          f"(median {top['importance_mean'].median():+.4f})")
    print(f"targets whose top feature is flagged: "
          f"conditional {int(top['conditional_flag'].sum())}, "
          f"lexical overlap >= 0.4 {(top['lexical_overlap'] >= 0.4).sum()}")
    print(f"\nwrote {OUT / 'feature_importance.csv'}")
    print(f"wrote {OUT / 'reserved_respondents.csv'} ({len(reserved):,} rows)")


if __name__ == "__main__":
    main()
