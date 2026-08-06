#!/usr/bin/env python
r"""Was the paper's echo readout recoverable all along?

Echo scoring ranks options by the mean log-probability of their own tokens,
which adds a per-option fluency constant to whatever the model believes about
the person. The domain-conditional PMI correction (Holtzman et al. 2021)
subtracts each option's log-probability under a neutral premise, cancelling
that constant. The PMI run scores two neutral premises next to a fresh
echo_plain control in one serving:

    echo_ctxfree   a bare "Answer:" premise: context-free fluency
    echo_qonly     the paper's template with the profile empty: fluency plus
                   fit to the question

which decomposes the paper's score, per option:

    echo_plain  =  fluency  +  question-fit  +  profile-driven
                  [ctxfree]   [qonly-ctxfree]   [plain-qonly]

Two derived rules follow: pmi_free = plain - ctxfree (the standard PMI rule)
and pmi_q = plain - qonly (the profile-driven component alone). The decisive
question: does either recover the label readout's accuracy? If yes, the
completion-probability assumption was sound and the flaw was normalization; if
no, showing the options genuinely adds information and the hidden-options
design cannot be repaired.

    python .../analyze_pmi.py --results <readout_pmi_<tag>.jsonl>
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_readout import (                                     # noqa: E402
    boot_ci, prior_corrected, qmean, weighted_auc)

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
RESULTS_DIR = SCALE / "readout_results"
OUT = REPO.parent / "analysis" / "readout"

BASE_ARMS = ("echo_plain", "echo_qonly", "echo_ctxfree")
DERIVED = ("pmi_free", "pmi_q")


def load_scores(path: Path, meta: dict, setname: str = "original") -> dict:
    """eid -> {arm: np.ndarray of scores aligned to the instance's options}."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = meta.get(r["example_id"])
        if m is None:
            continue
        opts = m["option_sets"]["original"]
        rec = {}
        for arm in BASE_ARMS:
            d = r["results"].get(f"{setname}|{arm}")
            if not d or "error" in d or not d.get("scores"):
                continue
            rec[arm] = np.array([d["scores"].get(o, float("-inf"))
                                 for o in opts], float)
        if len(rec) == len(BASE_ARMS):
            out[r["example_id"]] = rec
    return out


def load_predictions(path: Path, meta: dict, arm: str) -> dict:
    """eid -> predicted option string for one arm of a results file."""
    preds = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["example_id"] not in meta:
            continue
        d = r["results"].get(f"original|{arm}")
        if d and "error" not in d and d.get("predicted") is not None:
            preds[r["example_id"]] = d["predicted"]
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="readout_pmi_<tag>.jsonl (or any results file "
                         "containing the three echo arms)")
    ap.add_argument("--input", type=Path, default=SCALE / "readout_set.jsonl")
    ap.add_argument("--label-results", type=Path, default=None,
                    help="results file holding label_num for the same model; "
                         "defaults to readout_results_<tag>.jsonl if present. "
                         "If it is a DIFFERENT serving, agreement with it is "
                         "read against the 64.4% cross-stack ceiling")
    args = ap.parse_args()

    tag = args.results.stem
    for prefix in ("readout_pmi_", "readout_results_", "readout_grid_"):
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
            break

    meta = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r

    scores = load_scores(args.results, meta)
    print(f"{len(scores):,} instances with all three echo arms, "
          f"model file {args.results.name}\n")
    if not scores:
        sys.exit("nothing to analyse")

    # ---- assemble per-instance rows ---------------------------------------
    recs = []
    for eid, rec in scores.items():
        m = meta[eid]
        opts = m["option_sets"]["original"]
        vec = {"echo_plain": rec["echo_plain"],
               "echo_qonly": rec["echo_qonly"],
               "echo_ctxfree": rec["echo_ctxfree"],
               "pmi_free": rec["echo_plain"] - rec["echo_ctxfree"],
               "pmi_q": rec["echo_plain"] - rec["echo_qonly"]}
        row = {"eid": eid, "q": f"{m['survey']}|{m['target_code']}",
               "truth": m["ground_truth"], "opts": tuple(opts),
               "finite": all(np.isfinite(v).all() for v in vec.values())}
        for rule, v in vec.items():
            row[f"pred_{rule}"] = opts[int(np.argmax(v))]
            row[f"scores_{rule}"] = v
        recs.append(row)
    df = pd.DataFrame(recs)
    n_nonfinite = int((~df.finite).sum())
    if n_nonfinite:
        print(f"note: {n_nonfinite} instances carry a non-finite score in some "
              f"arm; they stay in accuracy, drop from AUC and prior correction\n")

    rules = list(BASE_ARMS) + list(DERIVED)
    M_by_q = df.groupby("q").truth.nunique()
    df["M"] = df.q.map(M_by_q)
    acc_df = df[df.M >= 2]

    # Modal option set per question, as in analyze_readout: AUC and prior
    # correction need option-aligned vectors across respondents.
    modal = {}
    for q, gq in df.groupby("q"):
        sets = collections.Counter(gq.opts)
        modal[q] = sets.most_common(1)[0][0]

    # ---- 1. headline metrics per rule -------------------------------------
    print("=== 1. per scoring rule, on identical instances in one serving ===")
    print(f"{'rule':<14}{'norm acc':>10}{'AUC':>8}{'prior only':>12}{'+model':>9}")
    summary = []
    for rule in rules:
        g = acc_df
        correct = (g[f"pred_{rule}"] == g.truth).to_numpy().astype(float)
        na = qmean(((correct - 1 / g.M) / (1 - 1 / g.M)).to_numpy(),
                   g.q.to_numpy())
        aucs, pc_prior, pc_corr = [], [], []
        for q, gq in df.groupby("q"):
            keep = gq[(gq.opts == modal[q]) & gq.finite]
            truth = keep.truth.to_numpy()
            if len(keep) < 20 or len(set(truth)) < 2:
                continue
            opts = list(modal[q])
            arr = np.array(keep[f"scores_{rule}"].tolist(), float)
            cent = arr - arr.mean(axis=1, keepdims=True)
            a = weighted_auc(cent, truth, opts)
            if not np.isnan(a):
                aucs.append(a)
            p0, p1 = prior_corrected(arr, truth, opts)
            pc_prior.append(p0)
            pc_corr.append(p1)
        s = {"rule": rule, "norm_acc": na,
             "auc": float(np.mean(aucs)) if aucs else np.nan,
             "n_q_auc": len(aucs),
             "prior_only": float(np.mean(pc_prior)) if pc_prior else np.nan,
             "corrected": float(np.mean(pc_corr)) if pc_corr else np.nan}
        summary.append(s)
        print(f"{rule:<14}{na:>10.4f}{s['auc']:>8.3f}"
              f"{s['prior_only']:>12.4f}{s['corrected']:>9.4f}")
    print("\nechofree/qonly rows are the NEUTRAL premises scored as if they "
          "were predictors;\ntheir accuracy is the fluency default's, a "
          "baseline rather than a reading of anyone")

    # ---- 2. paired contrasts against echo_plain ----------------------------
    print("\n=== 2. paired against echo_plain, question-clustered ===")
    base_corr = (acc_df["pred_echo_plain"] == acc_df.truth).to_numpy().astype(float)
    qq = acc_df.q.to_numpy()
    contrasts = []
    for rule in DERIVED:
        corr = (acc_df[f"pred_{rule}"] == acc_df.truth).to_numpy().astype(float)
        d = corr - base_corr
        m, (lo, hi) = qmean(d, qq), boot_ci(d, qq)
        norm_d = (d / (1 - 1 / acc_df.M)).to_numpy()
        mn, (nlo, nhi) = qmean(norm_d, qq), boot_ci(norm_d, qq)
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {rule:<12} raw {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {star}"
              f"   normalized {mn:+.4f} [{nlo:+.4f}, {nhi:+.4f}]")
        contrasts.append({"rule": rule, "d_acc": m, "lo": lo, "hi": hi,
                          "d_norm_acc": mn, "norm_lo": nlo, "norm_hi": nhi,
                          "n": len(d)})

    # ---- 3. argmax decomposition ------------------------------------------
    print("\n=== 3. how much of the paper's argmax the neutral premises "
          "already determine ===")
    rows = []
    for q, gq in df.groupby("q"):
        rows.append({
            "q": q, "n": len(gq),
            "ctxfree_eq_plain": (gq.pred_echo_ctxfree == gq.pred_echo_plain).mean(),
            "qonly_eq_plain": (gq.pred_echo_qonly == gq.pred_echo_plain).mean(),
            "pmi_free_eq_plain": (gq.pred_pmi_free == gq.pred_echo_plain).mean(),
            "plain_pred_entropy": len(set(gq.pred_echo_plain)),
        })
    dec = pd.DataFrame(rows)
    print(f"  argmax(echo_ctxfree) == argmax(echo_plain) on "
          f"{dec.ctxfree_eq_plain.mean():.1%} of instances (question mean)")
    print(f"  argmax(echo_qonly)   == argmax(echo_plain) on "
          f"{dec.qonly_eq_plain.mean():.1%}")
    print(f"  the PMI correction changes the paper's prediction on "
          f"{1 - dec.pmi_free_eq_plain.mean():.1%} of instances")
    print("  the first line is the share of the paper's predictions already "
          "determined\n  before the model has seen anything about the person "
          "or the question")

    # ---- 4. score-variance decomposition ----------------------------------
    print("\n=== 4. variance decomposition of the paper's scores, per "
          "question ===")
    var_rows = []
    for q, gq in df.groupby("q"):
        keep = gq[(gq.opts == modal[q]) & gq.finite]
        if len(keep) < 20 or len(modal[q]) < 2:
            continue
        arr = np.array(keep["scores_echo_plain"].tolist(), float)
        cent = arr - arr.mean(axis=1, keepdims=True)      # respondent-centred
        opt_main = cent.mean(axis=0)                       # per-option constant
        resid = cent - opt_main
        tot = float(np.var(cent))
        share_const = float(np.var(np.broadcast_to(opt_main, cent.shape))) / tot \
            if tot > 0 else np.nan
        ctx = np.array(keep["scores_echo_ctxfree"].iloc[0], float)
        ctx_cent = ctx - ctx.mean()
        denom = float(np.dot(ctx_cent, ctx_cent))
        if denom > 0:
            beta = float(np.dot(opt_main, ctx_cent)) / denom
            r2 = 1 - float(np.var(opt_main - beta * ctx_cent)) / float(np.var(opt_main)) \
                if np.var(opt_main) > 0 else np.nan
        else:
            r2 = np.nan
        var_rows.append({"q": q, "n": len(keep), "M": len(modal[q]),
                         "share_option_constant": share_const,
                         "r2_ctxfree_on_constant": r2})
    var = pd.DataFrame(var_rows)
    print(f"  option-constant share of score variance: "
          f"{var.share_option_constant.mean():.1%} (mean over "
          f"{len(var)} questions)")
    # A regression across M options has M points: at M=2 the fit is exact by
    # construction and at M=3 nearly so, which would inflate the mean. The
    # printed figure restricts to M >= 4; the CSV keeps every question with M.
    var4 = var[var.M >= 4]
    print(f"  of that constant, context-free fluency explains R^2 = "
          f"{var4.r2_ctxfree_on_constant.mean():.2f} "
          f"(questions with M >= 4 options, n={len(var4)})")
    print("  a large first number with a large second is the fluency story in "
          "one line:\n  most of what separates options is constant across "
          "respondents, and most of\n  the constant is how the option reads "
          "with no context at all")

    # ---- 5. against the label readout -------------------------------------
    label_path = args.label_results
    if label_path is None:
        cand = RESULTS_DIR / f"readout_results_{tag}.jsonl"
        label_path = cand if cand.exists() else None
    if label_path and label_path.exists():
        same_serving = label_path.resolve() == args.results.resolve()
        label = load_predictions(label_path, meta, "label_num")
        common = [e for e in df.eid if e in label]
        if common:
            sub = df.set_index("eid").loc[common]
            lab = pd.Series({e: label[e] for e in common})
            print(f"\n=== 5. against label_num from {label_path.name} ===")
            if not same_serving:
                print("  DIFFERENT SERVING: agreement is read against the "
                      "64.4% cross-stack ceiling\n  (checks/control_check.py), "
                      "not against 1; accuracy levels shift ~0.002")
            for rule in ("echo_plain", "pmi_free", "pmi_q"):
                agree = (sub[f"pred_{rule}"] == lab).mean()
                print(f"  argmax({rule}) == argmax(label_num) on {agree:.1%}")
            subM = sub[sub.M >= 2]
            labM = lab.loc[subM.index]
            na_lab = qmean((((labM == subM.truth).astype(float) - 1 / subM.M)
                            / (1 - 1 / subM.M)).to_numpy(), subM.q.to_numpy())
            print(f"  label_num normalized accuracy on these instances: "
                  f"{na_lab:.4f}")
    else:
        print("\n(no label_num results found for this tag; section 5 skipped)")

    # ---- 6. replicate agreement, persisted --------------------------------
    print("\n=== 6. replicate: the same options scored twice ===")
    rep_scores = load_scores(args.results, meta, setname="original_replicate")
    rep_rows = []
    if rep_scores:
        for arm in BASE_ARMS:
            pairs = [(scores[e][arm], rep_scores[e][arm])
                     for e in rep_scores if e in scores]
            agree = np.mean([np.argmax(a) == np.argmax(b) for a, b in pairs])
            drift = np.mean([np.abs(a - b).mean() for a, b in pairs])
            print(f"  {arm:<14} argmax agreement {agree:>6.1%}   "
                  f"mean |score drift| {drift:.4f} nats   n={len(pairs):,}")
            rep_rows.append({"arm": arm, "agreement": agree,
                             "score_drift_nats": drift, "n": len(pairs)})
    else:
        print("  no replicate records found")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / f"pmi_summary_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / f"pmi_contrasts_{tag}.csv", index=False)
    dec.to_csv(OUT / f"pmi_argmax_decomposition_{tag}.csv", index=False)
    var.to_csv(OUT / f"pmi_variance_decomposition_{tag}.csv", index=False)
    if rep_rows:
        pd.DataFrame(rep_rows).to_csv(OUT / f"pmi_replicate_{tag}.csv",
                                      index=False)
    print(f"\nwrote pmi_summary / pmi_contrasts / pmi_argmax_decomposition / "
          f"pmi_variance_decomposition CSVs for tag '{tag}' to {OUT}")


if __name__ == "__main__":
    main()
