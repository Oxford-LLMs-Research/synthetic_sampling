#!/usr/bin/env python
r"""Does the readout change what the model appears to know about people?

The paper ranks answer options by the mean log-probability of their own tokens.
That ranking is contaminated by how fluent each option's wording is, which is a
per-option constant: it can destroy argmax accuracy while leaving person-level
information intact. On the main grid Qwen 3 32B scores 0.118 normalized accuracy
yet carries discrimination AUC 0.593, with 83% of questions above a permutation
null. This compares readouts on identical instances in one serving, so any
difference is the readout and nothing else.

Reported per arm, and paired against echo_plain, the paper's exact condition:

  normalized accuracy   the paper's metric, (acc - 1/M) / (1 - 1/M), M distinct
                        answers given, averaged within question then across
  discrimination AUC    fix an option, ask across respondents whether it scores
                        higher for the people who chose it. Invariant to any
                        constant added to that option, so it sees signal the
                        argmax throws away. Null is a permutation of answers.
  prior-corrected acc   base rates handed to the model and one temperature
                        cross-fitted per question, separating what the model
                        knows about people from what it knows about base rates

Three controls decide how much any of it means:

  replicate     the unchanged options scored twice. Its disagreement is the
                measurement's own noise, and every agreement rate is read
                against it rather than against 1. The same model under a
                different serving stack already agrees only 64.4% of the time.
  stored echo   the fresh echo_plain arm against the paper's stored SGLang
                scores on the same example_ids: harness check and a measure of
                the serving shift on this exact sample.
  dose-response if a readout recovers signal, it should recover more where
                there is more to recover. XGBoost's per-question accuracy is the
                moderator: independent of any LLM readout, so no selection
                artefact, and already a quantity the paper reports.

    python .../analyze_readout.py --results <readout_results.jsonl>
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
ANALYSIS = REPO.parent / "analysis"
RESULTS = REPO.parent / "results"
OUT = ANALYSIS / "readout"
ALPHAS = np.concatenate([[0.0], np.geomspace(0.05, 8.0, 24)])

# Served-model tag -> results/ directory holding the paper's stored SGLang
# scores, for the section-5 harness control. Tags follow score_formats.py's
# convention (HF id, '/'->'_', lowercased). New-to-grid models (the 235B
# anchor) have no stored run and the control is skipped for them.
TAG_TO_STORED = {
    "qwen_qwen3-32b": "qwen3-32b",
    "qwen_qwen3-4b": "qwen3-4b",
    "google_gemma-3-27b-it": "gemma-3-27b-instruct",
    "openai_gpt-oss-120b": "gpt_oss",
    "meta-llama_llama-3.1-8b": "llama3.1-8b-base",
    "meta-llama_llama-3.1-8b-instruct": "llama3.1-8b-instruct",
    "meta-llama_llama-3.1-70b": "llama3.1-70b-base",
    "meta-llama_llama-3.1-70b-instruct": "llama3.1-70b-instruct",
    "allenai_olmo-3-1025-7b": "olmo3-7b-base",
    "allenai_olmo-3-7b-dpo": "olmo3-7b-dpo",
    "allenai_olmo-3-1025-32b": "olmo3-32b-base",
    "allenai_olmo-3.1-32b-instruct-dpo": "olmo3-32b-dpo",
    "deepseek-ai_deepseek-v3.1-terminus": "deepseek-v3p1-terminus",
}


def tag_of(path: Path) -> str:
    stem = path.stem
    for prefix in ("readout_results_", "readout_grid_", "readout_pmi_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


# --------------------------------------------------------------------------
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def weighted_auc(cent: np.ndarray, truth: np.ndarray, opts: list) -> float:
    a, w = [], []
    for i, o in enumerate(opts):
        y = (truth == o).astype(int)
        v = auc(cent[:, i], y)
        if not np.isnan(v):
            a.append(v)
            w.append(y.sum())
    return float(np.average(a, weights=w)) if a else float("nan")


def boot_ci(v: np.ndarray, clusters: np.ndarray, n: int = 2000,
            seed: int = 42) -> tuple[float, float]:
    """Question-clustered: resample questions, averaging within one first."""
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(clusters, return_inverse=True)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    m = np.array([v[inv == i].mean() for i in range(len(keys))])
    draws = m[rng.integers(0, len(keys), size=(n, len(keys)))].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def qmean(v: np.ndarray, clusters: np.ndarray) -> float:
    s = pd.Series(v).groupby(clusters).mean()
    return float(s.mean())


def prior_corrected(arr: np.ndarray, y: np.ndarray, opts: list,
                    folds: int = 5, seed: int = 42) -> tuple[float, float]:
    """Cross-fitted (base rates only, base rates + model) normalized accuracy."""
    n = len(y)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % folds
    cent = arr - arr.mean(axis=1, keepdims=True)
    idx = {o: i for i, o in enumerate(opts)}
    yi = np.array([idx[v] for v in y])
    prior_hit, corr_hit = [], []
    for f in range(folds):
        tr, te = fold != f, fold == f
        if not tr.any() or not te.any():
            continue
        cnt = np.bincount(yi[tr], minlength=len(opts)).astype(float)
        logp = np.log((cnt + 0.5) / (cnt.sum() + 0.5 * len(opts)))
        best, best_ll = 0.0, -np.inf
        for a in ALPHAS:
            s = a * cent[tr] + logp
            s = s - s.max(axis=1, keepdims=True)
            ll = (s[np.arange(tr.sum()), yi[tr]] - np.log(np.exp(s).sum(axis=1))).mean()
            if ll > best_ll:
                best_ll, best = ll, a
        prior_hit.append((np.full(te.sum(), logp.argmax()) == yi[te]).mean())
        corr_hit.append(((best * cent[te] + logp).argmax(axis=1) == yi[te]).mean())
    M = len(set(y))
    norm = lambda a: (a - 1 / M) / (1 - 1 / M)
    return norm(float(np.mean(prior_hit))), norm(float(np.mean(corr_hit)))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


_SCAFFOLD = re.compile(r"^(?:answer\s*[:.\-]?\s*|[-*]\s+|\d+[.)]\s*|[a-z][.)]\s+)",
                       re.IGNORECASE)


def _clean(text: str) -> str:
    t = (text or "").strip().split("\n")[0].strip()
    for _ in range(3):
        new = _SCAFFOLD.sub("", t).strip().strip('"“”\'`').strip()
        if new == t:
            break
        t = new
    return t.rstrip(".!,;: ").strip()


def relaxed_pick(raw: str, options: list) -> str | None:
    """Recover an answer the strict matcher rejected, without guessing.

    Two tiers, both of which resolve ambiguity rather than tolerating it.

    Unique prefix, allowed only where no option's wording begins another's: an
    abbreviated reply then identifies one option and nothing else.

    Longest unique containment, for the case the data actually turned up. The
    model frequently emits a valid option wrapped in commentary, and 31% of
    unmatched generations contain exactly one option. Where several are
    contained, the longest subsumes the others ("Strongly agree" contains
    "Agree"), so the longest is taken and a tie is refused. This is parsing a
    generation, not scoring an option, so preferring the longer string carries
    none of the length bias that ruled substring matching out of the scorer.

    Note the ceiling on this: `raw` stores only the first 80 characters, so an
    answer beyond that cut cannot be recovered, which affects 516 cases.
    """
    by = {_norm(o): o for o in options}
    norms = list(by)
    t = _norm(_clean(raw))
    if not t:
        return None
    if not any(a != b and b.startswith(a) for a in norms for b in norms):
        cand = [o for o in norms if o.startswith(t)]
        if len(cand) == 1:
            return by[cand[0]]
    hits = sorted((o for o in norms if o and o in t), key=len, reverse=True)
    if hits and (len(hits) == 1 or len(hits[0]) > len(hits[1])):
        return by[hits[0]]
    return None


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=SCALE / "readout_results" / "readout_results_qwen_qwen3-32b.jsonl")
    ap.add_argument("--input", type=Path, default=SCALE / "readout_set.jsonl")
    ap.add_argument("--perm", type=int, default=300)
    ap.add_argument("--stored-model", default=None,
                    help="results/ dir for the section-5 stored-scores control;"
                         " defaults via TAG_TO_STORED, skipped when unknown")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output CSVs; defaults from the "
                         "results filename so per-model runs never clobber "
                         "each other")
    args = ap.parse_args()
    tag = args.tag or tag_of(args.results)
    stored_model = args.stored_model or TAG_TO_STORED.get(tag)

    meta = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r
    rows = [json.loads(l) for l in open(args.results, encoding="utf-8")]
    print(f"{len(rows):,} scored instances, {len(meta):,} in the input set")

    arms = sorted({k.split("|", 1)[1] for r in rows for k in r["results"]})
    print(f"arms: {', '.join(arms)}\n")

    # Option hygiene. These are the paper's own harmonised option sets, and a
    # corrupted or non-substantive option is scored like any other, so it is
    # worth knowing how many there are before reading anything else.
    suspect = collections.Counter()
    for m in meta.values():
        for o in m["option_sets"]["original"]:
            if "�" in o:
                suspect["mojibake (replacement character)"] += 1
            elif re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", o):
                suspect["bare numeric code"] += 1
    if suspect:
        print("option hygiene, across the sampled sets:")
        for k, v in suspect.most_common():
            print(f"  {k}: {v} option strings")
        print()

    # ---- assemble a tidy frame -------------------------------------------
    recs = []
    for r in rows:
        m = meta.get(r["example_id"])
        if m is None:
            continue
        opts = m["option_sets"]["original"]
        truth = m["ground_truth"]
        for key, d in r["results"].items():
            setname, arm = key.split("|", 1)
            if "error" in d:
                continue
            rec = {"eid": r["example_id"], "set": setname, "arm": arm,
                   "survey": m["survey"], "target_code": m["target_code"],
                   "q": f"{m['survey']}|{m['target_code']}",
                   "truth": truth, "n_opts": len(opts)}
            if d.get("scores"):
                vals = [d["scores"].get(o, float("-inf")) for o in opts]
                rec["scores"] = vals
                # A -inf means no label token was found for that option. Only
                # label_num_natural has any (4.4% of rows: one presentation, so
                # a miss is terminal, where the Latin square recovers). Centring
                # a vector containing -inf yields NaN, so such rows are excluded
                # from AUC and prior correction. The argmax is still defined, so
                # they stay in the accuracy figures.
                rec["finite"] = all(np.isfinite(v) for v in vals)
                rec["pred"] = d.get("predicted")
            else:
                # Generation: unmatched is scored INCORRECT, never dropped.
                rec["pred"] = d.get("predicted")
                rec["unmatched"] = d.get("predicted") is None
                rec["n_draws"] = d.get("n_draws", 0)
                rec["n_matched"] = d.get("n_matched", 0)
                # Older runs dropped the relaxed tallies; recover what `raw`
                # allows, which is the greedy draw only.
                rel = d.get("picked_relaxed")
                if rel is None and d.get("predicted") is None:
                    rel = relaxed_pick(d.get("raw", ""), opts)
                rec["pred_relaxed"] = d.get("predicted") or rel
                rec["counts"] = d.get("counts") or {}
            rec["correct"] = float(rec.get("pred") == truth)
            if "pred_relaxed" in rec:
                rec["correct_relaxed"] = float(rec["pred_relaxed"] == truth)
            recs.append(rec)
    df = pd.DataFrame(recs)
    main_set = df[df["set"] == "original"].copy()

    # ---- 1. headline metrics per arm --------------------------------------
    # Option lists are not constant within a question: some respondents get an
    # extra residual category ("Don't know", "Can't choose"). AUC and prior
    # correction compare option-aligned score vectors across respondents, so
    # they are computed on each question's MODAL option set. Accuracy is
    # per-instance and needs no alignment, so it uses everything.
    modal, dropped = {}, 0
    for q, gq in main_set.groupby("q"):
        sets = collections.Counter(
            tuple(meta[e]["option_sets"]["original"]) for e in gq.eid.unique())
        modal[q] = sets.most_common(1)[0][0]
        dropped += sum(n for s, n in sets.items() if s != modal[q])
    if dropped:
        print(f"note: {dropped} of {main_set.eid.nunique()} instances use a "
              f"non-modal option set and are excluded from AUC and prior "
              f"correction (accuracy still uses all)\n")

    dropped_nf = collections.Counter()
    print("=== 1. per readout, on identical instances ===")
    print(f"{'arm':<20}{'norm acc':>10}{'AUC':>8}{'>null':>10}"
          f"{'prior only':>12}{'+model':>9}{'unmatched':>11}")
    summary = []
    for arm in arms:
        g = main_set[main_set.arm == arm]
        if g.empty:
            continue
        aucs, above, pc_prior, pc_corr = [], 0, [], []
        has_scores = "scores" in g.columns and g.scores.notna().any()
        if has_scores:
            for q, gq in g.groupby("q"):
                opts = list(modal[q])
                keep = gq[[tuple(meta[e]["option_sets"]["original"]) == modal[q]
                           for e in gq.eid]]
                keep = keep[keep.scores.notna()]
                if "finite" in keep.columns:
                    dropped_nf[arm] += int((~keep.finite.fillna(True)).sum())
                    keep = keep[keep.finite.fillna(True)]
                truth = keep.truth.to_numpy()
                if len(keep) < 20 or len(set(truth)) < 2:
                    continue
                arr = np.array(keep.scores.tolist(), float)
                cent = arr - arr.mean(axis=1, keepdims=True)
                a = weighted_auc(cent, truth, opts)
                if not np.isnan(a):
                    rng = np.random.default_rng(42)
                    null = [weighted_auc(cent, rng.permutation(truth), opts)
                            for _ in range(args.perm)]
                    null = [x for x in null if not np.isnan(x)]
                    aucs.append(a)
                    above += bool(null) and a > np.percentile(null, 97.5)
                p0, p1 = prior_corrected(arr, truth, opts)
                pc_prior.append(p0)
                pc_corr.append(p1)
        M_by_q = g.groupby("q").truth.nunique()
        acc = g.assign(M=g.q.map(M_by_q))
        acc = acc[acc.M >= 2]
        norm_acc = qmean(((acc.correct - 1 / acc.M) / (1 - 1 / acc.M)).to_numpy(),
                         acc.q.to_numpy())
        um = (g.unmatched.mean()
              if "unmatched" in g.columns and g.unmatched.notna().any() else np.nan)
        s = {"arm": arm, "norm_acc": norm_acc,
             "auc": float(np.mean(aucs)) if aucs else np.nan,
             "n_q_auc": len(aucs), "above_null": above,
             "prior_only": float(np.mean(pc_prior)) if pc_prior else np.nan,
             "corrected": float(np.mean(pc_corr)) if pc_corr else np.nan,
             "unmatched": um}
        summary.append(s)
        f = lambda v, w, p=4: (f"{v:>{w}.{p}f}" if not np.isnan(v) else f"{'-':>{w}}")
        print(f"{arm:<20}{s['norm_acc']:>10.4f}{f(s['auc'], 8, 3)}"
              f"{(str(above) + '/' + str(len(aucs))) if aucs else '-':>10}"
              f"{f(s['prior_only'], 12)}{f(s['corrected'], 9)}"
              f"{(f'{um:.1%}' if not np.isnan(um) else '-'):>11}")

    for a, n in dropped_nf.items():
        if n:
            print(f"note: {a} lost {n} rows to a missing label token; excluded "
                  f"from AUC and prior correction, kept in accuracy")

    # ---- 2. paired against the paper's condition ---------------------------
    print("\n=== 2. paired against echo_plain, question-clustered ===")
    print("negative on accuracy favours echo_plain; positive favours the arm")
    base = main_set[main_set.arm == "echo_plain"].set_index("eid")
    contrasts = []
    for arm in arms:
        if arm == "echo_plain":
            continue
        g = main_set[main_set.arm == arm].set_index("eid")
        common = base.index.intersection(g.index)
        if len(common) == 0:
            continue
        d = (g.loc[common, "correct"].to_numpy()
             - base.loc[common, "correct"].to_numpy())
        qq = base.loc[common, "q"].to_numpy()
        m = qmean(d, qq)
        lo, hi = boot_ci(d, qq)
        # Normalized contrast, the quantity app:readout quotes. M is the
        # number of distinct answers observed per question, the paper's
        # convention. Shipping it here closes audit item N26: the CSV used to
        # carry only the raw contrast while the supplement quoted normalized.
        M = base.loc[common, "q"].map(main_set.groupby("q").truth.nunique())
        keep = (M >= 2).to_numpy()
        nd = (d[keep] / (1 - 1 / M.to_numpy()[keep]))
        nm = qmean(nd, qq[keep])
        nlo, nhi = boot_ci(nd, qq[keep])
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {arm:<20} raw accuracy {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {star}"
              f"   normalized {nm:+.4f} [{nlo:+.4f}, {nhi:+.4f}]"
              f"   n={len(common):,}")
        contrasts.append({"arm": arm, "d_acc": m, "lo": lo, "hi": hi,
                          "d_norm_acc": nm, "norm_lo": nlo, "norm_hi": nhi,
                          "n": len(common)})

    # ---- 3. the measurement's own noise floor ------------------------------
    print("\n=== 3. replicate: the same options scored twice ===")
    print("this is the ceiling every agreement rate must be read against")
    rep = df[df["set"] == "original_replicate"]
    rep_rows = []
    if rep.empty:
        print("  no replicate records found")
    else:
        for arm in arms:
            a = main_set[main_set.arm == arm].set_index("eid")["pred"]
            b = rep[rep.arm == arm].set_index("eid")["pred"]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            agree = (a.loc[common] == b.loc[common]).mean()
            print(f"  {arm:<20} {agree:>6.1%} agreement   n={len(common):,}")
            rep_rows.append({"arm": arm, "agreement": agree,
                             "n": len(common)})

    # ---- 4. generation matching -------------------------------------------
    gen = main_set[main_set.arm.str.startswith("generate")]
    if not gen.empty:
        print("\n=== 4. generation: strict vs unique-prefix matching ===")
        for arm, g in gen.groupby("arm"):
            strict = 1 - g.unmatched.mean()
            rel = g.pred_relaxed.notna().mean()
            M_by_q = g.groupby("q").truth.nunique()
            gg = g.assign(M=g.q.map(M_by_q))
            gg = gg[gg.M >= 2]
            na_s = qmean(((gg.correct - 1 / gg.M) / (1 - 1 / gg.M)).to_numpy(),
                         gg.q.to_numpy())
            na_r = qmean(((gg.correct_relaxed - 1 / gg.M) / (1 - 1 / gg.M)).to_numpy(),
                         gg.q.to_numpy())
            print(f"  {arm:<20} matched strict {strict:>6.1%}  relaxed {rel:>6.1%}"
                  f"   norm acc {na_s:+.4f} -> {na_r:+.4f}")
        print("  unmatched replies are scored INCORRECT, not dropped")

    # ---- 5. harness control against the stored run ------------------------
    print("\n=== 5. fresh echo_plain against the paper's stored scores ===")
    stored = {}
    if stored_model is None:
        print(f"  no stored run mapped for tag '{tag}' (new-to-grid model?); "
              f"skipping")
    else:
        d = RESULTS / stored_model
        want = set(base.index)
        for f in sorted(d.glob("*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r["example_id"] in want:
                    stored[r["example_id"]] = r
    if stored_model is not None and not stored:
        print("  stored results not found; skipping")
    elif stored:
        ids = [i for i in base.index if i in stored]
        agree = np.mean([stored[i]["predicted"] == base.loc[i, "pred"] for i in ids])
        shift = np.mean([
            np.mean(list(stored[i]["option_logprobs"].values()))
            - np.mean([v for v in base.loc[i, "scores"] if np.isfinite(v)])
            for i in ids])
        print(f"  {len(ids):,} instances matched")
        print(f"  predictions agree on {agree:.1%}")
        print(f"  mean option logprob shift (stored - fresh) {shift:+.4f} nats")
        print("  read the 64.4% cross-stack figure from control_check.py "
              "alongside this")

    # ---- 6. dose-response --------------------------------------------------
    print("\n=== 6. does a readout recover more where there is more to recover? ===")
    xgb = pd.read_csv(ANALYSIS / "xgboost_baseline" / "results.csv")
    xgb = xgb[xgb.profile_type == "s6m4"][["survey", "target_code", "xgb_norm_acc"]]
    xgb["q"] = xgb.survey + "|" + xgb.target_code
    xmap = dict(zip(xgb.q, xgb.xgb_norm_acc))
    for arm in arms:
        if arm == "echo_plain" or arm.startswith("generate"):
            continue
        g = main_set[main_set.arm == arm].set_index("eid")
        common = base.index.intersection(g.index)
        if len(common) == 0:
            continue
        d = pd.DataFrame({
            "q": base.loc[common, "q"].to_numpy(),
            "d": (g.loc[common, "correct"].to_numpy()
                  - base.loc[common, "correct"].to_numpy())})
        per_q = d.groupby("q").d.mean().reset_index()
        per_q["xgb"] = per_q.q.map(xmap)
        per_q = per_q.dropna()
        if len(per_q) < 8:
            continue
        r = per_q[["d", "xgb"]].corr(method="spearman").iloc[0, 1]
        print(f"  {arm:<20} Spearman(gain, XGBoost accuracy) {r:+.3f}"
              f"   n={len(per_q)} questions")
    print("  a positive correlation means the readout helps most where the "
          "features\n  carry the most signal, which is what genuine recovery "
          "looks like")

    OUT.mkdir(parents=True, exist_ok=True)
    # Tag-suffixed always: the fixed names readout_summary.csv /
    # readout_contrasts.csv were clobbered per run (the unsuffixed file on disk
    # is silently the Qwen 3 4B run, a documented trap).
    pd.DataFrame(summary).to_csv(OUT / f"readout_summary_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / f"readout_contrasts_{tag}.csv",
                                   index=False)
    if rep_rows:
        pd.DataFrame(rep_rows).to_csv(OUT / f"readout_replicate_{tag}.csv",
                                      index=False)
    print(f"\nwrote readout_summary / readout_contrasts"
          f"{' / readout_replicate' if rep_rows else ''} CSVs for tag "
          f"'{tag}' to {OUT}")


if __name__ == "__main__":
    main()
