"""Claim checks on the Phase 2 grid: dissent, entropy, margins, the XGBoost gap.

    python scripts/phase2/grid_claims.py \
        ../outputs_recovered/phase2_grid_r0-50 outputs/phase2/results/xgb_grid_r0-50 \
        --instances outputs/phase2/inputs/instances_r0-50_short.jsonl \
        --first-look ../analysis/phase2/first_look_grid_16oct.csv \
        --out ../analysis/phase2/grid_claims

Written 25 Oct 2026 to put the AAAI-27 review claims on the grid rather than on
the 1-3 model batteries (aaai_reviews/ASSESSMENT_2026-10-24.md). Each root holds
serving folders of `*_shard*of*.jsonl`; XGBoost out-of-fold dumps from
`xgb_grid_baseline.py` are read as servings.

One streaming pass per serving builds per-cell sufficient statistics, a cell
being survey x target x option set x country: confusion counts of truth against
the argmax, and summed predicted distributions, raw (softmax of the stored
scores) and after one temperature cross-fitted over targets (GroupKFold, 5
folds, NLL; T0.1's convention). Every metric is a function of those counts.

Averaging follows first_look.py: normalized accuracy (acc - 1/M) / (1 - 1/M)
with M the option count, averaged within question (n-weighted over the five WVS
targets carrying two option sets), then across questions. Intervals resample
questions (2,000 draws, seed 42).

Dissent, pooled split (the paper's Table 1 split): the mode is the group's most
common recorded answer over all countries. Two nulls, per group:
- N0, no person information: predictions independent of the respondent, with
  the serving's own prediction shares (a within-question permutation). Its
  penalty is what output concentration alone produces.
- N1, evenly spread person information: with probability lam the truth, else a
  draw from q = (p_y - lam p_t) / (1 - lam). This matches the serving's
  prediction shares AND its accuracy; lam is clipped to [0, min p_y / p_t] and
  clipping is counted. A serving that never emits some answer forces lam = 0,
  so N1 collapses to N0 in those groups.
The primary contrast is the lift over N0 in each group (observed minus N0):
what the serving's person information buys modal respondents and dissenters.
The excess of the observed penalty over N0's equals lift_mod - lift_dis.

Margins, cells with n >= 30: total variation from the cell's recorded shares to
the tally, the raw and tempered mean distributions, the pooled margin, the
leave-one-country-out margin (groups asked in 2+ countries), and a control: the
expected TV of a tally of n draws from the leave-one-out margin (Monte Carlo,
400 draws, seeded), which charges the baseline the same discretisation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold

ARMS = ("original|label_num", "original|xgb")
MAXM = 9
LOGP_FLOOR = math.log(1e-12)
N_BOOT = 2000
CELL_DISSENT_MIN = 20
CELL_MARGIN_MIN = 30
CONTROL_DRAWS = 400
BINS = (0.0, 0.4, 0.6, 0.8, 1.0000001)

_META: dict = {}
_GROUP_OPTIONS: list = []
_CELL_GROUP: np.ndarray = np.zeros(0)


# ---------------------------------------------------------------- instances


def load_instances(path: str):
    """Cells, groups and the example_id -> cell map, from the grid input."""
    group_ix, cell_ix = {}, {}
    groups, cells, meta = [], [], {}
    truth = collections.Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gk = (d["survey"], d["target_code"], tuple(d["options"]))
            if gk not in group_ix:
                group_ix[gk] = len(groups)
                groups.append(gk)
            g = group_ix[gk]
            ck = (g, d["country"])
            if ck not in cell_ix:
                cell_ix[ck] = len(cells)
                cells.append((g, d["country"], d.get("country_label", "")))
            c = cell_ix[ck]
            meta[d["example_id"]] = c
            truth[(c, d["ground_truth_index"])] += 1
    if max(len(gk[2]) for gk in groups) > MAXM:
        raise SystemExit("an option set exceeds MAXM; this script is for the short-list grid")
    cell_truth = np.zeros((len(cells), MAXM), dtype=np.int64)
    for (c, t), n in truth.items():
        cell_truth[c, t] = n
    return groups, cells, meta, cell_truth


# ------------------------------------------------------------------ servings


def _init(meta, group_options, cell_group):
    global _META, _GROUP_OPTIONS, _CELL_GROUP
    _META, _GROUP_OPTIONS, _CELL_GROUP = meta, group_options, cell_group


def _softmax(s: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = s / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(log_t: float, s: np.ndarray, t: np.ndarray) -> float:
    z = s / math.exp(log_t)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    lp = z[np.arange(len(t)), t] - lse
    return float(-np.maximum(lp, LOGP_FLOOR).mean())


def read_serving(folder: str) -> dict:
    """One streaming pass: per-cell confusion counts and summed distributions."""
    name = os.path.basename(folder.rstrip("/\\"))
    rows, errors, unknown, dup = 0, 0, 0, 0
    seen = set()
    ci, tt, yy, ss = [], [], [], []
    for path in sorted(glob.glob(os.path.join(folder, "*_shard*of*.jsonl"))):
        with open(path, "rb") as fh:
            for line in fh:
                r = json.loads(line)
                rows += 1
                rec = next((r["results"][a] for a in ARMS if a in r["results"]), None)
                if rec is None or "error" in rec:
                    errors += 1
                    continue
                c = _META.get(r["example_id"])
                if c is None:
                    unknown += 1
                    continue
                if r["example_id"] in seen:
                    dup += 1
                    continue
                seen.add(r["example_id"])
                opts = _GROUP_OPTIONS[_CELL_GROUP[c]]
                sc = rec["scores"]
                row = [sc[o] for o in opts] + [-math.inf] * (MAXM - len(opts))
                ci.append(c)
                tt.append(r["ground_truth_index"])
                yy.append(rec["predicted_index"])
                ss.append(row)
    ci = np.asarray(ci, dtype=np.int64)
    tt = np.asarray(tt, dtype=np.int64)
    yy = np.asarray(yy, dtype=np.int64)
    ss = np.asarray(ss, dtype=np.float64)
    argmax_mismatch = int((ss.argmax(axis=1) != yy).sum())

    # cross-fitted temperature, folds over targets
    groups = _CELL_GROUP[ci]
    p_t = np.zeros_like(ss)
    temps = []
    for train, test in GroupKFold(n_splits=5).split(ss, tt, groups):
        res = minimize_scalar(_nll, bounds=(math.log(0.05), math.log(50.0)),
                              args=(ss[train], tt[train]), method="bounded",
                              options={"xatol": 1e-4})
        temp = math.exp(res.x)
        temps.append(temp)
        p_t[test] = _softmax(ss[test], temp)
    p_1 = _softmax(ss)

    n_cells = len(_CELL_GROUP)
    conf = np.zeros((n_cells, MAXM, MAXM), dtype=np.int64)
    np.add.at(conf, (ci, tt, yy), 1)
    soft1 = np.zeros((n_cells, MAXM))
    softt = np.zeros((n_cells, MAXM))
    np.add.at(soft1, ci, p_1)
    np.add.at(softt, ci, p_t)
    # soft confusion: distributions summed by truth class, for a sampled panel
    sconf1 = np.zeros((n_cells, MAXM, MAXM))
    sconft = np.zeros((n_cells, MAXM, MAXM))
    np.add.at(sconf1, (ci, tt), p_1)
    np.add.at(sconft, (ci, tt), p_t)
    nll1 = _nll(0.0, ss, tt)
    return {"name": name, "rows": rows, "errors": errors, "unknown": unknown,
            "duplicates": dup, "scored": len(ci), "argmax_mismatch": argmax_mismatch,
            "temps": temps, "nll_raw": nll1, "conf": conf, "soft1": soft1,
            "softt": softt, "sconf1": sconf1, "sconft": sconft}


# ------------------------------------------------------------------- metrics


def norm(acc, m):
    return (acc - 1.0 / m) / (1.0 - 1.0 / m)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


class QuestionMean:
    """Mean within question (n-weighted over groups), then across questions."""

    def __init__(self, group_question: np.ndarray, n_questions: int, seed: int = 42):
        self.gq = group_question
        self.nq = n_questions
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, n_questions, size=(N_BOOT, n_questions))

    def per_question(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        ok = ~np.isnan(values)
        num = np.bincount(self.gq[ok], weights=(values * weights)[ok], minlength=self.nq)
        den = np.bincount(self.gq[ok], weights=weights[ok], minlength=self.nq)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    def mean_ci(self, values: np.ndarray, weights: np.ndarray):
        pq = self.per_question(values, weights)
        ok = ~np.isnan(pq)
        mean = float(pq[ok].mean()) if ok.any() else float("nan")
        sel = np.where(ok[self.draws], pq[self.draws], 0.0)
        cnt = ok[self.draws].sum(axis=1)
        boot = sel.sum(axis=1) / np.maximum(cnt, 1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return mean, float(lo), float(hi), int(ok.sum())

    def mean_ci_groups(self, values, weights, group_index):
        """mean_ci for values given only on a subset of groups."""
        full_v = np.full(len(self.gq), np.nan)
        full_w = np.zeros(len(self.gq))
        full_v[group_index] = values
        full_w[group_index] = weights
        return self.mean_ci(full_v, full_w)


def loo_mode_accuracy(counts: np.ndarray) -> float:
    """Accuracy of predicting each respondent by the mode of everyone else."""
    n = counts.sum()
    hit = 0
    for a in np.flatnonzero(counts):
        c = counts.copy()
        c[a] -= 1
        hit += counts[a] * (int(np.argmax(c)) == a)
    return hit / n


def group_metrics(G: np.ndarray, soft1: np.ndarray, softt: np.ndarray, m_opts: int,
                  sc1: np.ndarray, sct: np.ndarray) -> dict:
    """Every per-group quantity, from the group's confusion matrices."""
    M = m_opts
    C = G[:M, :M].astype(float)
    n = C.sum()
    rows = C.sum(axis=1)
    cols = C.sum(axis=0)
    p_t, p_y = rows / n, cols / n
    acc = np.trace(C) / n
    mode = int(np.argmax(rows))
    out = {"n": n, "M": M, "acc": acc, "norm": norm(acc, M),
           "modal_share": p_t[mode], "modal_norm": norm(p_t[mode], M),
           "modal_loo_norm": norm(loo_mode_accuracy(rows.astype(np.int64)), M),
           "emits_modal": p_y[mode], "top_is_mode": float(int(np.argmax(cols)) == mode),
           "constant_share": p_y.max()}
    n_mod, n_dis = rows[mode], n - rows[mode]
    if n_mod > 0 and n_dis > 0:
        a_mod = C[mode, mode] / n_mod
        a_dis = (np.trace(C) - C[mode, mode]) / n_dis
        others = np.arange(M) != mode
        a_mod0 = p_y[mode]
        a_dis0 = float((p_y[others] * p_t[others]).sum() / (1 - p_t[mode]))
        base = float((p_y * p_t).sum())
        denom = 1 - float((p_t ** 2).sum())
        lam = (acc - base) / denom if denom > 0 else 0.0
        pos = p_t > 0
        lam_max = float((p_y[pos] / p_t[pos]).min())
        clip_low, clip_high = lam < 0, lam > lam_max
        lam = min(max(lam, 0.0), lam_max)
        q = p_y - lam * p_t
        a_mod1 = lam + q[mode]
        a_dis1 = lam + float((q[others] * p_t[others]).sum() / (1 - p_t[mode]))
        # a sampled panel: expected accuracy is the probability on the truth;
        # the reference sampler draws from the true marginal with no person signal
        samp = {}
        for tag, S in (("raw", sc1[:M, :M]), ("T", sct[:M, :M])):
            d = np.diag(S)
            samp[f"mod_norm_samp_{tag}"] = norm(d[mode] / n_mod, M)
            samp[f"dis_norm_samp_{tag}"] = norm((d.sum() - d[mode]) / n_dis, M)
            samp[f"norm_samp_{tag}"] = norm(d.sum() / n, M)
        samp["mod_norm_samp_marg"] = norm(p_t[mode], M)
        samp["dis_norm_samp_marg"] = norm(
            float((p_t[others] ** 2).sum() / (1 - p_t[mode])), M)
        out.update(samp)
        out.update({
            "acc_mod": a_mod, "acc_dis": a_dis,
            "mod_norm": norm(a_mod, M), "dis_norm": norm(a_dis, M),
            "mod_norm_n0": norm(a_mod0, M), "dis_norm_n0": norm(a_dis0, M),
            "mod_norm_n1": norm(a_mod1, M), "dis_norm_n1": norm(a_dis1, M),
            "lam": lam, "n1_clip_low": float(clip_low), "n1_clip_high": float(clip_high),
            "n1_acc_check": p_t[mode] * a_mod1 + (1 - p_t[mode]) * a_dis1,
        })
        order = np.argsort(-rows, kind="stable")
        second, rare = order[1], order[2:]
        rare = rare[rows[rare] > 0]
        if rows[second] > 0:
            out["acc_second"] = C[second, second] / rows[second]
            out["second_share_of_dissent"] = rows[second] / n_dis
        if len(rare):
            out["acc_rare"] = C[rare, rare].sum() / rows[rare].sum()
    h_t = entropy(p_t)
    if h_t > 0:
        out["er_hard"] = entropy(p_y) / h_t
        out["er_soft"] = entropy(soft1[:M] / soft1[:M].sum()) / h_t
        out["er_softT"] = entropy(softt[:M] / softt[:M].sum()) / h_t
    return out


def margin_control(cell_truth: np.ndarray, cells: list, groups: list, seed: int = 42):
    """Model-free cell quantities: pooled and LOO margins and the tallied control."""
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    g_of = np.array([c[0] for c in cells])
    g_truth = np.zeros((n_groups, MAXM))
    np.add.at(g_truth, g_of, cell_truth)
    countries = np.bincount(g_of[cell_truth.sum(axis=1) > 0], minlength=n_groups)
    out = []
    for c, (g, country, label) in enumerate(cells):
        n_c = cell_truth[c].sum()
        if n_c < CELL_MARGIN_MIN:
            continue
        M = len(groups[g][2])
        q = cell_truth[c, :M] / n_c
        p = g_truth[g, :M] / g_truth[g, :M].sum()
        rec = {"cell": c, "group": g, "country": country, "n": int(n_c),
               "tv_pooled": tv(p, q), "tv_loo": float("nan"),
               "tv_loo_tally": float("nan")}
        rest = g_truth[g, :M] - cell_truth[c, :M]
        if countries[g] >= 2 and rest.sum() > 0:
            r = rest / rest.sum()
            rec["tv_loo"] = tv(r, q)
            draws = rng.multinomial(int(n_c), r, size=CONTROL_DRAWS) / n_c
            rec["tv_loo_tally"] = float(0.5 * np.abs(draws - q).sum(axis=1).mean())
        out.append(rec)
    return out


def summarise(res: dict, groups: list, cells: list, cell_truth: np.ndarray,
              qm: QuestionMean, control: list) -> tuple:
    conf, soft1, softt = res["conf"], res["soft1"], res["softt"]
    g_of = np.array([c[0] for c in cells])
    n_groups = len(groups)
    G = np.zeros((n_groups, MAXM, MAXM))
    S1 = np.zeros((n_groups, MAXM))
    ST = np.zeros((n_groups, MAXM))
    SC1 = np.zeros((n_groups, MAXM, MAXM))
    SCT = np.zeros((n_groups, MAXM, MAXM))
    np.add.at(G, g_of, conf)
    np.add.at(S1, g_of, soft1)
    np.add.at(ST, g_of, softt)
    np.add.at(SC1, g_of, res["sconf1"])
    np.add.at(SCT, g_of, res["sconft"])

    rows = []
    for g, gk in enumerate(groups):
        if G[g].sum() == 0:
            continue
        rec = group_metrics(G[g], S1[g], ST[g], len(gk[2]), SC1[g], SCT[g])
        rec.update({"group": g, "survey": gk[0], "target_code": gk[1]})
        rows.append(rec)
    gdf = pd.DataFrame(rows)
    gix = gdf["group"].to_numpy()
    w = gdf["n"].to_numpy()

    def col(name):
        return gdf[name].to_numpy(dtype=float) if name in gdf else np.full(len(gdf), np.nan)

    def mc(values, weights=w):
        return qm.mean_ci_groups(values, weights, gix)

    summary = {"serving": res["name"], "rows": res["rows"], "errors": res["errors"],
               "unknown": res["unknown"], "duplicates": res["duplicates"],
               "scored": res["scored"], "argmax_mismatch": res["argmax_mismatch"],
               "temp_mean": float(np.mean(res["temps"])),
               "temp_min": float(np.min(res["temps"])),
               "temp_max": float(np.max(res["temps"])), "nll_raw": res["nll_raw"]}
    summary["acc"] = float(np.trace(G, axis1=1, axis2=2).sum() / G.sum())
    for key in ("norm", "modal_norm", "modal_loo_norm", "emits_modal", "top_is_mode",
                "constant_share", "modal_share"):
        m, lo, hi, nq = mc(col(key))
        summary[key] = m
        if key == "norm":
            summary.update({"norm_lo": lo, "norm_hi": hi, "questions": nq})
    summary["emits_modal_inst"] = float(
        (gdf["emits_modal"] * gdf["n"]).sum() / gdf["n"].sum())
    summary["modal_share_inst"] = float(
        (gdf["modal_share"] * gdf["n"]).sum() / gdf["n"].sum())

    # dissent: penalty per question, then mean, so CIs are on the penalty itself
    both = ~np.isnan(col("mod_norm")) & ~np.isnan(col("dis_norm"))
    for tag, mod, dis in (("obs", "mod_norm", "dis_norm"),
                          ("n0", "mod_norm_n0", "dis_norm_n0"),
                          ("n1", "mod_norm_n1", "dis_norm_n1")):
        vm = np.where(both, col(mod), np.nan)
        vd = np.where(both, col(dis), np.nan)
        summary[f"mod_{tag}"] = mc(vm)[0]
        summary[f"dis_{tag}"] = mc(vd)[0]
        m, lo, hi, nq = mc(vm - vd)
        summary.update({f"pen_{tag}": m, f"pen_{tag}_lo": lo, f"pen_{tag}_hi": hi})
    for tag in ("n0", "n1"):
        ex = (col("mod_norm") - col("dis_norm")) - (col(f"mod_norm_{tag}") - col(f"dis_norm_{tag}"))
        m, lo, hi, _ = mc(np.where(both, ex, np.nan))
        summary.update({f"excess_{tag}": m, f"excess_{tag}_lo": lo, f"excess_{tag}_hi": hi})
    # lift over N0: what the serving's person information buys each group;
    # excess_n0 is exactly lift_mod - lift_dis
    for grp in ("mod", "dis"):
        lift = col(f"{grp}_norm") - col(f"{grp}_norm_n0")
        m, lo, hi, _ = mc(np.where(both, lift, np.nan))
        summary.update({f"lift_{grp}": m, f"lift_{grp}_lo": lo, f"lift_{grp}_hi": hi})
    summary["n1_clip_low"] = int(np.nansum(col("n1_clip_low")))
    summary["n1_clip_high"] = int(np.nansum(col("n1_clip_high")))
    ok = both & (col("n1_clip_low") == 0) & (col("n1_clip_high") == 0)
    summary["n1_acc_check_maxdev"] = float(
        np.abs(col("n1_acc_check")[ok] - col("acc")[ok]).max()) if ok.any() else float("nan")
    summary["dissent_groups"] = int(both.sum())
    # the same split for a sampled panel, raw and tempered, and the marginal sampler
    for tag in ("raw", "T", "marg"):
        vm = np.where(both, col(f"mod_norm_samp_{tag}"), np.nan)
        vd = np.where(both, col(f"dis_norm_samp_{tag}"), np.nan)
        summary[f"mod_samp_{tag}"] = mc(vm)[0]
        summary[f"dis_samp_{tag}"] = mc(vd)[0]
        m, lo, hi, _ = mc(vm - vd)
        summary.update({f"pen_samp_{tag}": m, f"pen_samp_{tag}_lo": lo,
                        f"pen_samp_{tag}_hi": hi})
    for tag in ("raw", "T"):
        summary[f"norm_samp_{tag}"] = mc(col(f"norm_samp_{tag}"))[0]
    for key in ("acc_second", "acc_rare", "second_share_of_dissent"):
        summary[key] = mc(col(key))[0]

    # entropy, question level
    for key in ("er_hard", "er_soft", "er_softT"):
        v = col(key)
        summary[key] = mc(v)[0]
        summary[f"{key}_pct_below1"] = float((v[~np.isnan(v)] < 1).mean())

    # country split (cells with n >= 20), cell-averaged as the equity audit did
    crow = []
    for c, (g, country, label) in enumerate(cells):
        C = conf[c].astype(float)
        n_c = C.sum()
        if n_c < CELL_DISSENT_MIN:
            continue
        M = len(groups[g][2])
        rws = C.sum(axis=1)[:M]
        mode = int(np.argmax(rws))
        if 0 < rws[mode] < n_c:
            crow.append((norm(C[mode, mode] / rws[mode], M),
                         norm((np.trace(C) - C[mode, mode]) / (n_c - rws[mode]), M)))
    if crow:
        arr = np.array(crow)
        summary.update({"cty_mod": float(arr[:, 0].mean()), "cty_dis": float(arr[:, 1].mean()),
                        "cty_pen": float((arr[:, 0] - arr[:, 1]).mean()),
                        "cty_cells": len(crow)})

    # margins
    mrows = []
    for rec in control:
        c = rec["cell"]
        n_c = rec["n"]
        M = len(groups[rec["group"]][2])
        q = cell_truth[c, :M] / n_c
        tally = conf[c].sum(axis=0)[:M] / n_c
        mrows.append({**rec, "tv_tally": tv(tally, q), "tv_soft": tv(soft1[c, :M] / n_c, q),
                      "tv_softT": tv(softt[c, :M] / n_c, q)})
    mdf = pd.DataFrame(mrows)
    loo = mdf["tv_loo"].notna()
    summary["margin_cells"] = len(mdf)
    summary["margin_cells_loo"] = int(loo.sum())
    for key in ("tv_tally", "tv_soft", "tv_softT", "tv_pooled", "tv_loo", "tv_loo_tally"):
        summary[f"{key}_mean"] = float(mdf.loc[loo, key].mean())
        summary[f"{key}_median"] = float(mdf.loc[loo, key].median())
    for key in ("tv_tally", "tv_soft", "tv_softT"):
        summary[f"beats_loo_{key}"] = float((mdf.loc[loo, key] < mdf.loc[loo, "tv_loo"]).mean())
    summary["beats_control_tv_tally"] = float(
        (mdf.loc[loo, "tv_tally"] < mdf.loc[loo, "tv_loo_tally"]).mean())

    # consensus bins (pooled split)
    brows = []
    share = col("modal_share")
    for lo_b, hi_b in zip(BINS[:-1], BINS[1:]):
        sel = both & (share >= lo_b) & (share < hi_b)
        brows.append({"serving": res["name"], "bin_lo": lo_b, "bin_hi": min(hi_b, 1.0),
                      "groups": int(sel.sum()),
                      "mod_obs": mc(np.where(sel, col("mod_norm"), np.nan))[0],
                      "dis_obs": mc(np.where(sel, col("dis_norm"), np.nan))[0],
                      "mod_n1": mc(np.where(sel, col("mod_norm_n1"), np.nan))[0],
                      "dis_n1": mc(np.where(sel, col("dis_norm_n1"), np.nan))[0]})
    gdf.insert(0, "serving", res["name"])
    mdf.insert(0, "serving", res["name"])
    return summary, gdf, mdf, brows


def compare_xgb(gdfs: dict, qm: QuestionMean) -> list:
    """Where the LLM-XGBoost gap sits: modal against dissenting respondents."""
    out = []
    xgbs = [k for k in gdfs if k.startswith("xgb_")]
    for x in xgbs:
        xg = gdfs[x].set_index("group")
        for s, g in gdfs.items():
            if s.startswith("xgb_"):
                continue
            g = g.set_index("group")
            common = g.index.intersection(xg.index)
            mask = (g.loc[common, "acc_mod"].notna().to_numpy()
                    & xg.loc[common, "acc_mod"].notna().to_numpy())
            both = common[mask]
            gi = np.asarray(both)
            w = g.loc[both, "n"].to_numpy()
            rec = {"serving": s, "xgb": x, "groups": len(both)}
            for key in ("norm", "mod_norm", "dis_norm"):
                d = g.loc[both, key].to_numpy() - xg.loc[both, key].to_numpy()
                m, lo, hi, _ = qm.mean_ci_groups(d, w, gi)
                rec.update({f"d_{key}": m, f"d_{key}_lo": lo, f"d_{key}_hi": hi})
            # instance-weighted raw-accuracy gap, split by where it accrues
            share = g.loc[both, "modal_share"].to_numpy()
            n = w.astype(float)
            d_mod = g.loc[both, "acc_mod"].to_numpy() - xg.loc[both, "acc_mod"].to_numpy()
            d_dis = g.loc[both, "acc_dis"].to_numpy() - xg.loc[both, "acc_dis"].to_numpy()
            from_mod = float((n * share * d_mod).sum() / n.sum())
            from_dis = float((n * (1 - share) * d_dis).sum() / n.sum())
            rec.update({"raw_gap": from_mod + from_dis, "raw_gap_from_modal": from_mod,
                        "raw_gap_from_dissent": from_dis})
            out.append(rec)
    return out


# ---------------------------------------------------------------------- main


def check_first_look(summaries: list, path: str) -> list:
    """Gate: overall accuracy must reproduce first_look.py's published value."""
    problems = []
    with open(path, encoding="utf-8") as fh:
        ref = {r["serving"]: r for r in csv.DictReader(fh)}
    for s in summaries:
        r = ref.get(s["serving"])
        if r is None:
            continue
        if abs(s["acc"] - float(r["acc"])) > 5e-7:
            problems.append(f"{s['serving']}: acc {s['acc']:.6f} vs first look {r['acc']}")
        if int(r["rows"]) != s["rows"]:
            problems.append(f"{s['serving']}: rows {s['rows']} vs first look {r['rows']}")
    return problems


def write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-look", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="comma-separated serving names (smoke)")
    args = ap.parse_args()

    t0 = time.time()
    groups, cells, meta, cell_truth = load_instances(args.instances)
    group_options = [gk[2] for gk in groups]
    cell_group = np.array([c[0] for c in cells])
    print(f"{len(meta)} instances, {len(groups)} groups, {len(cells)} cells "
          f"({time.time() - t0:.0f}s)", flush=True)

    folders = []
    for root in args.roots:
        for f in sorted(glob.glob(os.path.join(root, "*", ""))):
            if glob.glob(os.path.join(f, "*_shard*of*.jsonl")):
                folders.append(f)
    if args.only:
        keep = set(args.only.split(","))
        folders = [f for f in folders if os.path.basename(f.rstrip("/\\")) in keep]
    print(f"{len(folders)} servings", flush=True)

    questions = sorted({(gk[0], gk[1]) for gk in groups})
    q_ix = {q: i for i, q in enumerate(questions)}
    qm = QuestionMean(np.array([q_ix[(gk[0], gk[1])] for gk in groups]), len(questions))
    control = margin_control(cell_truth, cells, groups)

    os.makedirs(os.path.join(args.out, "stats"), exist_ok=True)
    summaries, gdfs, mdfs, bins, problems = [], {}, [], [], []
    with Pool(args.workers, initializer=_init,
              initargs=(meta, group_options, cell_group)) as pool:
        for res in pool.imap_unordered(read_serving, folders):
            name = res["name"]
            truth_rows = res["conf"].sum(axis=2)
            is_xgb = name.startswith("xgb_")
            covered = truth_rows.sum(axis=1) > 0
            if not np.array_equal(truth_rows[covered], cell_truth[covered]):
                problems.append(f"{name}: confusion row sums differ from instance truth")
            if not is_xgb and not np.array_equal(truth_rows, cell_truth):
                problems.append(f"{name}: not every instance scored exactly once")
            if res["argmax_mismatch"]:
                problems.append(f"{name}: {res['argmax_mismatch']} predictions differ from argmax")
            np.savez_compressed(os.path.join(args.out, "stats", f"{name}.npz"),
                                conf=res["conf"], soft1=res["soft1"], softt=res["softt"],
                                sconf1=res["sconf1"], sconft=res["sconft"],
                                temps=np.array(res["temps"]))
            s, gdf, mdf, b = summarise(res, groups, cells, cell_truth, qm, control)
            summaries.append(s)
            gdfs[name] = gdf
            mdfs.append(mdf)
            bins.extend(b)
            print(f"{name:52s} acc {s['acc']:.4f} norm {s['norm']:.4f} pen {s['pen_obs']:+.3f} "
                  f"(N0 {s['pen_n0']:+.3f}, N1 {s['pen_n1']:+.3f}) er {s['er_hard']:.2f}/"
                  f"{s['er_soft']:.2f}/{s['er_softT']:.2f} tv {s['tv_tally_mean']:.3f} "
                  f"loo {s['tv_loo_mean']:.3f} [{time.time() - t0:.0f}s]", flush=True)

    if args.first_look:
        problems += check_first_look(summaries, args.first_look)

    summaries.sort(key=lambda s: (s["serving"].startswith("xgb_"), -s["norm"]))
    write_csv(summaries, os.path.join(args.out, "by_serving.csv"))
    pd.concat(gdfs.values()).to_csv(os.path.join(args.out, "by_group.csv"), index=False,
                                    float_format="%.10f")
    pd.concat(mdfs).to_csv(os.path.join(args.out, "by_cell.csv"), index=False,
                           float_format="%.10f")
    write_csv(bins, os.path.join(args.out, "consensus_bins.csv"))
    write_csv(compare_xgb(gdfs, qm), os.path.join(args.out, "xgb_gap.csv"))
    pd.DataFrame([{"cell": c, "group": g, "survey": groups[g][0],
                   "target_code": groups[g][1], "country": country, "country_label": label,
                   "n": int(cell_truth[c].sum())} for c, (g, country, label) in enumerate(cells)]
                 ).to_csv(os.path.join(args.out, "cells.csv"), index=False)
    with open(os.path.join(args.out, "gates.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(problems) if problems else "all gates passed\n")
    print("gates:", "; ".join(problems) if problems else "all passed")
    print(f"wrote {args.out} [{time.time() - t0:.0f}s]")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
