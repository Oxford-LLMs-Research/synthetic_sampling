"""Tests for the null models and margins in scripts/phase2/grid_claims.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase2" / "grid_claims.py"


def load():
    spec = importlib.util.spec_from_file_location("grid_claims", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gc = load()
M = 4


def padded(C: np.ndarray) -> np.ndarray:
    out = np.zeros((gc.MAXM, gc.MAXM))
    out[: C.shape[0], : C.shape[1]] = C
    return out


def metrics(C: np.ndarray) -> dict:
    zero = np.zeros(gc.MAXM)
    return gc.group_metrics(padded(C), zero + 1, zero + 1, C.shape[0],
                            np.zeros((gc.MAXM, gc.MAXM)), np.zeros((gc.MAXM, gc.MAXM)))


def test_n1_is_exact_on_data_from_the_n1_family():
    # truth shares p_t; with probability lam the truth, else a draw from q
    p_t = np.array([0.5, 0.25, 0.15, 0.10])
    q = np.array([0.40, 0.30, 0.20, 0.10])
    lam, n = 0.3, 10_000
    C = n * p_t[:, None] * (lam * np.eye(M) + (1 - lam) * q[None, :])
    m = metrics(C)
    assert abs(m["lam"] - lam) < 1e-12
    assert abs(m["mod_norm"] - m["mod_norm_n1"]) < 1e-12
    assert abs(m["dis_norm"] - m["dis_norm_n1"]) < 1e-12
    assert abs(m["n1_acc_check"] - m["acc"]) < 1e-12


def test_n0_matches_a_permutation_of_predictions():
    rng = np.random.default_rng(0)
    C = rng.integers(1, 40, size=(M, M)).astype(float)
    m = metrics(C)
    truth = np.repeat(np.arange(M), C.sum(axis=1).astype(int))
    pred = np.concatenate([np.repeat(np.arange(M), C[i].astype(int)) for i in range(M)])
    mode = int(np.argmax(C.sum(axis=1)))
    mods, diss = [], []
    for _ in range(4000):
        p = rng.permutation(pred)
        mods.append((p[truth == mode] == mode).mean())
        diss.append((p[truth != mode] == truth[truth != mode]).mean())
    assert abs(gc.norm(np.mean(mods), M) - m["mod_norm_n0"]) < 0.01
    assert abs(gc.norm(np.mean(diss), M) - m["dis_norm_n0"]) < 0.01


def test_loo_mode_accuracy():
    assert gc.loo_mode_accuracy(np.array([3, 1])) == 0.75
    assert gc.loo_mode_accuracy(np.array([2, 2])) == 0.0


def test_pooled_margin_identity():
    # AI review: p = w q + (1 - w) r implies TV(p, q) = (1 - w) TV(r, q)
    groups = [("s", "t", ("a", "b", "c"))]
    cells = [(0, "X", "X"), (0, "Y", "Y")]
    truth = np.zeros((2, gc.MAXM), dtype=np.int64)
    truth[0, :3] = [20, 10, 10]
    truth[1, :3] = [10, 30, 20]
    rec = {r["country"]: r for r in gc.margin_control(truth, cells, groups)}
    w = 40 / 100
    assert abs(rec["X"]["tv_pooled"] - (1 - w) * rec["X"]["tv_loo"]) < 1e-12


def test_marginal_sampler_reference():
    p_t = np.array([0.6, 0.3, 0.1, 0.0])
    C = 1000 * p_t[:, None] * np.full((1, M), 0.25)
    m = metrics(C)
    assert abs(m["mod_norm_samp_marg"] - gc.norm(0.6, M)) < 1e-12
    expect = (0.3 ** 2 + 0.1 ** 2) / 0.4
    assert abs(m["dis_norm_samp_marg"] - gc.norm(expect, M)) < 1e-12
