"""Prior-corrected accuracy: base rates +/- model scores, cross-fitted."""

from __future__ import annotations

import numpy as np

from .metrics import normalized_accuracy

ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def prior_corrected(
    arr: np.ndarray,
    y: np.ndarray,
    opts: list,
    folds: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Cross-fitted (base rates only, base rates + model) normalized accuracy.

    Returns (prior_only_norm_acc, prior_plus_model_norm_acc).
    """
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
            ll = (s[np.arange(tr.sum()), yi[tr]]
                  - np.log(np.exp(s).sum(axis=1))).mean()
            if ll > best_ll:
                best_ll, best = ll, a
        prior_hit.append((np.full(te.sum(), logp.argmax()) == yi[te]).mean())
        corr_hit.append(
            ((best * cent[te] + logp).argmax(axis=1) == yi[te]).mean())
    m = len(set(y))
    return (
        normalized_accuracy(float(np.mean(prior_hit)), m),
        normalized_accuracy(float(np.mean(corr_hit)), m),
    )
