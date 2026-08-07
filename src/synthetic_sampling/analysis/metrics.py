"""Core metrics: normalized accuracy and discrimination AUC."""

from __future__ import annotations

import numpy as np


def normalized_accuracy(acc: float, n_options: int) -> float:
    """(acc - 1/M) / (1 - 1/M) with M = stated number of options."""
    if n_options < 2:
        return float("nan")
    chance = 1.0 / n_options
    return (acc - chance) / (1.0 - chance)


def question_averaged_norm_acc(
    correct: np.ndarray,
    n_options: np.ndarray,
    clusters: np.ndarray,
) -> float:
    """Mean within-question normalized accuracy, then mean across questions."""
    import pandas as pd

    raw = correct.astype(float)
    norm = np.array([
        normalized_accuracy(a, int(m)) for a, m in zip(raw, n_options)
    ])
    # Prefer question-level mean of instance-level norms when M is constant
    # within question; otherwise average instance norms within question.
    s = pd.Series(norm).groupby(clusters).mean()
    return float(s.mean())


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC of scores vs binary labels."""
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def weighted_auc(cent: np.ndarray, truth: np.ndarray, opts: list) -> float:
    """Option-weighted mean of one-vs-rest AUCs on column-centred scores."""
    a, w = [], []
    for i, o in enumerate(opts):
        y = (truth == o).astype(int)
        v = auc(cent[:, i], y)
        if not np.isnan(v):
            a.append(v)
            w.append(int(y.sum()))
    return float(np.average(a, weights=w)) if a else float("nan")
