"""Question / target-clustered bootstrap."""

from __future__ import annotations

import numpy as np


def question_mean(v: np.ndarray, clusters: np.ndarray) -> float:
    """Mean within cluster, then mean across clusters."""
    import pandas as pd
    return float(pd.Series(v).groupby(clusters).mean().mean())


def clustered_bootstrap_ci(
    v: np.ndarray,
    clusters: np.ndarray,
    n: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Question-clustered percentile CI of the cluster-mean of v."""
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(clusters, return_inverse=True)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    m = np.array([v[inv == i].mean() for i in range(len(keys))])
    draws = m[rng.integers(0, len(keys), size=(n, len(keys)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)
