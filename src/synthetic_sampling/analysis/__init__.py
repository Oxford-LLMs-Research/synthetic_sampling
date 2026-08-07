"""Analysis core: accuracy, AUC, prior correction, bootstrap, replicate."""

from .metrics import (
    auc,
    normalized_accuracy,
    question_averaged_norm_acc,
    weighted_auc,
)
from .prior_correct import prior_corrected
from .bootstrap import clustered_bootstrap_ci, question_mean
from .replicate import (
    CROSS_SERVING_FLOOR,
    pair_agreement,
    replicate_agreement,
    summarize_controls,
)

__all__ = [
    "auc",
    "normalized_accuracy",
    "question_averaged_norm_acc",
    "weighted_auc",
    "prior_corrected",
    "clustered_bootstrap_ci",
    "question_mean",
    "CROSS_SERVING_FLOOR",
    "pair_agreement",
    "replicate_agreement",
    "summarize_controls",
]
