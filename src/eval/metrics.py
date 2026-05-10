"""Classification metrics for the commit-risk-scorer.

Used both directly (post-training evaluation) and as the `compute_metrics` callback
for HuggingFace Trainer. The same metric definitions back the regression-gated CI
described in docs/design-doc.md §Eval Methodology.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Container for binary classification metrics."""

    f1: float
    precision: float
    recall: float
    accuracy: float
    auc_roc: float | None  # None when probability scores are not available

    def to_dict(self) -> dict[str, float | None]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute F1, precision, recall, accuracy, and (optionally) AUC-ROC.

    Args:
        y_true: ground-truth labels (0 or 1)
        y_pred: predicted labels (0 or 1)
        y_proba: predicted probabilities for the positive class (for AUC-ROC).
    """
    return ClassificationMetrics(
        f1=float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        precision=float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc_roc=(float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None),
    )


def hf_compute_metrics(eval_pred) -> dict[str, float | None]:
    """HuggingFace Trainer-compatible compute_metrics callback.

    Receives an EvalPrediction-like (logits, labels) tuple and returns a dict of
    metrics that the Trainer logs and uses for early-stopping / regression gates.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # softmax for the positive-class probability — used for AUC-ROC.
    # numerically stable softmax:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    proba = exp[:, 1] / exp.sum(axis=-1)

    return compute_metrics(labels, preds, proba).to_dict()
