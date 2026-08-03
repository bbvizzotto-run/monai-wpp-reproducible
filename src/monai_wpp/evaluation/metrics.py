"""Binary diagnostic-performance metrics derived from case-level probabilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    y_true: np.ndarray | list[int],
    y_probability: np.ndarray | list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate case-level binary metrics using a fixed classification threshold."""
    true = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_probability, dtype=float)

    if true.ndim != 1 or probability.ndim != 1:
        raise ValueError("y_true and y_probability must be one-dimensional.")
    if len(true) != len(probability) or len(true) == 0:
        raise ValueError("Inputs must have the same non-zero length.")
    if not np.isin(true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("y_probability must contain finite values between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    auc = roc_auc_score(true, probability) if len(np.unique(true)) == 2 else float("nan")

    return {
        "n": int(len(true)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(true, predicted)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "roc_auc": float(auc),
        "brier_score": float(brier_score_loss(true, probability)),
    }
