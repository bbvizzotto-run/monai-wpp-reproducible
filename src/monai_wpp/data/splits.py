"""Patient-grouped, stratified cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class SplitValidationError(ValueError):
    """Raised when leakage-free grouped folds cannot be created."""


def _score_assignment(
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    n_splits: int,
) -> tuple[int, float, float]:
    """Score a candidate assignment, prioritising class presence in every fold."""
    overall_counts = np.bincount(labels, minlength=2).astype(float)
    expected_class_counts = overall_counts / n_splits
    expected_fold_size = len(labels) / n_splits

    missing_class_cells = 0
    class_imbalance = 0.0
    size_imbalance = 0.0

    for fold in range(n_splits):
        fold_labels = labels[folds == fold]
        fold_counts = np.bincount(fold_labels, minlength=2).astype(float)
        missing_class_cells += int(fold_counts[0] == 0) + int(fold_counts[1] == 0)
        class_imbalance += float(
            np.sum(
                ((fold_counts - expected_class_counts) / np.maximum(expected_class_counts, 1.0))
                ** 2
            )
        )
        size_imbalance += float(
            ((len(fold_labels) - expected_fold_size) / max(expected_fold_size, 1.0)) ** 2
        )

    return missing_class_cells, class_imbalance, size_imbalance


def create_grouped_stratified_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    label_column: str = "label",
    group_column: str = "patient_id",
    max_attempts: int = 128,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a zero-based ``fold`` column.

    All rows sharing the same group remain in the same fold. Original and
    WhatsApp paths are columns in the same row, so paired observations cannot
    be separated by this operation.

    ``StratifiedGroupKFold`` is approximate when grouping constraints apply.
    Several deterministic candidate assignments are therefore evaluated and
    the best one is retained. A valid result must contain both binary classes
    in every fold, because fold-level AUC and related metrics are otherwise
    undefined.
    """
    required = {label_column, group_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SplitValidationError("Missing split columns: " + ", ".join(missing))

    if n_splits < 2:
        raise SplitValidationError("n_splits must be at least 2.")
    if max_attempts < 1:
        raise SplitValidationError("max_attempts must be at least 1.")

    unique_groups = frame[group_column].nunique(dropna=False)
    if unique_groups < n_splits:
        raise SplitValidationError(
            f"Cannot create {n_splits} folds from only {unique_groups} unique groups."
        )

    class_counts = frame[label_column].value_counts()
    observed_classes = set(class_counts.index.tolist())
    if observed_classes != {0, 1}:
        raise SplitValidationError("Labels must contain exactly the binary classes 0 and 1.")

    groups_per_class = (
        frame[[group_column, label_column]]
        .drop_duplicates()
        .groupby(label_column)[group_column]
        .nunique()
    )
    insufficient = {
        label: int(groups_per_class.get(label, 0))
        for label in (0, 1)
        if int(groups_per_class.get(label, 0)) < n_splits
    }
    if insufficient:
        details = ", ".join(
            f"class {label}: {count} groups" for label, count in insufficient.items()
        )
        raise SplitValidationError(
            f"Cannot place both classes in all {n_splits} folds ({details})."
        )

    labels = frame[label_column].to_numpy(dtype=int)
    groups = frame[group_column].to_numpy()
    dummy_features = np.zeros((len(frame), 1), dtype=np.float32)

    best_folds: np.ndarray | None = None
    best_score: tuple[int, float, float] | None = None

    for attempt in range(max_attempts):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + attempt,
        )
        candidate = np.full(len(frame), fill_value=-1, dtype=int)

        try:
            for fold, (_, validation_indices) in enumerate(
                splitter.split(dummy_features, labels, groups=groups)
            ):
                candidate[validation_indices] = fold
        except ValueError as exc:
            raise SplitValidationError(
                f"Could not create grouped stratified folds: {exc}"
            ) from exc

        if (candidate < 0).any():
            continue

        score = _score_assignment(labels, candidate, n_splits=n_splits)
        if best_score is None or score < best_score:
            best_score = score
            best_folds = candidate.copy()

        if score[0] == 0 and score[1] < 1e-12 and score[2] < 1e-12:
            break

    if best_folds is None or best_score is None:
        raise SplitValidationError("At least one case was not assigned to a fold.")
    if best_score[0] > 0:
        raise SplitValidationError(
            "Could not obtain a grouped split containing both classes in every fold. "
            "Reduce n_splits or review the patient-level class distribution."
        )

    result = frame.copy()
    result["fold"] = best_folds

    leakage = result.groupby(group_column)["fold"].nunique()
    if (leakage > 1).any():
        leaking_groups = leakage[leakage > 1].index.tolist()
        raise SplitValidationError(
            "Patient/group leakage detected for: " + ", ".join(map(str, leaking_groups[:5]))
        )

    return result
