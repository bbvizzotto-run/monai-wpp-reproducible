"""Patient-grouped, stratified cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class SplitValidationError(ValueError):
    """Raised when leakage-free grouped folds cannot be created."""


def create_grouped_stratified_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    label_column: str = "label",
    group_column: str = "patient_id",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a zero-based ``fold`` column.

    All rows sharing the same group remain in the same fold. Original and
    WhatsApp paths are columns in the same row, so the paired observations
    cannot be separated by this operation.
    """
    required = {label_column, group_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SplitValidationError("Missing split columns: " + ", ".join(missing))

    if n_splits < 2:
        raise SplitValidationError("n_splits must be at least 2.")

    unique_groups = frame[group_column].nunique(dropna=False)
    if unique_groups < n_splits:
        raise SplitValidationError(
            f"Cannot create {n_splits} folds from only {unique_groups} unique groups."
        )

    class_counts = frame[label_column].value_counts()
    missing_classes = {0, 1}.difference(class_counts.index)
    if missing_classes:
        raise SplitValidationError("Both binary classes must be present before splitting.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(frame), fill_value=-1, dtype=int)
    dummy_features = np.zeros((len(frame), 1), dtype=np.float32)

    try:
        for fold, (_, validation_indices) in enumerate(
            splitter.split(
                dummy_features,
                frame[label_column].to_numpy(),
                groups=frame[group_column].to_numpy(),
            )
        ):
            folds[validation_indices] = fold
    except ValueError as exc:
        raise SplitValidationError(f"Could not create grouped stratified folds: {exc}") from exc

    if (folds < 0).any():
        raise SplitValidationError("At least one case was not assigned to a fold.")

    result = frame.copy()
    result["fold"] = folds

    leakage = result.groupby(group_column)["fold"].nunique()
    if (leakage > 1).any():
        leaking_groups = leakage[leakage > 1].index.tolist()
        raise SplitValidationError(
            "Patient/group leakage detected for: " + ", ".join(map(str, leaking_groups[:5]))
        )

    return result
