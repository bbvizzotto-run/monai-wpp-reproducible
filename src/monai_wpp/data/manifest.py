"""Validation utilities for privacy-preserving paired-image manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = (
    "case_id",
    "patient_id",
    "original_path",
    "whatsapp_path",
    "label",
)


class ManifestValidationError(ValueError):
    """Raised when a dataset manifest violates the expected schema."""


def _format_examples(values: Iterable[object], limit: int = 5) -> str:
    items = [str(value) for value in list(values)[:limit]]
    return ", ".join(items)


def load_and_validate_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
    check_files: bool = True,
) -> pd.DataFrame:
    """Load and validate a paired original/WhatsApp image manifest.

    Parameters
    ----------
    manifest_path:
        CSV file containing one row per independent classification case.
    root:
        Base directory used to resolve relative image paths. Defaults to the
        manifest's parent directory.
    check_files:
        Verify that both image files exist for every row.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ManifestValidationError(f"Manifest not found: {manifest_path}")

    try:
        frame = pd.read_csv(manifest_path)
    except Exception as exc:  # pandas exposes several parser-specific errors
        raise ManifestValidationError(f"Could not read manifest: {exc}") from exc

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ManifestValidationError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if frame.empty:
        raise ManifestValidationError("Manifest contains no cases.")

    frame = frame.copy()
    for column in ("case_id", "patient_id", "original_path", "whatsapp_path"):
        frame[column] = frame[column].astype("string").str.strip()
        empty_mask = frame[column].isna() | frame[column].eq("")
        if empty_mask.any():
            rows = (frame.index[empty_mask] + 2).tolist()
            raise ManifestValidationError(
                f"Column '{column}' contains empty values at CSV rows: "
                f"{_format_examples(rows)}"
            )

    duplicate_cases = frame.loc[frame["case_id"].duplicated(keep=False), "case_id"].unique()
    if len(duplicate_cases):
        raise ManifestValidationError(
            "case_id must be unique. Duplicates: " + _format_examples(duplicate_cases)
        )

    try:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    except Exception as exc:
        raise ManifestValidationError("label must contain integer values 0 or 1.") from exc

    invalid_labels = sorted(set(frame.loc[~frame["label"].isin([0, 1]), "label"].tolist()))
    if invalid_labels:
        raise ManifestValidationError(
            "label must contain only 0 and 1. Invalid values: "
            + _format_examples(invalid_labels)
        )

    for column in ("original_path", "whatsapp_path"):
        duplicate_paths = frame.loc[frame[column].duplicated(keep=False), column].unique()
        if len(duplicate_paths):
            raise ManifestValidationError(
                f"Each '{column}' value must identify one case. Duplicates: "
                f"{_format_examples(duplicate_paths)}"
            )

    if "side" in frame.columns:
        frame["side"] = frame["side"].astype("string").str.strip().str.lower()
        invalid_side = frame["side"].notna() & ~frame["side"].isin(["left", "right", ""])
        if invalid_side.any():
            values = frame.loc[invalid_side, "side"].unique()
            raise ManifestValidationError(
                "side must be 'left', 'right', or empty. Invalid values: "
                + _format_examples(values)
            )

    if check_files:
        base = Path(root) if root is not None else manifest_path.parent
        missing_files: list[str] = []
        for column in ("original_path", "whatsapp_path"):
            for relative_path in frame[column]:
                candidate = Path(relative_path)
                resolved = candidate if candidate.is_absolute() else base / candidate
                if not resolved.is_file():
                    missing_files.append(str(resolved))
        if missing_files:
            raise ManifestValidationError(
                f"{len(missing_files)} image file(s) were not found. Examples: "
                + _format_examples(missing_files)
            )

    return frame
