"""Validation and preparation utilities for original-only clinical case metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CLINICAL_REQUIRED_COLUMNS = (
    "case_id",
    "patient_id",
    "original_path",
    "label",
    "tooth",
    "side",
)


class ClinicalValidationError(ValueError):
    """Raised when original-only clinical metadata violate the expected schema."""


def _format_examples(values: list[object], limit: int = 5) -> str:
    return ", ".join(str(value) for value in values[:limit])


def load_and_validate_clinical_cases(
    cases_path: str | Path,
    *,
    root: str | Path | None = None,
    check_files: bool = True,
) -> pd.DataFrame:
    """Load one row per tooth-level clinical case before WhatsApp pairing.

    Duplicate ``original_path`` values are allowed only when they belong to the
    same patient, which supports bilateral teeth extracted from one panoramic
    radiograph. The same source path may never be shared by different patients.
    """
    cases_path = Path(cases_path)
    if not cases_path.is_file():
        raise ClinicalValidationError(f"Clinical cases file not found: {cases_path}")

    try:
        frame = pd.read_csv(cases_path)
    except Exception as exc:
        raise ClinicalValidationError(f"Could not read clinical cases file: {exc}") from exc

    missing = [column for column in CLINICAL_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ClinicalValidationError(
            "Missing required clinical columns: " + ", ".join(missing)
        )
    if frame.empty:
        raise ClinicalValidationError("Clinical cases file contains no rows.")

    frame = frame.copy()
    for column in ("case_id", "patient_id", "original_path", "side"):
        frame[column] = frame[column].astype("string").str.strip()
        empty = frame[column].isna() | frame[column].eq("")
        if empty.any():
            rows = (frame.index[empty] + 2).tolist()
            raise ClinicalValidationError(
                f"Column '{column}' contains empty values at CSV rows: "
                + _format_examples(rows)
            )

    duplicate_cases = frame.loc[
        frame["case_id"].duplicated(keep=False), "case_id"
    ].unique().tolist()
    if duplicate_cases:
        raise ClinicalValidationError(
            "case_id must be unique. Duplicates: " + _format_examples(duplicate_cases)
        )

    try:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    except Exception as exc:
        raise ClinicalValidationError("label must contain integer values 0 or 1.") from exc

    invalid_labels = sorted(
        set(frame.loc[~frame["label"].isin([0, 1]), "label"].tolist())
    )
    if invalid_labels:
        raise ClinicalValidationError(
            "label must contain only 0 and 1. Invalid values: "
            + _format_examples(invalid_labels)
        )

    try:
        frame["tooth"] = pd.to_numeric(frame["tooth"], errors="raise").astype(int)
    except Exception as exc:
        raise ClinicalValidationError("tooth must contain integer values 38 or 48.") from exc

    invalid_teeth = sorted(
        set(frame.loc[~frame["tooth"].isin([38, 48]), "tooth"].tolist())
    )
    if invalid_teeth:
        raise ClinicalValidationError(
            "tooth must contain only 38 and 48. Invalid values: "
            + _format_examples(invalid_teeth)
        )

    frame["side"] = frame["side"].str.lower()
    invalid_side = ~frame["side"].isin(["left", "right"])
    if invalid_side.any():
        values = frame.loc[invalid_side, "side"].unique().tolist()
        raise ClinicalValidationError(
            "side must be 'left' or 'right'. Invalid values: "
            + _format_examples(values)
        )

    mismatched = (
        ((frame["tooth"] == 38) & (frame["side"] != "left"))
        | ((frame["tooth"] == 48) & (frame["side"] != "right"))
    )
    if mismatched.any():
        examples = frame.loc[mismatched, "case_id"].tolist()
        raise ClinicalValidationError(
            "Tooth/side mismatch detected. Tooth 38 must be left and tooth 48 "
            "must be right. Cases: "
            + _format_examples(examples)
        )

    path_patient_counts = frame.groupby("original_path")["patient_id"].nunique()
    shared_between_patients = path_patient_counts[path_patient_counts > 1].index.tolist()
    if shared_between_patients:
        raise ClinicalValidationError(
            "An original_path may be reused only by cases from the same patient. "
            "Cross-patient duplicates: "
            + _format_examples(shared_between_patients)
        )

    if "fold" in frame.columns:
        try:
            frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        except Exception as exc:
            raise ClinicalValidationError("fold must contain integer values.") from exc
        if (frame["fold"] < 0).any():
            raise ClinicalValidationError("fold values must be zero or greater.")
        leakage = frame.groupby("patient_id")["fold"].nunique()
        if (leakage > 1).any():
            leaking = leakage[leakage > 1].index.tolist()
            raise ClinicalValidationError(
                "Patient leakage detected across folds: " + _format_examples(leaking)
            )

    if check_files:
        base = Path(root) if root is not None else cases_path.parent
        missing_files: list[str] = []
        for relative_path in frame["original_path"]:
            candidate = Path(relative_path)
            resolved = candidate if candidate.is_absolute() else base / candidate
            if not resolved.is_file():
                missing_files.append(str(resolved))
        if missing_files:
            raise ClinicalValidationError(
                f"{len(missing_files)} original image file(s) were not found. Examples: "
                + _format_examples(missing_files)
            )

    return frame


def summarize_fold_assignment(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize case, patient, class, tooth, and bilateral counts by fold."""
    if "fold" not in frame.columns:
        raise ClinicalValidationError("Cannot summarize folds without a 'fold' column.")

    leakage = frame.groupby("patient_id")["fold"].nunique()
    if (leakage > 1).any():
        leaking = leakage[leakage > 1].index.tolist()
        raise ClinicalValidationError(
            "Patient leakage detected across folds: " + _format_examples(leaking)
        )

    records: list[dict[str, int]] = []
    for fold, subset in frame.groupby("fold", sort=True):
        patient_case_counts = subset.groupby("patient_id").size()
        labels = subset["label"].value_counts()
        teeth = subset["tooth"].value_counts()
        mixed_labels = subset.groupby("patient_id")["label"].nunique().gt(1).sum()

        records.append(
            {
                "fold": int(fold),
                "cases": int(len(subset)),
                "patients": int(subset["patient_id"].nunique()),
                "label_0": int(labels.get(0, 0)),
                "label_1": int(labels.get(1, 0)),
                "tooth_38": int(teeth.get(38, 0)),
                "tooth_48": int(teeth.get(48, 0)),
                "bilateral_patients": int(patient_case_counts.gt(1).sum()),
                "mixed_label_patients": int(mixed_labels),
            }
        )

    summary = pd.DataFrame.from_records(records)
    if (summary[["label_0", "label_1"]] == 0).any(axis=None):
        raise ClinicalValidationError("Each retained fold must contain both binary classes.")
    return summary


def create_patient_transmission_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one WhatsApp transmission-log row per unique patient radiograph."""
    rows: list[dict[str, object]] = []
    grouped = frame.groupby("patient_id", sort=True)

    for send_order, (patient_id, subset) in enumerate(grouped, start=1):
        original_paths = subset["original_path"].drop_duplicates().tolist()
        if len(original_paths) != 1:
            raise ClinicalValidationError(
                f"Patient {patient_id} has {len(original_paths)} original paths; "
                "one complete panoramic radiograph is required per patient."
            )

        folds = subset["fold"].drop_duplicates().tolist() if "fold" in subset else []
        if len(folds) > 1:
            raise ClinicalValidationError(
                f"Patient leakage detected while creating transmission log: {patient_id}"
            )

        rows.append(
            {
                "send_order": send_order,
                "patient_id": patient_id,
                "fold": int(folds[0]) if folds else "",
                "original_path": original_paths[0],
                "whatsapp_received_filename": "",
                "whatsapp_full_path": "",
                "send_mode": "photo",
                "quality_mode": "standard",
                "sender_device": "",
                "sender_os": "",
                "receiver_device": "",
                "receiver_os": "",
                "whatsapp_version_sender": "",
                "whatsapp_version_receiver": "",
                "send_datetime": "",
                "received_width": "",
                "received_height": "",
                "received_size_bytes": "",
                "received_sha256": "",
                "pair_status": "pending",
                "notes": "",
            }
        )

    return pd.DataFrame.from_records(rows)
