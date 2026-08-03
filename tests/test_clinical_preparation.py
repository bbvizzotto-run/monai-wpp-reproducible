from pathlib import Path

import pandas as pd
import pytest

from monai_wpp.data.clinical import (
    ClinicalValidationError,
    create_patient_transmission_log,
    load_and_validate_clinical_cases,
    summarize_fold_assignment,
)
from monai_wpp.data.splits import create_grouped_stratified_folds


def _write_cases(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "cases.csv"
    frame.to_csv(path, index=False)
    return path


def test_bilateral_cases_may_share_one_source_path(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["c_left", "c_right"],
            "patient_id": ["p1", "p1"],
            "original_path": ["patient_1.jpg", "patient_1.jpg"],
            "label": [0, 1],
            "tooth": [38, 48],
            "side": ["left", "right"],
        }
    )

    loaded = load_and_validate_clinical_cases(
        _write_cases(tmp_path, frame),
        check_files=False,
    )

    assert len(loaded) == 2
    assert loaded["original_path"].nunique() == 1


def test_source_path_cannot_be_shared_between_patients(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "patient_id": ["p1", "p2"],
            "original_path": ["shared.jpg", "shared.jpg"],
            "label": [0, 1],
            "tooth": [38, 48],
            "side": ["left", "right"],
        }
    )

    with pytest.raises(ClinicalValidationError, match="Cross-patient duplicates"):
        load_and_validate_clinical_cases(
            _write_cases(tmp_path, frame),
            check_files=False,
        )


def test_tooth_side_mapping_is_validated(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["c1"],
            "patient_id": ["p1"],
            "original_path": ["patient_1.jpg"],
            "label": [1],
            "tooth": [38],
            "side": ["right"],
        }
    )

    with pytest.raises(ClinicalValidationError, match="Tooth/side mismatch"):
        load_and_validate_clinical_cases(
            _write_cases(tmp_path, frame),
            check_files=False,
        )


def test_grouped_clinical_folds_and_transmission_log() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(20)],
            "patient_id": [f"p{i}" for i in range(20)],
            "original_path": [f"patient_{i}.jpg" for i in range(20)],
            "label": [i % 2 for i in range(20)],
            "tooth": [38 if i % 2 == 0 else 48 for i in range(20)],
            "side": ["left" if i % 2 == 0 else "right" for i in range(20)],
        }
    )

    folded = create_grouped_stratified_folds(frame, n_splits=5, seed=42)
    summary = summarize_fold_assignment(folded)
    transmission = create_patient_transmission_log(folded)

    assert len(summary) == 5
    assert (summary["label_0"] == 2).all()
    assert (summary["label_1"] == 2).all()
    assert (folded.groupby("patient_id")["fold"].nunique() == 1).all()
    assert len(transmission) == 20
    assert transmission["patient_id"].is_unique
