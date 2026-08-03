import pandas as pd
import pytest

from monai_wpp.data.splits import SplitValidationError, create_grouped_stratified_folds


def test_grouped_folds_prevent_patient_leakage() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(12)],
            "patient_id": [f"p{i // 2}" for i in range(12)],
            "label": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "original_path": [f"o{i}.jpg" for i in range(12)],
            "whatsapp_path": [f"w{i}.jpg" for i in range(12)],
        }
    )
    result = create_grouped_stratified_folds(frame, n_splits=3, seed=7)

    assert sorted(result["fold"].unique().tolist()) == [0, 1, 2]
    assert (result.groupby("patient_id")["fold"].nunique() == 1).all()
    assert (result.groupby("fold")["label"].nunique() == 2).all()


def test_balanced_independent_cases_are_balanced_across_five_folds() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(20)],
            "patient_id": [f"p{i}" for i in range(20)],
            "label": [i % 2 for i in range(20)],
            "original_path": [f"o{i}.jpg" for i in range(20)],
            "whatsapp_path": [f"w{i}.jpg" for i in range(20)],
        }
    )

    result = create_grouped_stratified_folds(frame, n_splits=5, seed=42)
    counts = result.groupby(["fold", "label"]).size().unstack(fill_value=0)

    assert counts.to_dict() == {0: {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}, 1: {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}}


def test_split_rejects_insufficient_patient_groups_per_class() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(10)],
            "patient_id": ["positive_patient"] * 5 + [f"negative_{i}" for i in range(5)],
            "label": [1] * 5 + [0] * 5,
            "original_path": [f"o{i}.jpg" for i in range(10)],
            "whatsapp_path": [f"w{i}.jpg" for i in range(10)],
        }
    )

    with pytest.raises(SplitValidationError, match="class 1: 1 groups"):
        create_grouped_stratified_folds(frame, n_splits=5, seed=42)
