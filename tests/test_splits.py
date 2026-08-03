import pandas as pd

from monai_wpp.data.splits import create_grouped_stratified_folds


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
