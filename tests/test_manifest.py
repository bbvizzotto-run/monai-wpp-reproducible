from pathlib import Path

import pandas as pd
import pytest

from monai_wpp.data.manifest import ManifestValidationError, load_and_validate_manifest


def _write_manifest(path: Path) -> None:
    pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "patient_id": ["p1", "p2"],
            "original_path": ["o1.jpg", "o2.jpg"],
            "whatsapp_path": ["w1.jpg", "w2.jpg"],
            "label": [0, 1],
        }
    ).to_csv(path, index=False)


def test_manifest_schema_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    frame = load_and_validate_manifest(manifest, check_files=False)
    assert frame["label"].tolist() == [0, 1]


def test_duplicate_case_id_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    frame = pd.read_csv(manifest)
    frame.loc[1, "case_id"] = "c1"
    frame.to_csv(manifest, index=False)

    with pytest.raises(ManifestValidationError, match="case_id must be unique"):
        load_and_validate_manifest(manifest, check_files=False)
