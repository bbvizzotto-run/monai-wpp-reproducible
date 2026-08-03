from pathlib import Path

import pandas as pd

from monai_wpp.data.datasets import PairedRadiographDataset, RadiographClassificationDataset
from monai_wpp.data.images import image_metadata, load_grayscale_image
from monai_wpp.utils.synthetic import generate_synthetic_dataset


def test_synthetic_images_load_as_2d_grayscale(tmp_path: Path) -> None:
    manifest_path = generate_synthetic_dataset(tmp_path, n_cases=6)
    frame = pd.read_csv(manifest_path)
    image = load_grayscale_image(tmp_path / frame.loc[0, "original_path"])
    assert image.ndim == 2
    assert image.dtype.name == "float32"
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    metadata = image_metadata(tmp_path / frame.loc[0, "original_path"])
    assert metadata["width"] > 0 and metadata["height"] > 0


def test_condition_and_paired_datasets_preserve_case_identity(tmp_path: Path) -> None:
    manifest_path = generate_synthetic_dataset(tmp_path, n_cases=6)
    frame = pd.read_csv(manifest_path)
    original_ds = RadiographClassificationDataset(frame, condition="original", root=tmp_path)
    paired_ds = PairedRadiographDataset(frame, root=tmp_path)

    item = original_ds[0]
    pair = paired_ds[0]
    assert item["image"].shape[0] == 1
    assert pair["original"].shape[0] == pair["whatsapp"].shape[0] == 1
    assert item["case_id"] == pair["case_id"] == frame.loc[0, "case_id"]
    assert item["label"] == pair["label"]
