"""Case-level datasets for condition-specific training and paired evaluation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from monai_wpp.data.images import load_grayscale_image

Condition = Literal["original", "whatsapp"]


def _as_channel_first(image):
    return image[None, ...]


class RadiographClassificationDataset:
    """Return one image condition and its case-level binary label."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        condition: Condition,
        root: str | Path | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.condition = condition
        self.root = Path(root) if root is not None else Path(".")
        self.transform = transform
        self.path_column = f"{condition}_path"
        if self.path_column not in self.frame.columns:
            raise ValueError(f"Missing manifest column: {self.path_column}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = self.root / str(row[self.path_column])
        item: dict[str, Any] = {
            "image": _as_channel_first(load_grayscale_image(path)),
            "label": int(row["label"]),
            "case_id": str(row["case_id"]),
            "patient_id": str(row["patient_id"]),
            "condition": self.condition,
            "path": str(path),
        }
        return self.transform(item) if self.transform is not None else item


class PairedRadiographDataset:
    """Return original and WhatsApp images for the same case.

    This dataset is intended for paired inference and quality-control checks.
    Random augmentation must not be applied here because it would destroy the
    paired comparison.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        root: str | Path | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.root = Path(root) if root is not None else Path(".")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        original_path = self.root / str(row["original_path"])
        whatsapp_path = self.root / str(row["whatsapp_path"])
        item: dict[str, Any] = {
            "original": _as_channel_first(load_grayscale_image(original_path)),
            "whatsapp": _as_channel_first(load_grayscale_image(whatsapp_path)),
            "label": int(row["label"]),
            "case_id": str(row["case_id"]),
            "patient_id": str(row["patient_id"]),
            "original_path": str(original_path),
            "whatsapp_path": str(whatsapp_path),
        }
        return self.transform(item) if self.transform is not None else item
