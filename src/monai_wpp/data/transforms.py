"""MONAI transforms for 2-D grayscale classification."""

from __future__ import annotations

from typing import Sequence


def build_classification_transforms(
    *,
    image_size: Sequence[int] = (224, 224),
    training: bool,
    rotation_degrees: float = 15.0,
    translation_fraction: float = 0.10,
    horizontal_flip_probability: float = 0.5,
    contrast_probability: float = 0.30,
    intensity_shift_probability: float = 0.30,
):
    """Build transforms that operate on ``image`` only, never on scalar labels.

    The import is local so manifest validation and statistical analysis remain
    usable without installing the optional deep-learning dependencies.
    """
    try:
        from monai.transforms import (
            Compose,
            EnsureTyped,
            RandAdjustContrastd,
            RandAffined,
            RandFlipd,
            RandShiftIntensityd,
            Resized,
            ScaleIntensityd,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "MONAI is required for training transforms. Install with: pip install -e '.[training]'"
        ) from exc

    height, width = (int(image_size[0]), int(image_size[1]))
    transforms = [
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        Resized(keys="image", spatial_size=(height, width), mode="bilinear"),
    ]
    if training:
        transforms.extend(
            [
                RandAffined(
                    keys="image",
                    prob=0.7,
                    rotate_range=(rotation_degrees * 3.141592653589793 / 180.0,),
                    translate_range=(height * translation_fraction, width * translation_fraction),
                    mode="bilinear",
                    padding_mode="border",
                ),
                RandFlipd(
                    keys="image", prob=horizontal_flip_probability, spatial_axis=1
                ),
                RandAdjustContrastd(keys="image", prob=contrast_probability, gamma=(0.8, 1.2)),
                RandShiftIntensityd(
                    keys="image", prob=intensity_shift_probability, offsets=0.10
                ),
            ]
        )
    transforms.append(EnsureTyped(keys=("image", "label"), track_meta=False))
    return Compose(transforms)


def build_paired_evaluation_transforms(*, image_size: Sequence[int] = (224, 224)):
    """Apply identical deterministic preprocessing to both paired conditions."""
    try:
        from monai.transforms import Compose, EnsureTyped, Resized, ScaleIntensityd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MONAI is required for paired transforms. Install with: pip install -e '.[training]'"
        ) from exc

    height, width = (int(image_size[0]), int(image_size[1]))
    return Compose(
        [
            ScaleIntensityd(keys=("original", "whatsapp"), minv=0.0, maxv=1.0),
            Resized(
                keys=("original", "whatsapp"),
                spatial_size=(height, width),
                mode=("bilinear", "bilinear"),
            ),
            EnsureTyped(keys=("original", "whatsapp", "label"), track_meta=False),
        ]
    )
