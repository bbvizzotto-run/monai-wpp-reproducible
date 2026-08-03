"""Safe loading and normalization of two-dimensional radiographs."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS: Final[set[str]] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ImageLoadingError(ValueError):
    """Raised when an image cannot be loaded as a finite 2-D grayscale array."""


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load a conventional image file as a float32 array in ``[0, 1]``.

    EXIF orientation is applied before conversion. DICOM is intentionally not
    handled here; it will be added only if the definitive dataset requires it.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise ImageLoadingError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadingError(
            f"Unsupported image extension '{image_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            array = np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageLoadingError(f"Could not load image '{image_path}': {exc}") from exc

    if array.ndim != 2 or min(array.shape) < 2:
        raise ImageLoadingError(f"Expected a non-empty 2-D image, got shape {array.shape}.")
    array /= 255.0
    if not np.isfinite(array).all():
        raise ImageLoadingError(f"Image contains non-finite values: {image_path}")
    return array


def image_metadata(path: str | Path) -> dict[str, int | str]:
    """Return basic non-identifying technical metadata for one image file."""
    image_path = Path(path)
    array = load_grayscale_image(image_path)
    return {
        "path": str(image_path),
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "size_bytes": int(image_path.stat().st_size),
        "extension": image_path.suffix.lower(),
    }
