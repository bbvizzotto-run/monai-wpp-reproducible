"""Similarity-based candidate generation for manually pairing unnamed images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from monai_wpp.data.images import SUPPORTED_EXTENSIONS, load_grayscale_image


def _resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def difference_hash(array: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Return a simple dHash bit vector robust to moderate JPEG compression."""
    small = _resize(array, (hash_size + 1, hash_size))
    return (small[:, 1:] >= small[:, :-1]).reshape(-1)


def similarity_score(first: np.ndarray, second: np.ndarray) -> float:
    """Combine perceptual-hash agreement and normalized grayscale correlation."""
    first_hash = difference_hash(first)
    second_hash = difference_hash(second)
    hash_similarity = 1.0 - np.mean(first_hash != second_hash)

    first_small = _resize(first, (128, 64)).reshape(-1)
    second_small = _resize(second, (128, 64)).reshape(-1)
    first_centered = first_small - first_small.mean()
    second_centered = second_small - second_small.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    correlation = float(np.dot(first_centered, second_centered) / denominator) if denominator else 0.0
    correlation = (correlation + 1.0) / 2.0
    return float(0.55 * hash_similarity + 0.45 * correlation)


def list_image_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Directory not found: {root}")
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def suggest_pair_candidates(
    original_dir: str | Path,
    whatsapp_dir: str | Path,
    *,
    top_k: int = 3,
) -> pd.DataFrame:
    """Rank WhatsApp candidates for every original image.

    The output is only a review aid. A human must confirm every final pairing.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    originals = list_image_files(original_dir)
    whatsapp = list_image_files(whatsapp_dir)
    if not originals or not whatsapp:
        raise ValueError("Both directories must contain supported image files.")

    original_arrays = {path: load_grayscale_image(path) for path in originals}
    whatsapp_arrays = {path: load_grayscale_image(path) for path in whatsapp}
    rows = []
    for original_path, original_array in original_arrays.items():
        ranking = sorted(
            (
                (similarity_score(original_array, candidate_array), candidate_path)
                for candidate_path, candidate_array in whatsapp_arrays.items()
            ),
            reverse=True,
            key=lambda item: item[0],
        )[: min(top_k, len(whatsapp))]
        for rank, (score, candidate_path) in enumerate(ranking, start=1):
            rows.append(
                {
                    "original_file": str(original_path),
                    "candidate_rank": rank,
                    "whatsapp_candidate": str(candidate_path),
                    "similarity": score,
                    "status": "review",
                }
            )
    return pd.DataFrame(rows)
