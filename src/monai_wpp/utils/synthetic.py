"""Generate a small non-clinical paired dataset for end-to-end smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def _synthetic_radiograph(seed: int, label: int, size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(seed)
    width, height = size
    base = np.tile(np.linspace(25, 195, width, dtype=np.float32), (height, 1))
    base += rng.normal(0, 7, size=(height, width))
    base = np.clip(base, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="L")
    draw = ImageDraw.Draw(image)
    # Seed-specific non-clinical structures make each pair uniquely matchable.
    for _ in range(7):
        x = int(rng.integers(10, max(11, width - 30)))
        y = int(rng.integers(10, max(11, height - 30)))
        radius_x = int(rng.integers(5, 18))
        radius_y = int(rng.integers(4, 14))
        tone = int(rng.integers(45, 225))
        draw.ellipse((x, y, x + radius_x, y + radius_y), outline=tone, width=2)
    for _ in range(4):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        x2 = int(rng.integers(0, width))
        y2 = int(rng.integers(0, height))
        draw.line((x1, y1, x2, y2), fill=int(rng.integers(55, 205)), width=2)

    # Non-clinical geometric proxy: class 1 has two structures closer together.
    offset = 8 if label else 28
    center_x, center_y = width // 2, height // 2
    draw.ellipse((center_x - 36, center_y - 24, center_x + 36, center_y + 24), outline=235, width=5)
    draw.line((center_x - 60, center_y + offset, center_x + 60, center_y + offset), fill=45, width=7)
    return image.filter(ImageFilter.GaussianBlur(radius=1.0))


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    n_cases: int = 20,
    seed: int = 42,
    size: tuple[int, int] = (384, 192),
) -> Path:
    """Create paired JPEG files and return the generated manifest path."""
    if n_cases < 4:
        raise ValueError("n_cases must be at least 4.")
    root = Path(output_dir)
    original_dir = root / "original"
    whatsapp_dir = root / "whatsapp"
    original_dir.mkdir(parents=True, exist_ok=True)
    whatsapp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index in range(n_cases):
        label = index % 2
        case_id = f"synthetic_{index + 1:03d}"
        original_name = f"original_{index + 1:03d}.jpg"
        whatsapp_name = f"WA_{(index * 7 + 3) % n_cases + 1:04d}.jpg"
        original = _synthetic_radiograph(seed + index, label, size)
        original.save(original_dir / original_name, quality=98, subsampling=0)
        # Downscale and recompress to mimic a messaging-derived counterpart.
        compressed = original.resize((size[0] * 3 // 4, size[1] * 3 // 4), Image.Resampling.LANCZOS)
        compressed.save(whatsapp_dir / whatsapp_name, quality=55, optimize=True)
        records.append(
            {
                "case_id": case_id,
                "patient_id": f"patient_{index + 1:03d}",
                "original_path": str(Path("original") / original_name),
                "whatsapp_path": str(Path("whatsapp") / whatsapp_name),
                "label": label,
                "side": "right" if index % 2 else "left",
            }
        )

    manifest = root / "metadata.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest
