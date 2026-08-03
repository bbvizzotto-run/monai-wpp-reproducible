#!/usr/bin/env python3
"""Generate non-clinical paired images for local smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from monai_wpp.utils.synthetic import generate_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate_synthetic_dataset(args.output, n_cases=args.n_cases, seed=args.seed)
    print(f"Synthetic manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
