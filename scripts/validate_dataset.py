#!/usr/bin/env python3
"""Validate a paired-image manifest before any experiment is run."""

from __future__ import annotations

import argparse
from pathlib import Path

from monai_wpp.data.manifest import ManifestValidationError, load_and_validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Validate schema only; do not require image files to exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        frame = load_and_validate_manifest(
            args.manifest,
            root=args.root,
            check_files=not args.skip_file_check,
        )
    except ManifestValidationError as exc:
        print(f"Manifest validation failed: {exc}")
        return 1

    print("Manifest validation passed.")
    print(f"Cases: {len(frame)}")
    print(f"Patients/groups: {frame['patient_id'].nunique()}")
    print("Class distribution:")
    print(frame["label"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
