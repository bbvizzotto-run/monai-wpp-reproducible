#!/usr/bin/env python3
"""Create deterministic patient-grouped stratified cross-validation folds."""

from __future__ import annotations

import argparse
from pathlib import Path

from monai_wpp.data.manifest import ManifestValidationError, load_and_validate_manifest
from monai_wpp.data.splits import SplitValidationError, create_grouped_stratified_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        frame = load_and_validate_manifest(
            args.manifest,
            root=args.root,
            check_files=not args.skip_file_check,
        )
        folded = create_grouped_stratified_folds(
            frame,
            n_splits=args.n_splits,
            seed=args.seed,
        )
    except (ManifestValidationError, SplitValidationError) as exc:
        print(f"Fold creation failed: {exc}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    folded.to_csv(args.output, index=False)
    print(f"Saved {len(folded)} cases to {args.output}")
    print("Cases by fold and class:")
    print(folded.groupby(["fold", "label"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
