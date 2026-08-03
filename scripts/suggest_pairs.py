#!/usr/bin/env python3
"""Create a review spreadsheet with likely original/WhatsApp image pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from monai_wpp.data.matching import suggest_pair_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--whatsapp-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/pair_candidates.csv"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    candidates = suggest_pair_candidates(
        args.original_dir, args.whatsapp_dir, top_k=args.top_k
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    print(f"Saved {len(candidates)} candidate rows to {args.output}")
    print("Review each match manually before creating the definitive manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
