#!/usr/bin/env python3
"""Prepare leakage-free folds and private workflow templates for clinical cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from monai_wpp.data.clinical import (
    ClinicalValidationError,
    create_patient_transmission_log,
    load_and_validate_clinical_cases,
    summarize_fold_assignment,
)
from monai_wpp.data.splits import SplitValidationError, create_grouped_stratified_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--patient-index-output", type=Path, default=None)
    parser.add_argument("--whatsapp-log-output", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        frame = load_and_validate_clinical_cases(
            args.cases,
            root=args.root,
            check_files=not args.skip_file_check,
        )
        folded = create_grouped_stratified_folds(
            frame,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        summary = summarize_fold_assignment(folded)
        transmission_log = create_patient_transmission_log(folded)
    except (ClinicalValidationError, SplitValidationError) as exc:
        print(f"Clinical preparation failed: {exc}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    folded.to_csv(args.output, index=False)

    summary_output = args.summary_output or args.output.with_name(
        f"{args.output.stem}.summary.csv"
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)

    if args.patient_index_output is not None:
        patient_index = (
            folded.groupby("patient_id", sort=True)
            .agg(
                fold=("fold", "first"),
                n_cases=("case_id", "size"),
                original_path=("original_path", "first"),
                n_labels=("label", "nunique"),
            )
            .reset_index()
        )
        patient_index["bilateral"] = patient_index["n_cases"].gt(1)
        patient_index["mixed_labels"] = patient_index["n_labels"].gt(1)
        args.patient_index_output.parent.mkdir(parents=True, exist_ok=True)
        patient_index.to_csv(args.patient_index_output, index=False)

    if args.whatsapp_log_output is not None:
        args.whatsapp_log_output.parent.mkdir(parents=True, exist_ok=True)
        transmission_log.to_csv(args.whatsapp_log_output, index=False)

    print(f"Saved {len(folded)} cases to {args.output}")
    print(f"Saved fold summary to {summary_output}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
