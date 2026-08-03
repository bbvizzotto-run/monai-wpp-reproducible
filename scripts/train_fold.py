#!/usr/bin/env python3
"""Train one grouped fold on original images and evaluate both paired conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from monai_wpp.data.datasets import RadiographClassificationDataset
from monai_wpp.data.manifest import load_and_validate_manifest
from monai_wpp.data.transforms import build_classification_transforms
from monai_wpp.evaluation.metrics import calculate_binary_metrics
from monai_wpp.models import create_model
from monai_wpp.training import fit_binary_classifier, predict_binary_classifier, set_reproducibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--architecture", default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _resolve_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loader(
    frame: pd.DataFrame,
    *,
    condition: str,
    root: Path,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    from torch.utils.data import DataLoader

    dataset = RadiographClassificationDataset(
        frame,
        condition=condition,
        root=root,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def _condition_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    return calculate_binary_metrics(
        predictions["label"].to_numpy(),
        predictions["probability"].to_numpy(),
        threshold=threshold,
    )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")

    import torch

    set_reproducibility(args.seed)
    device = _resolve_device(args.device)
    print(f"Device: {device}")

    frame = load_and_validate_manifest(
        args.manifest,
        root=args.root,
        check_files=True,
    )
    if "fold" not in frame.columns:
        raise ValueError("Manifest must contain a 'fold' column created by create_folds.py.")
    available_folds = sorted(frame["fold"].astype(int).unique().tolist())
    if args.fold not in available_folds:
        raise ValueError(f"Fold {args.fold} not found. Available folds: {available_folds}")

    train_frame = frame[frame["fold"].astype(int) != args.fold].reset_index(drop=True)
    validation_frame = frame[frame["fold"].astype(int) == args.fold].reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Training and validation partitions must both be non-empty.")
    if validation_frame["label"].nunique() != 2:
        raise ValueError("Validation fold must contain both binary classes.")

    training_transform = build_classification_transforms(
        image_size=args.image_size,
        training=True,
    )
    evaluation_transform = build_classification_transforms(
        image_size=args.image_size,
        training=False,
    )

    train_loader = _build_loader(
        train_frame,
        condition="original",
        root=args.root,
        transform=training_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_original_loader = _build_loader(
        validation_frame,
        condition="original",
        root=args.root,
        transform=evaluation_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    validation_whatsapp_loader = _build_loader(
        validation_frame,
        condition="whatsapp",
        root=args.root,
        transform=evaluation_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_model(
        args.architecture,
        input_channels=1,
        num_classes=1,
        pretrained=False,
    )

    fold_dir = args.output_dir / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best_model.pt"
    history_path = fold_dir / "history.csv"

    fit_binary_classifier(
        model,
        train_loader,
        validation_original_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        threshold=args.threshold,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        metadata={
            "fold": int(args.fold),
            "architecture": args.architecture,
            "seed": int(args.seed),
            "image_size": list(args.image_size),
            "training_condition": "original",
            "selection_condition": "original",
        },
    )

    original_predictions = predict_binary_classifier(
        model,
        validation_original_loader,
        device,
    ).rename(columns={"probability": "probability_original"})
    whatsapp_predictions = predict_binary_classifier(
        model,
        validation_whatsapp_loader,
        device,
    ).rename(columns={"probability": "probability_whatsapp"})

    paired = original_predictions.merge(
        whatsapp_predictions[
            ["case_id", "patient_id", "label", "probability_whatsapp"]
        ],
        on=["case_id", "patient_id", "label"],
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(validation_frame):
        raise RuntimeError("Original and WhatsApp predictions did not form a complete pairing.")
    paired.insert(2, "fold", int(args.fold))
    paired.to_csv(fold_dir / "predictions.csv", index=False)

    original_for_metrics = paired.rename(
        columns={"probability_original": "probability"}
    )[["label", "probability"]]
    whatsapp_for_metrics = paired.rename(
        columns={"probability_whatsapp": "probability"}
    )[["label", "probability"]]
    metrics = {
        "fold": int(args.fold),
        "device": str(device),
        "architecture": args.architecture,
        "training_condition": "original",
        "model_selection_condition": "original",
        "original": _condition_metrics(original_for_metrics, args.threshold),
        "whatsapp": _condition_metrics(whatsapp_for_metrics, args.threshold),
    }
    (fold_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved paired predictions: {fold_dir / 'predictions.csv'}")
    print(json.dumps(metrics, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
