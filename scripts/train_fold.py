#!/usr/bin/env python3
"""Train one outer fold and evaluate paired original/WhatsApp held-out cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from monai_wpp.data.datasets import RadiographClassificationDataset
from monai_wpp.data.manifest import load_and_validate_manifest
from monai_wpp.data.splits import create_grouped_stratified_folds
from monai_wpp.data.transforms import build_classification_transforms
from monai_wpp.evaluation.metrics import calculate_binary_metrics
from monai_wpp.models import create_model
from monai_wpp.training import (
    fit_binary_classifier,
    predict_binary_classifier,
    set_reproducibility,
    train_one_epoch,
)


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
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Optional seed used only to create the grouped inner split. "
            "When omitted, --seed is used for both splitting and training."
        ),
    )
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--inner-fold", type=int, default=0)
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


def _json_ready(value):
    import math

    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if args.inner_fold < 0 or args.inner_fold >= args.inner_splits:
        raise ValueError("inner-fold must be between 0 and inner-splits - 1.")

    import torch

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

    outer_train = frame[frame["fold"].astype(int) != args.fold].reset_index(drop=True)
    outer_test = frame[frame["fold"].astype(int) == args.fold].reset_index(drop=True)
    if outer_train.empty or outer_test.empty:
        raise ValueError("Outer training and test partitions must both be non-empty.")
    if outer_test["label"].nunique() != 2:
        raise ValueError("Outer test fold must contain both binary classes.")

    # Keep the grouped inner partition fixed independently from model randomness
    # when --split-seed is supplied. This is useful for seed-stability analyses.
    split_base_seed = args.seed if args.split_seed is None else args.split_seed
    inner_split_seed = split_base_seed + (args.fold * 1000)
    selection_seed = args.seed + (args.fold * 1000)

    inner_assigned = create_grouped_stratified_folds(
        outer_train,
        n_splits=args.inner_splits,
        seed=inner_split_seed,
    )
    inner_train = inner_assigned[
        inner_assigned["fold"].astype(int) != args.inner_fold
    ].reset_index(drop=True)
    inner_validation = inner_assigned[
        inner_assigned["fold"].astype(int) == args.inner_fold
    ].reset_index(drop=True)

    fold_dir = args.output_dir / f"fold_{args.fold}"
    selection_dir = fold_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: select the epoch count using only the outer-training partition.
    set_reproducibility(selection_seed)
    selection_train_transform = build_classification_transforms(
        image_size=args.image_size,
        training=True,
    )
    evaluation_transform = build_classification_transforms(
        image_size=args.image_size,
        training=False,
    )
    selection_train_loader = _build_loader(
        inner_train,
        condition="original",
        root=args.root,
        transform=selection_train_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    selection_validation_loader = _build_loader(
        inner_validation,
        condition="original",
        root=args.root,
        transform=evaluation_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    selection_model = create_model(
        args.architecture,
        input_channels=1,
        num_classes=1,
        pretrained=False,
    )
    selection_history = fit_binary_classifier(
        selection_model,
        selection_train_loader,
        selection_validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        threshold=args.threshold,
        checkpoint_path=selection_dir / "best_model.pt",
        history_path=selection_dir / "history.csv",
        metadata={
            "outer_fold": int(args.fold),
            "inner_fold": int(args.inner_fold),
            "inner_splits": int(args.inner_splits),
            "architecture": args.architecture,
            "training_seed": int(selection_seed),
            "inner_split_seed": int(inner_split_seed),
            "image_size": list(args.image_size),
            "selection_condition": "original",
        },
    )
    improved_rows = selection_history[selection_history["improved"].astype(bool)]
    if improved_rows.empty:
        raise RuntimeError("Model selection did not record a best epoch.")
    best_epoch = int(improved_rows.iloc[-1]["epoch"])
    print(f"Selected epoch count: {best_epoch}")

    # Stage 2: reinitialize and train on the complete outer-training partition.
    refit_seed = selection_seed + 500_000
    set_reproducibility(refit_seed)
    refit_transform = build_classification_transforms(
        image_size=args.image_size,
        training=True,
    )
    refit_loader = _build_loader(
        outer_train,
        condition="original",
        root=args.root,
        transform=refit_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    final_model = create_model(
        args.architecture,
        input_channels=1,
        num_classes=1,
        pretrained=False,
    ).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    refit_rows: list[dict[str, float | int]] = []
    for epoch in range(1, best_epoch + 1):
        train_loss = train_one_epoch(
            final_model,
            refit_loader,
            optimizer,
            loss_function,
            device,
        )
        refit_rows.append({"epoch": epoch, "train_loss": float(train_loss)})
        print(f"refit_epoch={epoch:03d} train_loss={train_loss:.6f}")
    pd.DataFrame(refit_rows).to_csv(fold_dir / "refit_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "epochs": int(best_epoch),
            "metadata": {
                "outer_fold": int(args.fold),
                "architecture": args.architecture,
                "training_seed": int(refit_seed),
                "inner_split_seed": int(inner_split_seed),
                "image_size": list(args.image_size),
                "training_condition": "original",
                "outer_test_used_for_selection": False,
            },
        },
        fold_dir / "best_model.pt",
    )

    # Stage 3: paired evaluation on the untouched outer test fold.
    outer_original_loader = _build_loader(
        outer_test,
        condition="original",
        root=args.root,
        transform=evaluation_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    outer_whatsapp_loader = _build_loader(
        outer_test,
        condition="whatsapp",
        root=args.root,
        transform=evaluation_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    original_predictions = predict_binary_classifier(
        final_model,
        outer_original_loader,
        device,
    ).rename(columns={"probability": "probability_original"})
    whatsapp_predictions = predict_binary_classifier(
        final_model,
        outer_whatsapp_loader,
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
    if len(paired) != len(outer_test):
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
        "outer_fold": int(args.fold),
        "device": str(device),
        "architecture": args.architecture,
        "training_condition": "original",
        "model_selection_condition": "inner_original",
        "outer_test_used_for_selection": False,
        "base_training_seed": int(args.seed),
        "selection_seed": int(selection_seed),
        "refit_seed": int(refit_seed),
        "inner_split_base_seed": int(split_base_seed),
        "inner_split_seed": int(inner_split_seed),
        "selected_epochs": int(best_epoch),
        "outer_train_cases": int(len(outer_train)),
        "outer_test_cases": int(len(outer_test)),
        "inner_train_cases": int(len(inner_train)),
        "inner_validation_cases": int(len(inner_validation)),
        "original": _condition_metrics(original_for_metrics, args.threshold),
        "whatsapp": _condition_metrics(whatsapp_for_metrics, args.threshold),
    }
    metrics = _json_ready(metrics)
    (fold_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Saved final checkpoint: {fold_dir / 'best_model.pt'}")
    print(f"Saved paired predictions: {fold_dir / 'predictions.csv'}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
