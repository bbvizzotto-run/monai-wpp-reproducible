#!/usr/bin/env python3
"""Run the prespecified ImageNet-pretrained ResNet-18 pilot on one outer fold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

from monai_wpp.data.transforms import build_classification_transforms as base_transforms
from monai_wpp.models import create_model as base_create_model


PILOT_CONFIG = {
    "architecture": "resnet18",
    "pretrained": True,
    "pretrained_source": "torchvision ImageNet DEFAULT weights",
    "grayscale_adaptation": "mean of RGB conv1 kernels",
    "image_size": [224, 224],
    "rotation_degrees": 7.0,
    "translation_fraction": 0.05,
    "horizontal_flip_probability": 0.0,
    "contrast_probability": 0.30,
    "intensity_shift_probability": 0.30,
    "epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 8,
    "threshold": 0.5,
    "inner_splits": 4,
    "inner_fold": 0,
    "seed": 42,
    "split_seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pilot_resnet18"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=PILOT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=PILOT_CONFIG["batch_size"])
    parser.add_argument(
        "--seed",
        type=int,
        default=PILOT_CONFIG["seed"],
        help="Training/augmentation seed. Use 42 for the primary run.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=PILOT_CONFIG["split_seed"],
        help="Seed that freezes the grouped inner validation partition.",
    )
    return parser.parse_args()


def load_train_fold_module():
    script_path = Path(__file__).resolve().with_name("train_fold.py")
    spec = importlib.util.spec_from_file_location("monai_wpp_train_fold", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pilot_create_model(
    architecture: str,
    *,
    input_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = False,
):
    """Force the frozen transfer-learning policy for the pilot."""
    if architecture.lower().replace("_", "-") != "resnet18":
        raise ValueError("The pilot script is frozen to ResNet-18.")
    return base_create_model(
        "resnet18",
        input_channels=input_channels,
        num_classes=num_classes,
        pretrained=True,
    )


def pilot_transforms(*, image_size=(224, 224), training: bool, **_ignored):
    """Apply the frozen conservative augmentation policy."""
    return base_transforms(
        image_size=image_size,
        training=training,
        rotation_degrees=PILOT_CONFIG["rotation_degrees"],
        translation_fraction=PILOT_CONFIG["translation_fraction"],
        horizontal_flip_probability=PILOT_CONFIG["horizontal_flip_probability"],
        contrast_probability=PILOT_CONFIG["contrast_probability"],
        intensity_shift_probability=PILOT_CONFIG["intensity_shift_probability"],
    )


def environment_metadata() -> dict[str, object]:
    import monai
    import torch
    import torchvision

    metadata: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "monai": monai.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_count"] = torch.cuda.device_count()
    return metadata


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")

    output_fold = args.output_dir / f"fold_{args.fold}"
    output_fold.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "pilot_config": {
            **PILOT_CONFIG,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
        },
        "environment": environment_metadata(),
        "outer_fold": int(args.fold),
    }
    (output_fold / "pilot_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    train_fold = load_train_fold_module()
    train_fold.create_model = pilot_create_model
    train_fold.build_classification_transforms = pilot_transforms

    original_argv = sys.argv
    try:
        sys.argv = [
            str(Path(__file__).resolve()),
            "--manifest",
            str(args.manifest),
            "--root",
            str(args.root),
            "--fold",
            str(args.fold),
            "--output-dir",
            str(args.output_dir),
            "--architecture",
            "resnet18",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(PILOT_CONFIG["learning_rate"]),
            "--weight-decay",
            str(PILOT_CONFIG["weight_decay"]),
            "--patience",
            str(PILOT_CONFIG["patience"]),
            "--threshold",
            str(PILOT_CONFIG["threshold"]),
            "--seed",
            str(args.seed),
            "--split-seed",
            str(args.split_seed),
            "--inner-splits",
            str(PILOT_CONFIG["inner_splits"]),
            "--inner-fold",
            str(PILOT_CONFIG["inner_fold"]),
            "--image-size",
            str(PILOT_CONFIG["image_size"][0]),
            str(PILOT_CONFIG["image_size"][1]),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]
        return int(train_fold.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
