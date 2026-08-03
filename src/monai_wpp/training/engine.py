"""Training and evaluation loops for binary image classification."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monai_wpp.evaluation.metrics import calculate_binary_metrics


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for training. Install with: pip install -e '.[training]'"
        ) from exc
    return torch


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and MONAI when available."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)

    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        from monai.utils import set_determinism

        set_determinism(seed=seed)
    except ImportError:
        pass


def _prepare_labels(batch: dict[str, Any], device: Any):
    torch = _require_torch()
    return torch.as_tensor(batch["label"], dtype=torch.float32, device=device).reshape(-1, 1)


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_function: Any,
    device: Any,
) -> float:
    """Train for one epoch and return the sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device=device, dtype=_require_torch().float32)
        labels = _prepare_labels(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader produced no samples.")
    return total_loss / total_samples


def predict_binary_classifier(model: Any, loader: Any, device: Any) -> pd.DataFrame:
    """Return case-level probabilities for one deterministic data loader."""
    torch = _require_torch()
    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            probabilities = torch.sigmoid(logits).detach().cpu().reshape(-1).numpy()
            labels = torch.as_tensor(batch["label"]).detach().cpu().reshape(-1).numpy()

            case_ids = list(batch["case_id"])
            patient_ids = list(batch["patient_id"])
            if not (
                len(case_ids)
                == len(patient_ids)
                == len(labels)
                == len(probabilities)
            ):
                raise ValueError("Batch metadata and predictions have inconsistent lengths.")

            for case_id, patient_id, label, probability in zip(
                case_ids,
                patient_ids,
                labels,
                probabilities,
                strict=True,
            ):
                rows.append(
                    {
                        "case_id": str(case_id),
                        "patient_id": str(patient_id),
                        "label": int(label),
                        "probability": float(probability),
                    }
                )

    if not rows:
        raise ValueError("Prediction loader produced no samples.")
    result = pd.DataFrame(rows)
    if result["case_id"].duplicated().any():
        raise ValueError("Prediction loader returned duplicate case_id values.")
    return result


def evaluate_binary_classifier(
    model: Any,
    loader: Any,
    loss_function: Any,
    device: Any,
    *,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float | int], pd.DataFrame]:
    """Evaluate one loader and return loss, metrics, and case-level predictions."""
    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = _prepare_labels(batch, device)
            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            loss = loss_function(logits, labels)
            probabilities = torch.sigmoid(logits).detach().cpu().reshape(-1).numpy()
            labels_cpu = labels.detach().cpu().reshape(-1).numpy().astype(int)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size

            case_ids = list(batch["case_id"])
            patient_ids = list(batch["patient_id"])
            for case_id, patient_id, label, probability in zip(
                case_ids,
                patient_ids,
                labels_cpu,
                probabilities,
                strict=True,
            ):
                rows.append(
                    {
                        "case_id": str(case_id),
                        "patient_id": str(patient_id),
                        "label": int(label),
                        "probability": float(probability),
                    }
                )

    if total_samples == 0:
        raise ValueError("Evaluation loader produced no samples.")

    predictions = pd.DataFrame(rows)
    metrics = calculate_binary_metrics(
        predictions["label"].to_numpy(),
        predictions["probability"].to_numpy(),
        threshold=threshold,
    )
    return total_loss / total_samples, metrics, predictions


def fit_binary_classifier(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    *,
    device: Any,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
    patience: int = 10,
    threshold: float = 0.5,
    checkpoint_path: str | Path,
    history_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train with AdamW and early stopping based on validation ROC AUC.

    The checkpoint contains only tensors and simple metadata. Model selection is
    performed on the original-image validation loader supplied by the caller.
    """
    torch = _require_torch()

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_auc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )
        validation_loss, validation_metrics, _ = evaluate_binary_classifier(
            model,
            validation_loader,
            loss_function,
            device,
            threshold=threshold,
        )
        validation_auc = float(validation_metrics["roc_auc"])
        if not np.isfinite(validation_auc):
            raise ValueError(
                "Validation ROC AUC is undefined. Every validation fold must contain both classes."
            )

        improved = validation_auc > best_auc + 1e-12
        if improved:
            best_auc = validation_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_roc_auc": validation_auc,
                "metadata": metadata or {},
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1

        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "validation_roc_auc": validation_auc,
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_sensitivity": float(validation_metrics["sensitivity"]),
            "validation_specificity": float(validation_metrics["specificity"]),
            "improved": improved,
        }
        history.append(row)

        if history_path is not None:
            pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"val_auc={validation_auc:.4f} "
            f"best_epoch={best_epoch:03d}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("Training ended without creating a checkpoint.")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    history_frame = pd.DataFrame(history)
    summary_path = checkpoint_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "best_epoch": int(best_epoch),
                "best_validation_roc_auc": float(best_auc),
                "epochs_completed": int(len(history_frame)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return history_frame
