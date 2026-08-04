from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monai_wpp.evaluation.seed_stability import (
    align_seed_runs,
    load_seed_run,
    parse_run_spec,
    patient_cluster_bootstrap,
    summarize_seeds,
)


def _write_seed(root: Path, seed: int, probabilities: list[tuple[float, float]]) -> Path:
    run_root = root / f"seed_{seed}"
    case_number = 0
    for fold in range(2):
        fold_dir = run_root / f"fold_{fold}"
        selection_dir = fold_dir / "selection"
        selection_dir.mkdir(parents=True)
        rows = []
        for local_index in range(4):
            label = local_index % 2
            original, whatsapp = probabilities[case_number]
            rows.append(
                {
                    "case_id": f"case_{case_number}",
                    "patient_id": f"patient_{case_number // 2}",
                    "fold": fold,
                    "label": label,
                    "probability_original": original,
                    "probability_whatsapp": whatsapp,
                }
            )
            case_number += 1
        pd.DataFrame(rows).to_csv(fold_dir / "predictions.csv", index=False)
        (fold_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "outer_fold": fold,
                    "selected_epochs": fold + 2,
                    "original": {"roc_auc": 0.8, "accuracy": 0.75},
                    "whatsapp": {"roc_auc": 0.8, "accuracy": 0.75},
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "epoch": 1,
                    "validation_roc_auc": 0.7,
                    "validation_accuracy": 0.5,
                    "improved": True,
                },
                {
                    "epoch": 2,
                    "validation_roc_auc": 0.6,
                    "validation_accuracy": 0.5,
                    "improved": False,
                },
            ]
        ).to_csv(selection_dir / "history.csv", index=False)
    return run_root


def test_parse_run_spec() -> None:
    seed, path = parse_run_spec("42=outputs/seed42")
    assert seed == 42
    assert path == Path("outputs/seed42")
    with pytest.raises(ValueError):
        parse_run_spec("missing-separator")


def test_alignment_summary_and_bootstrap(tmp_path: Path) -> None:
    probabilities_42 = [
        (0.1, 0.1), (0.8, 0.8), (0.2, 0.25), (0.7, 0.72),
        (0.3, 0.3), (0.9, 0.88), (0.4, 0.45), (0.6, 0.62),
    ]
    probabilities_43 = [
        (0.15, 0.14), (0.85, 0.84), (0.25, 0.27), (0.75, 0.76),
        (0.35, 0.34), (0.95, 0.93), (0.45, 0.47), (0.65, 0.66),
    ]
    run_42 = load_seed_run(42, _write_seed(tmp_path, 42, probabilities_42), expected_folds=2)
    run_43 = load_seed_run(43, _write_seed(tmp_path, 43, probabilities_43), expected_folds=2)

    wide = align_seed_runs([run_42, run_43])
    assert len(wide) == 8
    summary = summarize_seeds([run_42, run_43])
    assert set(summary["seed"]) == {42, 43}
    assert (summary["delta_roc_auc"].abs() < 1e-12).all()

    confidence_intervals, exploratory = patient_cluster_bootstrap(
        wide,
        [42, 43],
        repetitions=100,
        random_seed=123,
    )
    assert not confidence_intervals.empty
    assert not exploratory.empty
    assert {
        "scope",
        "condition",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
    }.issubset(confidence_intervals.columns)


def test_alignment_rejects_different_labels(tmp_path: Path) -> None:
    probabilities = [(0.1, 0.1), (0.8, 0.8)] * 4
    run_42 = load_seed_run(42, _write_seed(tmp_path, 42, probabilities), expected_folds=2)
    second_root = _write_seed(tmp_path, 43, probabilities)
    prediction_path = second_root / "fold_1" / "predictions.csv"
    frame = pd.read_csv(prediction_path)
    frame.loc[0, "label"] = 1 - int(frame.loc[0, "label"])
    frame.to_csv(prediction_path, index=False)
    run_43 = load_seed_run(43, second_root, expected_folds=2)

    with pytest.raises(ValueError, match="same cases"):
        align_seed_runs([run_42, run_43])
