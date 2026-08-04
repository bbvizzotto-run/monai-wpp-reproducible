"""Multi-seed paired analysis for original and WhatsApp radiograph predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import fleiss_kappa

from monai_wpp.evaluation.metrics import calculate_binary_metrics


CONDITIONS = ("original", "whatsapp")
BOOTSTRAP_METRICS = (
    "roc_auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "brier_score",
)
REQUIRED_PREDICTION_COLUMNS = {
    "case_id",
    "patient_id",
    "fold",
    "label",
    "probability_original",
    "probability_whatsapp",
}


@dataclass(frozen=True)
class SeedRun:
    """Validated outputs from one complete multi-fold model seed."""

    seed: int
    root: Path
    predictions: pd.DataFrame
    fold_summary: pd.DataFrame


def parse_run_spec(specification: str) -> tuple[int, Path]:
    """Parse a command-line run specification in the form ``SEED=PATH``."""
    seed_text, separator, path_text = specification.partition("=")
    if not separator or not seed_text.strip() or not path_text.strip():
        raise ValueError(f"Invalid run specification {specification!r}; expected SEED=PATH.")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise ValueError(f"Run seed must be an integer: {seed_text!r}.") from error
    return seed, Path(path_text).expanduser()


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file: {path}") from error


def _validate_probabilities(frame: pd.DataFrame, source: Path) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Prediction file is empty: {source}")
    if frame["case_id"].duplicated().any():
        raise ValueError(f"Duplicate case_id values found in {source}")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"Labels must be binary in {source}")
    for column in ("probability_original", "probability_whatsapp"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must contain finite values between 0 and 1 in {source}")


def load_seed_run(seed: int, root: Path, *, expected_folds: int = 5) -> SeedRun:
    """Load and validate all fold outputs for one training seed."""
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {resolved_root}")

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    seen_cases: set[str] = set()

    for fold in range(expected_folds):
        fold_dir = resolved_root / f"fold_{fold}"
        predictions_path = fold_dir / "predictions.csv"
        metrics_path = fold_dir / "metrics.json"
        history_path = fold_dir / "selection" / "history.csv"

        for required_path in (predictions_path, metrics_path, history_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Required result file not found: {required_path}")

        predictions = pd.read_csv(predictions_path)
        _validate_probabilities(predictions, predictions_path)
        predictions["fold"] = predictions["fold"].astype(int)
        if set(predictions["fold"].unique()) != {fold}:
            raise ValueError(f"{predictions_path} contains rows outside fold {fold}")

        duplicated_across_folds = seen_cases.intersection(predictions["case_id"].astype(str))
        if duplicated_across_folds:
            raise ValueError(
                f"Cases occur in more than one fold for seed {seed}: "
                f"{sorted(duplicated_across_folds)[:5]}"
            )
        seen_cases.update(predictions["case_id"].astype(str))
        prediction_frames.append(predictions)

        metrics = _read_json(metrics_path)
        if int(metrics.get("outer_fold", -1)) != fold:
            raise ValueError(f"outer_fold mismatch in {metrics_path}")
        history = pd.read_csv(history_path)
        if history.empty or "epoch" not in history or "improved" not in history:
            raise ValueError(f"Invalid selection history: {history_path}")
        improved = history[history["improved"].astype(bool)]
        if improved.empty:
            raise ValueError(f"Selection history has no improved epoch: {history_path}")
        best_row = improved.iloc[-1]

        fold_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "selected_epochs": int(metrics["selected_epochs"]),
                "selection_epochs_run": int(history["epoch"].max()),
                "best_inner_validation_auc": float(best_row["validation_roc_auc"]),
                "best_inner_validation_accuracy": float(best_row["validation_accuracy"]),
                "outer_auc_original": float(metrics["original"]["roc_auc"]),
                "outer_auc_whatsapp": float(metrics["whatsapp"]["roc_auc"]),
                "outer_accuracy_original": float(metrics["original"]["accuracy"]),
                "outer_accuracy_whatsapp": float(metrics["whatsapp"]["accuracy"]),
                "inner_split_seed": metrics.get("inner_split_seed"),
                "selection_seed": metrics.get("selection_seed"),
            }
        )

    combined = pd.concat(prediction_frames, ignore_index=True)
    combined["seed"] = seed
    return SeedRun(
        seed=seed,
        root=resolved_root,
        predictions=combined,
        fold_summary=pd.DataFrame(fold_rows),
    )


def align_seed_runs(runs: Iterable[SeedRun]) -> pd.DataFrame:
    """Verify fixed cases/folds across seeds and return one aligned wide table."""
    ordered_runs = sorted(runs, key=lambda run: run.seed)
    if len(ordered_runs) < 2:
        raise ValueError("At least two seed runs are required for stability analysis.")
    if len({run.seed for run in ordered_runs}) != len(ordered_runs):
        raise ValueError("Each seed must appear only once.")

    key_columns = ["case_id", "patient_id", "fold", "label"]
    reference = (
        ordered_runs[0].predictions[key_columns]
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    if reference["case_id"].duplicated().any():
        raise ValueError("Reference seed contains duplicate cases.")

    wide = reference.copy()
    for run in ordered_runs:
        keys = run.predictions[key_columns].sort_values("case_id").reset_index(drop=True)
        if not reference.equals(keys):
            raise ValueError(
                f"Seed {run.seed} does not have the same cases, patients, labels, and folds."
            )
        probability_frame = run.predictions[
            ["case_id", "probability_original", "probability_whatsapp"]
        ].rename(
            columns={
                "probability_original": f"original_seed_{run.seed}",
                "probability_whatsapp": f"whatsapp_seed_{run.seed}",
            }
        )
        wide = wide.merge(probability_frame, on="case_id", validate="one_to_one")
    return wide


def _exact_mcnemar(correct_original: np.ndarray, correct_whatsapp: np.ndarray) -> float:
    both_correct = int(np.sum(correct_original & correct_whatsapp))
    original_only = int(np.sum(correct_original & ~correct_whatsapp))
    whatsapp_only = int(np.sum(~correct_original & correct_whatsapp))
    both_wrong = int(np.sum(~correct_original & ~correct_whatsapp))
    return float(
        mcnemar(
            [[both_correct, original_only], [whatsapp_only, both_wrong]],
            exact=True,
        ).pvalue
    )


def summarize_seeds(runs: Iterable[SeedRun], *, threshold: float = 0.5) -> pd.DataFrame:
    """Calculate out-of-fold metrics and paired changes for each seed."""
    rows: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda item: item.seed):
        frame = run.predictions
        label = frame["label"].to_numpy(dtype=int)
        probability_original = frame["probability_original"].to_numpy(dtype=float)
        probability_whatsapp = frame["probability_whatsapp"].to_numpy(dtype=float)
        original = calculate_binary_metrics(label, probability_original, threshold=threshold)
        whatsapp = calculate_binary_metrics(label, probability_whatsapp, threshold=threshold)

        predicted_original = probability_original >= threshold
        predicted_whatsapp = probability_whatsapp >= threshold
        correct_original = predicted_original == label
        correct_whatsapp = predicted_whatsapp == label

        row: dict[str, object] = {"seed": run.seed}
        row.update({f"original_{key}": value for key, value in original.items()})
        row.update({f"whatsapp_{key}": value for key, value in whatsapp.items()})
        for metric in BOOTSTRAP_METRICS:
            row[f"delta_{metric}"] = float(whatsapp[metric]) - float(original[metric])
        row.update(
            {
                "classification_changed": int(np.sum(predicted_original != predicted_whatsapp)),
                "correct_to_wrong": int(np.sum(correct_original & ~correct_whatsapp)),
                "wrong_to_correct": int(np.sum(~correct_original & correct_whatsapp)),
                "mcnemar_exact_p": _exact_mcnemar(correct_original, correct_whatsapp),
                "original_whatsapp_pearson": float(
                    pearsonr(probability_original, probability_whatsapp).statistic
                ),
                "original_whatsapp_spearman": float(
                    spearmanr(probability_original, probability_whatsapp).statistic
                ),
                "mean_absolute_probability_change": float(
                    np.mean(np.abs(probability_whatsapp - probability_original))
                ),
                "median_absolute_probability_change": float(
                    np.median(np.abs(probability_whatsapp - probability_original))
                ),
                "max_absolute_probability_change": float(
                    np.max(np.abs(probability_whatsapp - probability_original))
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_seed_stability(wide: pd.DataFrame, seeds: Iterable[int]) -> pd.DataFrame:
    """Summarize probability agreement between every pair of training seeds."""
    ordered_seeds = sorted(seeds)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for position, seed_a in enumerate(ordered_seeds):
            for seed_b in ordered_seeds[position + 1 :]:
                first = wide[f"{condition}_seed_{seed_a}"].to_numpy(dtype=float)
                second = wide[f"{condition}_seed_{seed_b}"].to_numpy(dtype=float)
                rows.append(
                    {
                        "condition": condition,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "pearson": float(pearsonr(first, second).statistic),
                        "spearman": float(spearmanr(first, second).statistic),
                        "mean_absolute_difference": float(np.mean(np.abs(first - second))),
                        "median_absolute_difference": float(np.median(np.abs(first - second))),
                        "binary_agreement": float(np.mean((first >= 0.5) == (second >= 0.5))),
                    }
                )
    return pd.DataFrame(rows)


def seed_consensus(
    wide: pd.DataFrame,
    seeds: Iterable[int],
    *,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate cross-seed consensus and model-averaged case probabilities."""
    enriched = wide.copy()
    rows: list[dict[str, object]] = []
    ordered_seeds = sorted(seeds)
    label = enriched["label"].to_numpy(dtype=int)

    for condition in CONDITIONS:
        columns = [f"{condition}_seed_{seed}" for seed in ordered_seeds]
        values = enriched[columns].to_numpy(dtype=float)
        votes = values >= threshold
        unanimous = np.all(votes == votes[:, [0]], axis=1)
        count_table = np.column_stack([np.sum(~votes, axis=1), np.sum(votes, axis=1)])
        mean_probability = np.mean(values, axis=1)
        probability_sd = np.std(values, axis=1, ddof=1)
        averaged_metrics = calculate_binary_metrics(
            label,
            mean_probability,
            threshold=threshold,
        )

        enriched[f"{condition}_mean_probability"] = mean_probability
        enriched[f"{condition}_probability_sd"] = probability_sd
        rows.append(
            {
                "condition": condition,
                "unanimous_cases": int(np.sum(unanimous)),
                "unanimous_percentage": float(np.mean(unanimous)),
                "non_unanimous_cases": int(np.sum(~unanimous)),
                "fleiss_kappa": float(fleiss_kappa(count_table)),
                "mean_case_probability_sd": float(np.mean(probability_sd)),
                "median_case_probability_sd": float(np.median(probability_sd)),
                **{f"model_averaged_{key}": value for key, value in averaged_metrics.items()},
            }
        )
    return pd.DataFrame(rows), enriched


def _fast_auc(label: np.ndarray, probability: np.ndarray) -> float:
    positives = label == 1
    positive_count = int(np.sum(positives))
    negative_count = int(label.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = rankdata(probability, method="average")
    rank_sum = float(np.sum(ranks[positives]))
    return (
        rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def _fast_metrics(
    label: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    predicted = probability >= threshold
    positive = label == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & negative))

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    return {
        "roc_auc": float(_fast_auc(label, probability)),
        "accuracy": float(np.mean(predicted == positive)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "brier_score": float(np.mean((probability - label) ** 2)),
    }


def patient_cluster_bootstrap(
    wide: pd.DataFrame,
    seeds: Iterable[int],
    *,
    repetitions: int = 10_000,
    random_seed: int = 20_260_804,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate patient-clustered CIs and exploratory patient+seed intervals.

    A common patient resample is used for all seeds and both image conditions in
    each repetition, preserving the paired comparison. The exploratory table also
    resamples the finite set of observed seeds and must not be interpreted as a
    population-level random-effects analysis.
    """
    ordered_seeds = sorted(seeds)
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required.")

    patient_values = wide["patient_id"].astype(str).to_numpy()
    unique_patients = pd.unique(patient_values)
    patient_indices = {
        patient: np.flatnonzero(patient_values == patient) for patient in unique_patients
    }
    label_all = wide["label"].to_numpy(dtype=int)
    probabilities = {
        (seed, condition): wide[f"{condition}_seed_{seed}"].to_numpy(dtype=float)
        for seed in ordered_seeds
        for condition in CONDITIONS
    }

    rng = np.random.default_rng(random_seed)
    distributions: dict[tuple[object, str, str], np.ndarray] = {}
    for seed in ordered_seeds:
        for condition in CONDITIONS:
            for metric in BOOTSTRAP_METRICS:
                distributions[(seed, condition, metric)] = np.empty(repetitions)
        for metric in BOOTSTRAP_METRICS:
            distributions[(seed, "delta", metric)] = np.empty(repetitions)
    for condition in CONDITIONS:
        for metric in BOOTSTRAP_METRICS:
            distributions[("three_seed_mean", condition, metric)] = np.empty(repetitions)
    for metric in BOOTSTRAP_METRICS:
        distributions[("three_seed_mean", "delta", metric)] = np.empty(repetitions)
        distributions[("patient_and_seed", "delta", metric)] = np.empty(repetitions)

    for repetition in range(repetitions):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_indices = np.concatenate([patient_indices[patient] for patient in sampled_patients])
        label = label_all[sampled_indices]
        seed_metrics: dict[tuple[int, str], dict[str, float]] = {}

        for seed in ordered_seeds:
            for condition in CONDITIONS:
                metrics = _fast_metrics(
                    label,
                    probabilities[(seed, condition)][sampled_indices],
                    threshold=threshold,
                )
                seed_metrics[(seed, condition)] = metrics
                for metric, value in metrics.items():
                    distributions[(seed, condition, metric)][repetition] = value
            for metric in BOOTSTRAP_METRICS:
                delta = (
                    seed_metrics[(seed, "whatsapp")][metric]
                    - seed_metrics[(seed, "original")][metric]
                )
                distributions[(seed, "delta", metric)][repetition] = delta

        for metric in BOOTSTRAP_METRICS:
            original_mean = float(
                np.mean([seed_metrics[(seed, "original")][metric] for seed in ordered_seeds])
            )
            whatsapp_mean = float(
                np.mean([seed_metrics[(seed, "whatsapp")][metric] for seed in ordered_seeds])
            )
            distributions[("three_seed_mean", "original", metric)][repetition] = original_mean
            distributions[("three_seed_mean", "whatsapp", metric)][repetition] = whatsapp_mean
            distributions[("three_seed_mean", "delta", metric)][repetition] = (
                whatsapp_mean - original_mean
            )

            sampled_seeds = rng.choice(ordered_seeds, size=len(ordered_seeds), replace=True)
            distributions[("patient_and_seed", "delta", metric)][repetition] = float(
                np.mean(
                    [
                        seed_metrics[(seed, "whatsapp")][metric]
                        - seed_metrics[(seed, "original")][metric]
                        for seed in sampled_seeds
                    ]
                )
            )

    def interval(values: np.ndarray) -> tuple[float, float]:
        low, high = np.nanpercentile(values, [2.5, 97.5])
        return float(low), float(high)

    point_metrics: dict[tuple[int, str], dict[str, float]] = {}
    for seed in ordered_seeds:
        for condition in CONDITIONS:
            point_metrics[(seed, condition)] = _fast_metrics(
                label_all,
                probabilities[(seed, condition)],
                threshold=threshold,
            )

    ci_rows: list[dict[str, object]] = []
    for seed in ordered_seeds:
        for condition in CONDITIONS:
            for metric in BOOTSTRAP_METRICS:
                low, high = interval(distributions[(seed, condition, metric)])
                ci_rows.append(
                    {
                        "scope": f"seed_{seed}",
                        "condition": condition,
                        "metric": metric,
                        "estimate": point_metrics[(seed, condition)][metric],
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        for metric in BOOTSTRAP_METRICS:
            low, high = interval(distributions[(seed, "delta", metric)])
            ci_rows.append(
                {
                    "scope": f"seed_{seed}",
                    "condition": "whatsapp_minus_original",
                    "metric": metric,
                    "estimate": (
                        point_metrics[(seed, "whatsapp")][metric]
                        - point_metrics[(seed, "original")][metric]
                    ),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    for condition in CONDITIONS:
        for metric in BOOTSTRAP_METRICS:
            low, high = interval(distributions[("three_seed_mean", condition, metric)])
            ci_rows.append(
                {
                    "scope": "seed_mean",
                    "condition": condition,
                    "metric": metric,
                    "estimate": float(
                        np.mean(
                            [point_metrics[(seed, condition)][metric] for seed in ordered_seeds]
                        )
                    ),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    for metric in BOOTSTRAP_METRICS:
        low, high = interval(distributions[("three_seed_mean", "delta", metric)])
        ci_rows.append(
            {
                "scope": "seed_mean",
                "condition": "whatsapp_minus_original",
                "metric": metric,
                "estimate": float(
                    np.mean(
                        [
                            point_metrics[(seed, "whatsapp")][metric]
                            - point_metrics[(seed, "original")][metric]
                            for seed in ordered_seeds
                        ]
                    )
                ),
                "ci_low": low,
                "ci_high": high,
            }
        )

    ci_frame = pd.DataFrame(ci_rows)
    ci_frame["bootstrap_unit"] = "patient"
    ci_frame["bootstrap_repetitions"] = repetitions
    ci_frame["bootstrap_random_seed"] = random_seed

    exploratory_rows: list[dict[str, object]] = []
    for metric in BOOTSTRAP_METRICS:
        low, high = interval(distributions[("patient_and_seed", "delta", metric)])
        exploratory_rows.append(
            {
                "metric": metric,
                "estimate": float(
                    np.mean(
                        [
                            point_metrics[(seed, "whatsapp")][metric]
                            - point_metrics[(seed, "original")][metric]
                            for seed in ordered_seeds
                        ]
                    )
                ),
                "ci_low": low,
                "ci_high": high,
                "resampled_units": "patients and observed seeds",
                "bootstrap_repetitions": repetitions,
                "bootstrap_random_seed": random_seed,
                "interpretation_note": (
                    "Exploratory because the finite set of observed seeds is small."
                ),
            }
        )
    return ci_frame, pd.DataFrame(exploratory_rows)
