import pytest

from monai_wpp.evaluation.metrics import calculate_binary_metrics


def test_binary_metrics_are_derived_from_case_level_probabilities() -> None:
    metrics = calculate_binary_metrics(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        threshold=0.5,
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
