from __future__ import annotations

from eval import calculate_metrics


def test_confusion_matrix_and_metrics() -> None:
    metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["true_positive"] == 2
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["accuracy"] == 0.75

