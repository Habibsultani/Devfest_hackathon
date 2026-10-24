import numpy as np
import pytest

from scripts.train_model import calibrate_threshold, balanced_threshold
from src.evaluation import image_metrics


def test_metrics_include_missed_defects_and_false_alarms():
    result = image_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5)
    assert [result[k] for k in ("TP", "FP", "TN", "FN")] == [1, 1, 1, 1]
    assert result["recall"] == result["precision"] == result["f1"] == result["false_positive_rate"] == 0.5
    assert result["image_auroc"] == 0.75


def test_undefined_metrics_are_explicit_nulls():
    result = image_metrics([0, 0], [0.1, 0.2], 1)
    assert result["recall"] is None
    assert result["precision"] is None
    assert result["image_auroc"] is None
    assert result["false_positive_rate"] == 0


def test_validation_only_calibration_handles_threshold_equality():
    threshold = calibrate_threshold([1.0, 2.0], 0.95)
    assert threshold > 2.0
    assert not np.array([1.0, 2.0]).__ge__(threshold).any()
    assert image_metrics([1], [threshold], threshold)["TP"] == 1
    with pytest.raises(ValueError):
        calibrate_threshold([float("nan")])


def test_balanced_threshold_uses_both_labels():
    threshold, metrics = balanced_threshold([0, 0, 1, 1], [1, 2, 3, 4])
    assert threshold == 3
    assert metrics["balanced_accuracy"] == 1
    assert metrics["false_positive_rate"] == 0
    with pytest.raises(ValueError, match="both labels"):
        balanced_threshold([0, 0], [1, 2])
