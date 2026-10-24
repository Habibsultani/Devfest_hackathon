"""Image-level metrics shared by final evaluation and controlled stress tests."""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def image_metrics(labels, scores, threshold: float) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Metrics require equally sized nonempty label and score arrays.")
    if (
        not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("Invalid metric inputs.")
    predictions = scores >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
        "image_auroc": float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2
        else None,
        "normal_images": tn + fp,
        "defect_images": tp + fn,
        "total_images": len(labels),
    }
