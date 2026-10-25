from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """Tính ma trận nhầm lẫn, hàng là nhãn thật và cột là nhãn dự đoán."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải có cùng kích thước")
    return sk_confusion_matrix(y_true, y_pred, labels=labels)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> dict:
    """Tính accuracy, precision, recall, F1 macro và ma trận nhầm lẫn."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": sk_confusion_matrix(y_true, y_pred, labels=labels),
    }