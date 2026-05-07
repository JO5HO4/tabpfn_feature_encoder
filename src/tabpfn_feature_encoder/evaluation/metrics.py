from __future__ import annotations

from typing import Any

import numpy as np

from tabpfn_feature_encoder.data.base import to_numpy_matrix


def accuracy(y_true: Any, y_pred: Any) -> float:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise ValueError("y_true and y_pred length mismatch.")
    return float(np.mean(y_true_np == y_pred_np))


def log_loss(y_true: Any, proba: Any, eps: float = 1e-15) -> float:
    y_true_np = np.asarray(y_true, dtype=np.int64).reshape(-1)
    proba_np = np.asarray(proba, dtype=np.float64)
    if proba_np.ndim != 2:
        raise ValueError("proba must be a 2D array.")
    if len(y_true_np) != len(proba_np):
        raise ValueError("y_true and proba length mismatch.")
    proba_np = np.clip(proba_np, eps, 1.0 - eps)
    proba_np = proba_np / proba_np.sum(axis=1, keepdims=True)
    classes = np.arange(proba_np.shape[1])
    class_to_col = {int(label): idx for idx, label in enumerate(classes)}
    try:
        cols = np.asarray([class_to_col[int(label)] for label in y_true_np], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("y_true contains a class outside the probability columns.") from exc
    return float(-np.mean(np.log(proba_np[np.arange(len(y_true_np)), cols])))


def binary_roc_auc(y_true: Any, scores: Any) -> float:
    y_true_np = np.asarray(y_true).reshape(-1)
    scores_np = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y_true_np) != len(scores_np):
        raise ValueError("y_true and scores length mismatch.")
    labels = np.unique(y_true_np)
    if len(labels) != 2:
        raise ValueError("binary_roc_auc requires exactly two classes.")
    positive = labels[-1]
    is_pos = y_true_np == positive
    n_pos = int(np.count_nonzero(is_pos))
    n_neg = int(len(y_true_np) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC requires positive and negative examples.")
    ranks = _average_ranks(scores_np)
    pos_rank_sum = float(np.sum(ranks[is_pos]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_auc(y_true: Any, proba: Any) -> float:
    y_true_np = np.asarray(y_true).reshape(-1)
    proba_np = np.asarray(proba, dtype=np.float64)
    if proba_np.ndim != 2:
        raise ValueError("proba must be a 2D array.")
    labels = np.unique(y_true_np)
    if len(labels) == 2:
        if proba_np.shape[1] < 2:
            raise ValueError("Binary ROC AUC requires probabilities for at least two classes.")
        return binary_roc_auc(y_true_np, proba_np[:, 1])

    aucs = []
    for class_label in labels:
        class_idx = int(class_label)
        if class_idx >= proba_np.shape[1]:
            raise ValueError("Class label is outside probability columns.")
        y_binary = (y_true_np == class_label).astype(np.int64)
        aucs.append(binary_roc_auc(y_binary, proba_np[:, class_idx]))
    return float(np.mean(aucs))


def evaluate_classifier(model: Any, X: Any, y: Any, metrics: list[str]) -> dict[str, float]:
    X_np = to_numpy_matrix(X)
    y_np = np.asarray(y)
    out: dict[str, float] = {}
    proba = None
    pred = None
    if any(metric in {"roc_auc", "log_loss"} for metric in metrics):
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not provide predict_proba.")
        proba = np.asarray(model.predict_proba(X_np))
    if "accuracy" in metrics:
        if hasattr(model, "predict"):
            pred = np.asarray(model.predict(X_np))
        elif proba is not None:
            pred = np.argmax(proba, axis=1)
        else:
            raise ValueError("Model does not provide predict or predict_proba.")

    for metric in metrics:
        if metric == "accuracy":
            out[metric] = accuracy(y_np, pred)
        elif metric == "roc_auc":
            out[metric] = roc_auc(y_np, proba)
        elif metric == "log_loss":
            out[metric] = log_loss(y_np, proba)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + 1 + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks
