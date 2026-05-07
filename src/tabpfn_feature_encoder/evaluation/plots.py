from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tabpfn_feature_encoder.evaluation.metrics import binary_roc_auc


def collect_outputs(model: Any, X: Any, y: Any) -> dict[str, np.ndarray]:
    proba = np.asarray(model.predict_proba(X))
    pred = np.asarray(model.predict(X)) if hasattr(model, "predict") else np.argmax(proba, axis=1)
    return {
        "y_true": np.asarray(y),
        "y_pred": pred,
        "proba": proba,
        "positive_scores": proba[:, 1],
    }


def save_binary_classification_plots(
    outputs: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    prefix: str = "encoder_tabpfn",
) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots. Install with "
            "`python -m pip install -e '.[plots]'`."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(outputs["y_true"])
    y_pred = np.asarray(outputs["y_pred"])
    scores = np.asarray(outputs["positive_scores"])
    saved: dict[str, Path] = {}

    cm = _confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["CP-even", "CP-odd"])
    ax.set_yticks([0, 1], labels=["CP-even", "CP-odd"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center")
    fig.tight_layout()
    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)
    saved["confusion_matrix"] = cm_path

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(scores[y_true == 0], bins=30, alpha=0.7, label="True CP-even", density=True)
    ax.hist(scores[y_true == 1], bins=30, alpha=0.7, label="True CP-odd", density=True)
    ax.set_xlabel("Predicted CP-odd probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    scores_path = out_dir / f"{prefix}_score_distribution.png"
    fig.savefig(scores_path, dpi=160)
    plt.close(fig)
    saved["score_distribution"] = scores_path

    fpr, tpr = _roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {binary_roc_auc(y_true, scores):.4f}", linewidth=2.0)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = out_dir / f"{prefix}_roc_curve.png"
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)
    saved["roc_curve"] = roc_path

    predictions_path = out_dir / f"{prefix}_predictions.npz"
    np.savez(
        predictions_path,
        y_true=y_true,
        y_pred=y_pred,
        positive_scores=scores,
        proba=np.asarray(outputs["proba"]),
    )
    saved["predictions"] = predictions_path
    return saved


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    out = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if int(true) in label_to_idx and int(pred) in label_to_idx:
            out[label_to_idx[int(true)], label_to_idx[int(pred)]] += 1
    return out


def _roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y_sorted = (y_true[order] == np.max(np.unique(y_true))).astype(np.int64)
    positives = max(1, int(np.sum(y_sorted)))
    negatives = max(1, len(y_sorted) - positives)
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    return fpr, tpr
