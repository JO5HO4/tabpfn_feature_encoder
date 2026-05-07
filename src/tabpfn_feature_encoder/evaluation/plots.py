from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from tabpfn_feature_encoder.evaluation.metrics import binary_roc_auc


CONTEXT_SCAN_TASKS = {
    "source_12_class_generalization": (
        "Source 12-Class",
        Path("source_generalization/source_12_class_generalization_context_scan_metrics.csv"),
    ),
    "cp_even_odd_generalization": (
        "CP Even/Odd",
        Path("cp_generalization/cp_even_odd_generalization_context_scan_metrics.csv"),
    ),
    "open_data_generalization": (
        "Open Data GamGam",
        Path("open_data_generalization/open_data_generalization_context_scan_metrics.csv"),
    ),
}


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
    prefix: str = "classifier",
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


def save_context_scan_plots(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    prefix: str = "context_scan",
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

    ok_records = [record for record in records if record.get("status") == "ok"]
    if not ok_records:
        return {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    for metric in ("roc_auc", "accuracy", "log_loss"):
        baseline = _aggregate_records(ok_records, "baseline_tabpfn", metric)
        frozen = _aggregate_records(ok_records, "frozen_encoder_tabpfn", metric)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.errorbar(
            baseline["context_size"],
            baseline["mean"],
            yerr=baseline["std"],
            marker="o",
            capsize=3,
            label="Baseline TabPFN",
        )
        ax.errorbar(
            frozen["context_size"],
            frozen["mean"],
            yerr=frozen["std"],
            marker="o",
            capsize=3,
            label="Frozen encoder + TabPFN",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Validation context events")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Context Scan: {metric.replace('_', ' ').title()} Mean +/- 1 Std")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{prefix}_context_scan_{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved[metric] = path

    return saved


def save_encoder_comparison_plots(
    run_dirs: list[tuple[str, str | Path]],
    output_dir: str | Path,
) -> dict[str, list[Path]]:
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
    saved: dict[str, list[Path]] = {}

    for task_name, (task_title, relative_csv) in CONTEXT_SCAN_TASKS.items():
        loaded = [
            (label, _read_context_scan_csv(Path(run_dir) / relative_csv))
            for label, run_dir in run_dirs
            if (Path(run_dir) / relative_csv).exists()
        ]
        loaded = [(label, rows) for label, rows in loaded if rows]
        if not loaded:
            continue

        task_paths: list[Path] = []
        for metric, ylabel in (("roc_auc", "AUC"), ("accuracy", "Accuracy")):
            fig, ax = plt.subplots(figsize=(7.0, 4.6))
            baseline_summary = _aggregate_csv_rows(loaded[0][1], f"baseline_{metric}")
            _plot_context_summary(
                ax,
                baseline_summary,
                label="Baseline TabPFN",
                marker="o",
                linewidth=2.0,
            )
            for label, rows in loaded:
                encoder_summary = _aggregate_csv_rows(rows, f"frozen_encoder_{metric}")
                _plot_context_summary(
                    ax,
                    encoder_summary,
                    label=label,
                    marker="s",
                    linewidth=1.8,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Validation context events")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{task_title}: {ylabel} vs Context Size")
            ax.legend()
            ax.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            path = out_dir / f"{task_name}_{metric}_comparison.pdf"
            fig.savefig(path)
            plt.close(fig)
            task_paths.append(path)
        saved[task_name] = task_paths

    return saved


def _aggregate_records(
    records: list[dict[str, Any]],
    family: str,
    metric: str,
) -> dict[str, np.ndarray]:
    context_sizes = sorted({int(record["context_size"]) for record in records})
    means: list[float] = []
    stds: list[float] = []
    for context_size in context_sizes:
        values = [
            float(record[family][metric])
            for record in records
            if int(record["context_size"]) == context_size
        ]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return {
        "context_size": np.asarray(context_sizes, dtype=np.float64),
        "mean": np.asarray(means, dtype=np.float64),
        "std": np.asarray(stds, dtype=np.float64),
    }


def _read_context_scan_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def _aggregate_csv_rows(rows: list[dict[str, str]], value_col: str) -> dict[str, np.ndarray]:
    context_sizes = sorted(
        {
            int(float(row["context_size"]))
            for row in rows
            if row.get(value_col) not in {None, ""}
        }
    )
    means: list[float] = []
    stds: list[float] = []
    for context_size in context_sizes:
        values = [
            float(row[value_col])
            for row in rows
            if int(float(row["context_size"])) == context_size
            and row.get(value_col) not in {None, ""}
        ]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return {
        "context_size": np.asarray(context_sizes, dtype=np.float64),
        "mean": np.asarray(means, dtype=np.float64),
        "std": np.asarray(stds, dtype=np.float64),
    }


def _plot_context_summary(
    ax: Any,
    summary: dict[str, np.ndarray],
    *,
    label: str,
    marker: str,
    linewidth: float,
) -> None:
    if len(summary["context_size"]) == 0:
        return
    ax.errorbar(
        summary["context_size"],
        summary["mean"],
        yerr=summary["std"],
        marker=marker,
        linewidth=linewidth,
        capsize=3,
        label=label,
    )


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
