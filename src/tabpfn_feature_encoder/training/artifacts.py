from __future__ import annotations

from pathlib import Path
from typing import Any

from tabpfn_feature_encoder.utils.io import save_json, save_pickle


def save_training_artifacts(
    *,
    output_dir: str | Path,
    metrics: dict[str, Any],
    training_summary: dict[str, Any],
    model: Any | None = None,
    model_artifact: str = "encoder_classifier.pkl",
    save_model: bool = True,
    save_metrics: bool = True,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    if save_metrics:
        metrics_path = out / "metrics.json"
        summary_path = out / "training_summary.json"
        save_json(metrics, metrics_path)
        save_json(training_summary, summary_path)
        saved["metrics"] = metrics_path
        saved["training_summary"] = summary_path
    if save_model and model is not None:
        model_path = out / model_artifact
        save_pickle(model, model_path)
        saved["model"] = model_path
    return saved
