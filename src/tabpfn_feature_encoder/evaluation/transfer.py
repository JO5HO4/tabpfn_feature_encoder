from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tabpfn_feature_encoder.data.base import DatasetBundle
from tabpfn_feature_encoder.data.graphs import EventGraphDataset
from tabpfn_feature_encoder.data.preprocessing import Standardizer, stratified_sample_indices
from tabpfn_feature_encoder.evaluation.metrics import accuracy, log_loss, roc_auc
from tabpfn_feature_encoder.models.tabpfn_adapter import TabPFNPromptAdapter
from tabpfn_feature_encoder.models.torch_utils import require_torch
from tabpfn_feature_encoder.training.encoder_classifier import EncoderOnlyClassifier
from tabpfn_feature_encoder.utils.io import save_json


def run_encoder_transfer_evaluation(
    *,
    trained: EncoderOnlyClassifier,
    dataset: DatasetBundle,
    output_dir: str | Path,
    context_size: int,
    query_chunk_size: int,
    device: str,
    random_state: int,
    name: str,
) -> dict[str, Any]:
    """Evaluate a frozen supervised encoder on a downstream TabPFN task."""

    effective_device = _effective_device(device)
    _move_encoder(trained, effective_device)
    y_train = np.asarray(dataset.y_train, dtype=np.int64)
    y_test = np.asarray(dataset.y_test, dtype=np.int64)
    context_idx = stratified_sample_indices(
        y_train,
        n_samples=min(int(context_size), len(y_train)),
        random_state=random_state + 30_000,
    )

    encoded_context = _encode_subset(
        trained=trained,
        dataset=dataset,
        split="train",
        indices=context_idx,
        batch_size=query_chunk_size,
    )
    encoded_test = _encode_subset(
        trained=trained,
        dataset=dataset,
        split="test",
        indices=None,
        batch_size=query_chunk_size,
    )
    encoded_proba = _tabpfn_predict_proba(
        X_context=encoded_context,
        y_context=y_train[context_idx],
        X_query=encoded_test,
        query_chunk_size=query_chunk_size,
        device=effective_device,
    )
    encoded_metrics = _classification_metrics(y_test, encoded_proba)

    flat_standardizer = Standardizer().fit(dataset.X_train.to_numpy(dtype=np.float32))
    flat_train = flat_standardizer.transform(dataset.X_train.to_numpy(dtype=np.float32))
    flat_test = flat_standardizer.transform(dataset.X_test.to_numpy(dtype=np.float32))
    baseline_proba = _tabpfn_predict_proba(
        X_context=flat_train[context_idx],
        y_context=y_train[context_idx],
        X_query=flat_test,
        query_chunk_size=query_chunk_size,
        device=effective_device,
    )
    baseline_metrics = _classification_metrics(y_test, baseline_proba)

    out = {
        "frozen_encoder_tabpfn": encoded_metrics,
        "baseline_tabpfn": baseline_metrics,
        "delta": {
            key: encoded_metrics[key] - baseline_metrics[key]
            for key in encoded_metrics
            if key in baseline_metrics
        },
        "context_size": int(len(context_idx)),
        "query_size": int(len(y_test)),
        "class_names": dataset.metadata.get("label_names", {}),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_flat_features": int(dataset.X_train.shape[1]),
        "n_encoded_features": int(encoded_context.shape[1]),
        "source_encoder_classes": (
            None if trained.classes_ is None else [int(label) for label in trained.classes_]
        ),
        "task_name": name,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(out, output_path / f"{name}_metrics.json")
    np.save(output_path / f"{name}_frozen_encoder_proba.npy", encoded_proba)
    np.save(output_path / f"{name}_baseline_proba.npy", baseline_proba)
    if str(effective_device).startswith("cuda"):
        torch_mod, _ = require_torch()
        if torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
    return out


def print_transfer_summary(name: str, metrics: dict[str, Any]) -> None:
    for family in ("baseline_tabpfn", "frozen_encoder_tabpfn", "delta"):
        values = metrics[family]
        text = ", ".join(f"{key}={value:.3f}" for key, value in values.items())
        print(f"{name} {family}: {text}")


def _encode_subset(
    *,
    trained: EncoderOnlyClassifier,
    dataset: DatasetBundle,
    split: str,
    indices: np.ndarray | None,
    batch_size: int,
) -> np.ndarray:
    if trained.is_graph_input_:
        graph_dataset = getattr(dataset, f"graph_{split}")
        if not isinstance(graph_dataset, EventGraphDataset):
            raise RuntimeError("Frozen graph encoder requires graph downstream features.")
        X = graph_dataset if indices is None else graph_dataset.subset(indices)
    else:
        X_df = getattr(dataset, f"X_{split}")
        X = X_df if indices is None else X_df.iloc[indices]
    return trained.encode(X, batch_size=batch_size)


def _move_encoder(trained: EncoderOnlyClassifier, device: str) -> None:
    if trained.encoder_model_ is not None:
        trained.encoder_model_.to(device)
    if trained.classifier_head_ is not None:
        trained.classifier_head_.to(device)


def _tabpfn_predict_proba(
    *,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    query_chunk_size: int,
    device: str,
) -> np.ndarray:
    torch_mod, _ = require_torch()
    adapter = TabPFNPromptAdapter(device=device).build()
    context_x = torch_mod.tensor(X_context, dtype=torch_mod.float32, device=device)
    context_y = torch_mod.tensor(y_context, dtype=torch_mod.long, device=device)
    adapter.fit_prompt(context_x, context_y)
    parts: list[np.ndarray] = []
    with torch_mod.no_grad():
        for start in range(0, len(X_query), int(query_chunk_size)):
            query_x = torch_mod.tensor(
                X_query[start : start + int(query_chunk_size)],
                dtype=torch_mod.float32,
                device=device,
            )
            parts.append(np.asarray(adapter.predict_proba(query_x)))
    adapter.clear_prompt()
    if str(device).startswith("cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()
    return np.concatenate(parts, axis=0)


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    return {
        "accuracy": accuracy(y_true, pred),
        "log_loss": log_loss(y_true, proba),
        "roc_auc": roc_auc(y_true, proba),
    }


def _effective_device(device: str) -> str:
    torch_mod, _ = require_torch()
    if str(device).startswith("cuda") and not torch_mod.cuda.is_available():
        return "cpu"
    return str(device)
