from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tabpfn_feature_encoder.data.graphs import EventGraphDataset, GraphStandardizer
from tabpfn_feature_encoder.data.preprocessing import Standardizer, stratified_sample_indices
from tabpfn_feature_encoder.evaluation.metrics import accuracy, log_loss, roc_auc
from tabpfn_feature_encoder.models.encoders import require_torch
from tabpfn_feature_encoder.models.tabpfn_adapter import TabPFNPromptAdapter
from tabpfn_feature_encoder.training.trainer import EncoderTabPFNClassifier
from tabpfn_feature_encoder.utils.io import load_pickle, save_json


def run_gnn_transfer_evaluation(
    *,
    encoder_model_path: str | Path,
    dataset: Any,
    output_dir: str | Path,
    context_size: int,
    query_chunk_size: int,
    device: str,
    random_state: int,
) -> dict[str, Any]:
    if dataset.graph_train is None or dataset.graph_test is None:
        raise RuntimeError("Transfer evaluation requires graph_train and graph_test.")

    trained = load_pickle(encoder_model_path)
    if not isinstance(trained, EncoderTabPFNClassifier):
        raise TypeError("encoder_model must be a saved EncoderTabPFNClassifier.")
    if trained.encoder_model_ is None:
        raise RuntimeError("Saved model does not contain an encoder_model_.")
    if getattr(trained.encoder_model_, "encoder_type", None) != "gnn":
        raise TypeError("Transfer evaluation currently expects a saved GNN encoder.")

    torch_mod, _ = require_torch()
    effective_device = _effective_device(device)
    encoder = trained.encoder_model_.to(effective_device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    graph_standardizer = trained.graph_standardizer_
    if not isinstance(graph_standardizer, GraphStandardizer):
        raise TypeError("Saved model does not contain a graph standardizer.")

    y_train = np.asarray(dataset.y_train, dtype=np.int64)
    y_test = np.asarray(dataset.y_test, dtype=np.int64)
    context_idx = stratified_sample_indices(
        y_train,
        n_samples=min(int(context_size), len(y_train)),
        random_state=random_state + 30_000,
    )

    graph_train = graph_standardizer.transform(dataset.graph_train)
    graph_test = graph_standardizer.transform(dataset.graph_test)
    encoded_context = encode_graph_dataset(
        encoder=encoder,
        dataset=graph_train.subset(context_idx),
        batch_size=query_chunk_size,
        device=effective_device,
    )
    encoded_test = encode_graph_dataset(
        encoder=encoder,
        dataset=graph_test,
        batch_size=query_chunk_size,
        device=effective_device,
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

    combined_context = np.concatenate([encoded_context, flat_train[context_idx]], axis=1)
    combined_test = np.concatenate([encoded_test, flat_test], axis=1)
    combined_proba = _tabpfn_predict_proba(
        X_context=combined_context,
        y_context=y_train[context_idx],
        X_query=combined_test,
        query_chunk_size=query_chunk_size,
        device=effective_device,
    )
    combined_metrics = _classification_metrics(y_test, combined_proba)

    out = {
        "frozen_gnn_tabpfn": encoded_metrics,
        "frozen_gnn_plus_flat_tabpfn": combined_metrics,
        "baseline_tabpfn": baseline_metrics,
        "delta": {
            key: encoded_metrics[key] - baseline_metrics[key]
            for key in encoded_metrics
            if key in baseline_metrics
        },
        "plus_flat_delta": {
            key: combined_metrics[key] - baseline_metrics[key]
            for key in combined_metrics
            if key in baseline_metrics
        },
        "context_size": int(len(context_idx)),
        "query_size": int(len(y_test)),
        "class_names": dataset.metadata.get("label_names", {}),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_flat_features": int(dataset.X_train.shape[1]),
        "n_encoded_features": int(encoded_context.shape[1]),
        "n_combined_features": int(combined_context.shape[1]),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(out, output_path / "transfer_metrics.json")
    np.save(output_path / "frozen_gnn_test_proba.npy", encoded_proba)
    np.save(output_path / "frozen_gnn_plus_flat_test_proba.npy", combined_proba)
    np.save(output_path / "baseline_test_proba.npy", baseline_proba)
    torch_mod.cuda.empty_cache() if str(effective_device).startswith("cuda") else None
    return out


def encode_graph_dataset(
    *,
    encoder: Any,
    dataset: EventGraphDataset,
    batch_size: int,
    device: str,
) -> np.ndarray:
    torch_mod, _ = require_torch()
    parts: list[np.ndarray] = []
    with torch_mod.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            stop = min(start + int(batch_size), len(dataset))
            batch = dataset.to_batch(np.arange(start, stop, dtype=np.int64), device=device)
            parts.append(encoder(batch).detach().cpu().numpy())
    if not parts:
        return np.zeros((0, int(encoder.output_dim)), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


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
