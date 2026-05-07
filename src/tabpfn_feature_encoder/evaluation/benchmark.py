from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tabpfn_feature_encoder.config import EncoderConfig
from tabpfn_feature_encoder.data.base import DatasetBundle
from tabpfn_feature_encoder.data.graphs import EventGraphDataset
from tabpfn_feature_encoder.data.preprocessing import Standardizer, stratified_sample_indices
from tabpfn_feature_encoder.evaluation.metrics import accuracy, log_loss, roc_auc
from tabpfn_feature_encoder.models.encoders import require_torch
from tabpfn_feature_encoder.models.tabpfn_adapter import TabPFNPromptAdapter
from tabpfn_feature_encoder.training.encoder_classifier import EncoderOnlyClassifier
from tabpfn_feature_encoder.training.trainer import EncoderTabPFNClassifier
from tabpfn_feature_encoder.utils.io import save_json

BENCHMARK_METRICS = ("accuracy", "roc_auc", "log_loss")


def run_nominal_benchmarks(
    *,
    dataset: DatasetBundle,
    trained_model: EncoderTabPFNClassifier,
    encoder_config: EncoderConfig,
    output_dir: str | Path,
    device: str,
    random_state: int,
) -> dict[str, Any]:
    """Run the nominal test comparison on the existing train/val/test split."""

    context_size = _benchmark_context_size(
        y_train=dataset.y_train,
        encoder_config=encoder_config,
    )
    query_chunk_size = max(1, int(encoder_config.batch_size) - context_size)
    context_idx = stratified_sample_indices(
        dataset.y_train,
        n_samples=context_size,
        random_state=random_state + 20_000,
    )

    baseline_proba = _baseline_tabpfn_proba(
        dataset=dataset,
        context_idx=context_idx,
        query_chunk_size=query_chunk_size,
        device=device,
    )
    baseline_metrics = _classification_metrics(dataset.y_test, baseline_proba)

    use_graph_encoder = isinstance(trained_model.X_train_, EventGraphDataset)
    if use_graph_encoder:
        if dataset.graph_train is None or dataset.graph_test is None:
            raise RuntimeError("Encoder+TabPFN benchmark requires graph_train and graph_test.")
        X_context = dataset.graph_train.subset(context_idx)
        X_query = dataset.graph_test
    else:
        X_context = dataset.X_train.iloc[context_idx]
        X_query = dataset.X_test
    encoder_tabpfn_proba = trained_model.predict_proba_with_context(
        X_context,
        dataset.y_train[context_idx],
        X_query,
        query_chunk_size=query_chunk_size,
    )
    encoder_tabpfn_metrics = _classification_metrics(dataset.y_test, encoder_tabpfn_proba)

    if use_graph_encoder:
        if dataset.graph_train is None or dataset.graph_val is None or dataset.graph_test is None:
            raise RuntimeError("Encoder-only benchmark requires graph train/val/test inputs.")
        encoder_only_X_train = dataset.graph_train
        encoder_only_X_val = dataset.graph_val
        encoder_only_X_test = dataset.graph_test
    else:
        encoder_only_X_train = dataset.X_train
        encoder_only_X_val = dataset.X_val
        encoder_only_X_test = dataset.X_test
    encoder_only = EncoderOnlyClassifier(
        encoder=encoder_config,
        device=device,
        random_state=random_state + 40_000,
    ).fit(
        encoder_only_X_train,
        dataset.y_train,
        X_val=encoder_only_X_val,
        y_val=dataset.y_val,
    )
    encoder_only_proba = encoder_only.predict_proba(encoder_only_X_test)
    encoder_only_metrics = encoder_only.metrics_from_proba(dataset.y_test, encoder_only_proba)

    out = {
        "baseline_tabpfn": baseline_metrics,
        "encoder_tabpfn": encoder_tabpfn_metrics,
        "encoder_only_classifier": encoder_only_metrics,
        "delta_vs_baseline": {
            key: encoder_tabpfn_metrics[key] - baseline_metrics[key]
            for key in BENCHMARK_METRICS
        },
        "encoder_only_delta_vs_baseline": {
            key: encoder_only_metrics[key] - baseline_metrics[key]
            for key in BENCHMARK_METRICS
        },
        "context_size": int(context_size),
        "query_chunk_size": int(query_chunk_size),
        "test_size": int(len(dataset.y_test)),
        "n_train": int(len(dataset.y_train)),
        "n_val": int(len(dataset.y_val)),
        "n_test": int(len(dataset.y_test)),
        "split": "existing_dataset_split",
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(out, output_path / "benchmark_metrics.json")
    np.save(output_path / "benchmark_baseline_tabpfn_proba.npy", baseline_proba)
    np.save(output_path / "benchmark_encoder_tabpfn_proba.npy", encoder_tabpfn_proba)
    np.save(output_path / "benchmark_encoder_only_proba.npy", encoder_only_proba)
    return out


def print_benchmark_summary(results: dict[str, Any]) -> None:
    print("Nominal benchmark test metrics:")
    for name in ("baseline_tabpfn", "encoder_tabpfn", "encoder_only_classifier"):
        metrics = results[name]
        print(
            f"{name}: "
            f"test_accuracy={metrics['accuracy']:.3f}, "
            f"test_roc_auc={metrics['roc_auc']:.3f}, "
            f"test_log_loss={metrics['log_loss']:.3f}"
        )
    print(
        "benchmark setup: "
        f"context={results['context_size']}, "
        f"query_chunk={results['query_chunk_size']}, "
        f"test={results['test_size']}"
    )


def _baseline_tabpfn_proba(
    *,
    dataset: DatasetBundle,
    context_idx: np.ndarray,
    query_chunk_size: int,
    device: str,
) -> np.ndarray:
    standardizer = Standardizer().fit(dataset.X_train.to_numpy(dtype=np.float32))
    X_train = standardizer.transform(dataset.X_train.to_numpy(dtype=np.float32))
    X_test = standardizer.transform(dataset.X_test.to_numpy(dtype=np.float32))
    return _tabpfn_predict_proba(
        X_context=X_train[context_idx],
        y_context=np.asarray(dataset.y_train, dtype=np.int64)[context_idx],
        X_query=X_test,
        query_chunk_size=query_chunk_size,
        device=device,
    )


def _tabpfn_predict_proba(
    *,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    query_chunk_size: int,
    device: str,
) -> np.ndarray:
    torch_mod, _ = require_torch()
    effective_device = _effective_device(device)
    adapter = TabPFNPromptAdapter(device=effective_device).build()
    context_x = torch_mod.tensor(X_context, dtype=torch_mod.float32, device=effective_device)
    context_y = torch_mod.tensor(y_context, dtype=torch_mod.long, device=effective_device)
    adapter.fit_prompt(context_x, context_y)
    parts: list[np.ndarray] = []
    with torch_mod.no_grad():
        for start in range(0, len(X_query), int(query_chunk_size)):
            query_x = torch_mod.tensor(
                X_query[start : start + int(query_chunk_size)],
                dtype=torch_mod.float32,
                device=effective_device,
            )
            parts.append(np.asarray(adapter.predict_proba(query_x)))
    adapter.clear_prompt()
    if str(effective_device).startswith("cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()
    return np.concatenate(parts, axis=0)


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    return {
        "accuracy": accuracy(y_true, pred),
        "roc_auc": roc_auc(y_true, proba),
        "log_loss": log_loss(y_true, proba),
    }


def _benchmark_context_size(*, y_train: np.ndarray, encoder_config: EncoderConfig) -> int:
    n_classes = len(np.unique(y_train))
    requested = int(round(encoder_config.batch_size * encoder_config.support_query_ratio))
    requested = max(n_classes, requested)
    return min(requested, len(y_train))


def _effective_device(device: str) -> str:
    torch_mod, _ = require_torch()
    if str(device).startswith("cuda") and not torch_mod.cuda.is_available():
        return "cpu"
    return str(device)
