from __future__ import annotations

import csv
import json
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
    return run_encoder_context_scan_evaluation(
        trained=trained,
        dataset=dataset,
        output_dir=output_dir,
        context_min_per_class=1,
        context_scan_points=1,
        context_repeats=1,
        max_context_size=context_size,
        query_chunk_size=query_chunk_size,
        device=device,
        random_state=random_state,
        name=name,
    )


def run_encoder_context_scan_evaluation(
    *,
    trained: EncoderOnlyClassifier,
    dataset: DatasetBundle,
    output_dir: str | Path,
    context_min_per_class: int,
    context_scan_points: int,
    context_repeats: int,
    max_context_size: int | None,
    query_chunk_size: int,
    device: str,
    random_state: int,
    name: str,
) -> dict[str, Any]:
    """Scan TabPFN downstream context size using validation as context and test as query."""
    effective_device = _effective_device(device)
    _move_encoder(trained, effective_device)
    y_context_pool = np.asarray(dataset.y_val, dtype=np.int64)
    y_test = np.asarray(dataset.y_test, dtype=np.int64)
    context_sizes = _context_scan_sizes(
        y_context_pool,
        min_per_class=context_min_per_class,
        n_points=context_scan_points,
        max_context_size=max_context_size,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / f"{name}_context_scan_metrics.json"
    expected_keys = {
        (int(size), int(repeat_idx))
        for size in context_sizes
        for repeat_idx in range(int(context_repeats))
    }
    loaded_records = _load_scan_records(records_path)
    records = [
        record
        for record in loaded_records
        if _scan_record_key(record) in expected_keys
    ]
    ignored_records = len(loaded_records) - len(records)
    completed_keys = {
        _scan_record_key(record)
        for record in records
        if record.get("status") == "ok"
    }

    print(
        f"{name} context scan setup: "
        f"context_pool={len(y_context_pool)}, query={len(y_test)}, "
        f"sizes={','.join(str(int(size)) for size in context_sizes)}, "
        f"repeats={context_repeats}, query_chunk={query_chunk_size}, "
        f"device={effective_device}",
        flush=True,
    )
    if records:
        print(
            f"{name} loaded {len(records)} existing context-scan record(s); "
            f"{len(completed_keys)} completed repeat(s) will be skipped.",
            flush=True,
        )
    if ignored_records:
        print(
            f"{name} ignored {ignored_records} existing context-scan record(s) "
            "outside the current scan grid.",
            flush=True,
        )
    print(f"{name} encoding frozen-encoder validation and test features...", flush=True)
    encoded_context_pool = _encode_subset(
        trained=trained,
        dataset=dataset,
        split="val",
        indices=None,
        batch_size=query_chunk_size,
    )
    encoded_test = _encode_subset(
        trained=trained,
        dataset=dataset,
        split="test",
        indices=None,
        batch_size=query_chunk_size,
    )
    print(
        f"{name} encoded features ready: "
        f"context_shape={encoded_context_pool.shape}, query_shape={encoded_test.shape}",
        flush=True,
    )
    flat_context_pool = dataset.X_val.to_numpy(dtype=np.float32)
    flat_test_raw = dataset.X_test.to_numpy(dtype=np.float32)

    last_encoded_proba: np.ndarray | None = None
    last_baseline_proba: np.ndarray | None = None
    for scan_idx, context_size in enumerate(context_sizes):
        size_oom = False
        print(
            f"{name} context size {scan_idx + 1}/{len(context_sizes)}: "
            f"requested={int(context_size)}, repeats={context_repeats}",
            flush=True,
        )
        for repeat_idx in range(int(context_repeats)):
            record_key = (int(context_size), int(repeat_idx))
            if record_key in completed_keys:
                print(
                    f"{name} context={int(context_size)} repeat={repeat_idx + 1}/"
                    f"{context_repeats}: already complete; skipping.",
                    flush=True,
                )
                continue
            context_idx = stratified_sample_indices(
                y_context_pool,
                n_samples=int(context_size),
                random_state=random_state + 30_000 + scan_idx * 1_000 + repeat_idx,
            )
            context_counts = _class_counts(y_context_pool[context_idx])
            print(
                f"{name} context={len(context_idx)} repeat={repeat_idx + 1}/{context_repeats}: "
                "running frozen-encoder and baseline TabPFN predictions...",
                flush=True,
            )
            try:
                encoded_proba = _tabpfn_predict_proba(
                    X_context=encoded_context_pool[context_idx],
                    y_context=y_context_pool[context_idx],
                    X_query=encoded_test,
                    query_chunk_size=query_chunk_size,
                    device=effective_device,
                )
                encoded_metrics = _classification_metrics(y_test, encoded_proba)

                flat_standardizer = Standardizer().fit(flat_context_pool[context_idx])
                flat_context = flat_standardizer.transform(flat_context_pool[context_idx])
                flat_test = flat_standardizer.transform(flat_test_raw)
                baseline_proba = _tabpfn_predict_proba(
                    X_context=flat_context,
                    y_context=y_context_pool[context_idx],
                    X_query=flat_test,
                    query_chunk_size=query_chunk_size,
                    device=effective_device,
                )
                baseline_metrics = _classification_metrics(y_test, baseline_proba)
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise
                _clear_cuda_cache(effective_device)
                _upsert_scan_record(
                    records,
                    {
                        "context_size": int(len(context_idx)),
                        "requested_context_size": int(context_size),
                        "repeat": int(repeat_idx),
                        "context_class_counts": context_counts,
                        "status": "oom",
                        "error": str(exc).splitlines()[0],
                    },
                )
                _save_context_scan_artifacts(
                    records=records,
                    output_path=output_path,
                    name=name,
                    context_sizes=context_sizes,
                    context_repeats=context_repeats,
                    y_test=y_test,
                    dataset=dataset,
                    encoded_feature_count=int(encoded_context_pool.shape[1]),
                    trained=trained,
                    save_plots=False,
                )
                size_oom = True
                break

            record = {
                "context_size": int(len(context_idx)),
                "requested_context_size": int(context_size),
                "repeat": int(repeat_idx),
                "context_class_counts": context_counts,
                "status": "ok",
                "baseline_tabpfn": baseline_metrics,
                "frozen_encoder_tabpfn": encoded_metrics,
                "delta": {
                    key: encoded_metrics[key] - baseline_metrics[key]
                    for key in encoded_metrics
                    if key in baseline_metrics
                },
            }
            _upsert_scan_record(records, record)
            completed_keys.add(record_key)
            last_encoded_proba = encoded_proba
            last_baseline_proba = baseline_proba
            print(
                f"{name} context={record['context_size']} repeat={repeat_idx + 1}/{context_repeats}: "
                f"baseline_auc={baseline_metrics['roc_auc']:.3f}, "
                f"encoder_auc={encoded_metrics['roc_auc']:.3f}, "
                f"delta_auc={record['delta']['roc_auc']:.3f}",
                flush=True,
            )
            _save_context_scan_artifacts(
                records=records,
                output_path=output_path,
                name=name,
                context_sizes=context_sizes,
                context_repeats=context_repeats,
                y_test=y_test,
                dataset=dataset,
                encoded_feature_count=int(encoded_context_pool.shape[1]),
                trained=trained,
                save_plots=False,
            )
        if size_oom:
            break

    out = _save_context_scan_artifacts(
        records=records,
        output_path=output_path,
        name=name,
        context_sizes=context_sizes,
        context_repeats=context_repeats,
        y_test=y_test,
        dataset=dataset,
        encoded_feature_count=int(encoded_context_pool.shape[1]),
        trained=trained,
        save_plots=True,
    )
    if not [record for record in records if record.get("status") == "ok"]:
        raise RuntimeError(f"No context sizes completed for {name}.")
    print(f"{name} saved context-scan artifacts: {output_path}", flush=True)
    if last_encoded_proba is not None:
        np.save(output_path / f"{name}_frozen_encoder_proba.npy", last_encoded_proba)
    if last_baseline_proba is not None:
        np.save(output_path / f"{name}_baseline_proba.npy", last_baseline_proba)
    _clear_cuda_cache(effective_device)
    return out


def _load_scan_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Context scan record file must contain a list: {path}")
    return [record for record in payload if isinstance(record, dict)]


def _scan_record_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        int(record.get("requested_context_size", record.get("context_size", 0))),
        int(record.get("repeat", 0)),
    )


def _upsert_scan_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    key = _scan_record_key(record)
    for idx, existing in enumerate(records):
        if _scan_record_key(existing) == key:
            records[idx] = record
            return
    records.append(record)


def _ordered_scan_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            int(record.get("requested_context_size", record.get("context_size", 0))),
            int(record.get("repeat", 0)),
        ),
    )


def _save_context_scan_artifacts(
    *,
    records: list[dict[str, Any]],
    output_path: Path,
    name: str,
    context_sizes: list[int],
    context_repeats: int,
    y_test: np.ndarray,
    dataset: DatasetBundle,
    encoded_feature_count: int,
    trained: EncoderOnlyClassifier,
    save_plots: bool,
) -> dict[str, Any]:
    output_path.mkdir(parents=True, exist_ok=True)
    ordered_records = _ordered_scan_records(records)
    successful = [record for record in ordered_records if record.get("status") == "ok"]
    expected_repeats = int(len(context_sizes) * int(context_repeats))
    out: dict[str, Any] = {
        "status": "complete" if len(successful) >= expected_repeats else "incomplete",
        "completed_repeats": int(len(successful)),
        "expected_repeats": expected_repeats,
        "context_split": "val",
        "context_scan": ordered_records,
        "context_scan_sizes": [int(size) for size in context_sizes],
        "context_repeats": int(context_repeats),
        "query_split": "test",
        "query_size": int(len(y_test)),
        "class_names": dataset.metadata.get("label_names", {}),
        "n_train": int(len(dataset.y_train)),
        "n_context_pool": int(len(dataset.y_val)),
        "n_test": int(len(y_test)),
        "n_flat_features": int(dataset.X_train.shape[1]),
        "n_encoded_features": int(encoded_feature_count),
        "source_encoder_classes": (
            None if trained.classes_ is None else [int(label) for label in trained.classes_]
        ),
        "task_name": name,
    }
    if successful:
        summary_records = _largest_context_records(successful)
        out.update(
            {
                "baseline_tabpfn": _mean_metrics(summary_records, "baseline_tabpfn"),
                "frozen_encoder_tabpfn": _mean_metrics(
                    summary_records,
                    "frozen_encoder_tabpfn",
                ),
                "delta": _mean_metrics(summary_records, "delta"),
                "context_size": int(summary_records[0]["context_size"]),
            }
        )
    else:
        out.update(
            {
                "baseline_tabpfn": {},
                "frozen_encoder_tabpfn": {},
                "delta": {},
                "context_size": None,
            }
        )
    save_json(out, output_path / f"{name}_metrics.json")
    save_json(ordered_records, output_path / f"{name}_context_scan_metrics.json")
    _save_scan_csv(ordered_records, output_path / f"{name}_context_scan_metrics.csv")
    if save_plots and successful:
        _save_scan_plots(ordered_records, output_path, name)
    return out


def print_transfer_summary(name: str, metrics: dict[str, Any]) -> None:
    if "context_scan" in metrics:
        n_ok = sum(1 for record in metrics["context_scan"] if record.get("status") == "ok")
        print(
            f"{name} context scan: split=val, query=test, "
            f"completed={n_ok}/{len(metrics['context_scan'])}, "
            f"summary_context={metrics['context_size']}"
        )
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
    y_context = np.asarray(y_context, dtype=np.int64)
    classes = np.unique(y_context)
    max_classes = 10
    if len(classes) > max_classes:
        return _tabpfn_predict_ecoc_proba(
            X_context=X_context,
            y_context=y_context,
            X_query=X_query,
            query_chunk_size=query_chunk_size,
            device=device,
            classes=classes,
            max_classes=max_classes,
        )
    return _tabpfn_predict_small_proba(
        X_context=X_context,
        y_context=y_context,
        X_query=X_query,
        query_chunk_size=query_chunk_size,
        device=device,
        n_classes=len(classes),
    )


def _tabpfn_predict_ecoc_proba(
    *,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    query_chunk_size: int,
    device: str,
    classes: np.ndarray,
    max_classes: int,
) -> np.ndarray:
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    y_context_encoded = np.asarray([class_to_idx[int(label)] for label in y_context], dtype=np.int64)
    codebook = EncoderOnlyClassifier._make_ecoc_codebook(
        n_classes=len(classes),
        alphabet_size=max_classes,
        redundancy=4,
        random_state=30_421,
    )
    if codebook is None:
        raise RuntimeError("ECOC codebook was unexpectedly empty for a many-class task.")
    class_scores = np.zeros((len(X_query), len(classes)), dtype=np.float64)
    for task_idx in range(codebook.shape[1]):
        y_task = codebook[y_context_encoded, task_idx]
        task_proba = _tabpfn_predict_small_proba(
            X_context=X_context,
            y_context=y_task,
            X_query=X_query,
            query_chunk_size=query_chunk_size,
            device=device,
            n_classes=max_classes,
        )
        class_scores += np.log(np.clip(task_proba[:, codebook[:, task_idx]], 1e-15, 1.0))
    return _softmax_np(class_scores)


def _tabpfn_predict_small_proba(
    *,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    query_chunk_size: int,
    device: str,
    n_classes: int,
) -> np.ndarray:
    torch_mod, _ = require_torch()
    adapter = TabPFNPromptAdapter(device=device, random_state=30_421).build()
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
    proba = np.concatenate(parts, axis=0)
    return _pad_proba(proba, n_classes)


def _pad_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
    if proba.shape[1] == n_classes:
        return proba
    if proba.shape[1] > n_classes:
        return proba[:, :n_classes]
    out = np.full((proba.shape[0], n_classes), 1e-15, dtype=np.float64)
    out[:, : proba.shape[1]] = proba
    return out / out.sum(axis=1, keepdims=True)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    return {
        "accuracy": accuracy(y_true, pred),
        "log_loss": log_loss(y_true, proba),
        "roc_auc": roc_auc(y_true, proba),
    }


def _context_scan_sizes(
    y_context: np.ndarray,
    *,
    min_per_class: int,
    n_points: int,
    max_context_size: int | None,
) -> list[int]:
    y_context = np.asarray(y_context)
    if len(y_context) == 0:
        raise ValueError("Context split is empty.")
    classes = np.unique(y_context)
    n_classes = max(1, int(len(classes)))
    max_size = (
        len(y_context)
        if max_context_size is None
        else min(int(max_context_size), len(y_context))
    )
    if max_size < n_classes:
        raise ValueError("max_context_size must be at least the number of classes.")
    min_size = min(max_size, max(n_classes, int(min_per_class) * n_classes))
    if n_points <= 1 or min_size == max_size:
        return [int(max_size)]
    raw = np.geomspace(min_size, max_size, num=int(n_points))
    sizes = sorted({int(round(value)) for value in raw})
    sizes[0] = int(min_size)
    sizes[-1] = int(max_size)
    return sizes


def _class_counts(y: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(classes, counts)}


def _save_scan_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "context_size",
        "requested_context_size",
        "repeat",
        "status",
        "baseline_accuracy",
        "baseline_log_loss",
        "baseline_roc_auc",
        "frozen_encoder_accuracy",
        "frozen_encoder_log_loss",
        "frozen_encoder_roc_auc",
        "delta_accuracy",
        "delta_log_loss",
        "delta_roc_auc",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(_scan_csv_row(record))


def _scan_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    baseline = record.get("baseline_tabpfn", {})
    frozen = record.get("frozen_encoder_tabpfn", {})
    delta = record.get("delta", {})
    return {
        "context_size": record.get("context_size"),
        "requested_context_size": record.get("requested_context_size"),
        "repeat": record.get("repeat"),
        "status": record.get("status"),
        "baseline_accuracy": baseline.get("accuracy"),
        "baseline_log_loss": baseline.get("log_loss"),
        "baseline_roc_auc": baseline.get("roc_auc"),
        "frozen_encoder_accuracy": frozen.get("accuracy"),
        "frozen_encoder_log_loss": frozen.get("log_loss"),
        "frozen_encoder_roc_auc": frozen.get("roc_auc"),
        "delta_accuracy": delta.get("accuracy"),
        "delta_log_loss": delta.get("log_loss"),
        "delta_roc_auc": delta.get("roc_auc"),
        "error": record.get("error"),
    }


def _save_scan_plots(records: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    try:
        from tabpfn_feature_encoder.evaluation.plots import save_context_scan_plots

        save_context_scan_plots(records, output_dir=output_dir, prefix=name)
    except ImportError:
        return


def _largest_context_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_context = max(int(record["context_size"]) for record in records)
    return [record for record in records if int(record["context_size"]) == max_context]


def _mean_metrics(records: list[dict[str, Any]], family: str) -> dict[str, float]:
    metrics = records[0][family].keys()
    out: dict[str, float] = {}
    for metric in metrics:
        values = [float(record[family][metric]) for record in records]
        out[metric] = float(np.mean(values))
    return out


def _is_cuda_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def _clear_cuda_cache(device: str) -> None:
    if str(device).startswith("cuda"):
        torch_mod, _ = require_torch()
        if torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()


def _effective_device(device: str) -> str:
    torch_mod, _ = require_torch()
    if str(device).startswith("cuda") and not torch_mod.cuda.is_available():
        return "cpu"
    return str(device)
