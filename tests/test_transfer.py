import json

import numpy as np
import pandas as pd
import pytest

from tabpfn_feature_encoder.data.base import DatasetBundle
from tabpfn_feature_encoder.evaluation.plots import save_encoder_comparison_plots
from tabpfn_feature_encoder.evaluation import transfer as transfer_mod
from tabpfn_feature_encoder.evaluation.transfer import (
    _context_scan_sizes,
    run_encoder_context_scan_evaluation,
)


class FakeEncoder:
    is_graph_input_ = False
    encoder_model_ = None
    classifier_head_ = None
    classes_ = np.array([0, 1])

    def encode(self, X, batch_size=None):
        return np.asarray(X, dtype=np.float32)


def test_context_scan_sizes_start_at_per_class_minimum_and_end_at_full_val() -> None:
    y_context = np.repeat([0, 1], 12_500)

    sizes = _context_scan_sizes(
        y_context,
        min_per_class=100,
        n_points=12,
        max_context_size=None,
    )

    assert sizes[0] == 200
    assert sizes[-1] == 25_000
    assert 10 <= len(sizes) <= 12
    assert sizes == sorted(set(sizes))


def test_context_scan_sizes_honor_optional_cap() -> None:
    y_context = np.repeat([0, 1, 2], 1_000)

    sizes = _context_scan_sizes(
        y_context,
        min_per_class=100,
        n_points=16,
        max_context_size=900,
    )

    assert sizes[0] == 300
    assert sizes[-1] == 900


def test_context_scan_reuses_existing_records_and_writes_rolling_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    def fake_predict_proba(**kwargs):
        calls.append(int(len(kwargs["X_context"])))
        y_context = np.asarray(kwargs["y_context"], dtype=np.int64)
        n_classes = max(2, int(np.max(y_context)) + 1)
        proba = np.full((len(kwargs["X_query"]), n_classes), 0.25, dtype=np.float64)
        proba[:, 0] = 0.75
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    monkeypatch.setattr(transfer_mod, "_tabpfn_predict_proba", fake_predict_proba)
    X_val = pd.DataFrame({"x0": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]})
    y_val = np.array([0, 0, 0, 1, 1, 1])
    X_test = pd.DataFrame({"x0": [-1.5, -0.25, 0.25, 1.5]})
    y_test = np.array([0, 0, 1, 1])
    dataset = DatasetBundle(
        X_train=X_val,
        y_train=y_val,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=["x0"],
        medians=X_val.median(),
    )
    output_dir = tmp_path / "scan"
    output_dir.mkdir()
    records_path = output_dir / "demo_context_scan_metrics.json"
    existing = {
        "context_size": 2,
        "requested_context_size": 2,
        "repeat": 0,
        "status": "ok",
        "baseline_tabpfn": {"accuracy": 0.5, "log_loss": 1.0, "roc_auc": 0.5},
        "frozen_encoder_tabpfn": {"accuracy": 0.5, "log_loss": 1.0, "roc_auc": 0.5},
        "delta": {"accuracy": 0.0, "log_loss": 0.0, "roc_auc": 0.0},
    }
    records_path.write_text(json.dumps([existing]), encoding="utf-8")

    metrics = run_encoder_context_scan_evaluation(
        trained=FakeEncoder(),
        dataset=dataset,
        output_dir=output_dir,
        context_min_per_class=1,
        context_scan_points=2,
        context_repeats=2,
        max_context_size=4,
        query_chunk_size=2,
        device="cpu",
        random_state=11,
        name="demo",
    )

    saved_records = json.loads(records_path.read_text(encoding="utf-8"))
    assert metrics["status"] == "complete"
    assert len(saved_records) == 4
    assert len(calls) == 6
    assert (output_dir / "demo_metrics.json").exists()
    assert (output_dir / "demo_context_scan_metrics.csv").exists()


def test_encoder_comparison_plots_save_pdf_outputs(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    csv_dir = tmp_path / "source_residual_mlp" / "source_generalization"
    csv_dir.mkdir(parents=True)
    (csv_dir / "source_12_class_generalization_context_scan_metrics.csv").write_text(
        "\n".join(
            [
                "context_size,status,baseline_accuracy,baseline_roc_auc,"
                "frozen_encoder_accuracy,frozen_encoder_roc_auc",
                "100,ok,0.50,0.55,0.60,0.65",
                "100,ok,0.52,0.57,0.62,0.67",
                "200,ok,0.54,0.59,0.64,0.69",
                "200,ok,0.56,0.61,0.66,0.71",
            ]
        ),
        encoding="utf-8",
    )

    saved = save_encoder_comparison_plots(
        [("MLP encoder", tmp_path / "source_residual_mlp")],
        tmp_path / "plots",
    )

    paths = saved["source_12_class_generalization"]
    assert {path.suffix for path in paths} == {".pdf"}
    assert {path.name for path in paths} == {
        "source_12_class_generalization_roc_auc_comparison.pdf",
        "source_12_class_generalization_accuracy_comparison.pdf",
    }
    assert all(path.exists() for path in paths)
