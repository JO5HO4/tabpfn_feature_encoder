from __future__ import annotations

import numpy as np
import pandas as pd

from tabpfn_feature_encoder.config import EncoderConfig
from tabpfn_feature_encoder.data.base import DatasetBundle
from tabpfn_feature_encoder.evaluation import benchmark


def _proba_from_first_feature(X: np.ndarray) -> np.ndarray:
    scores = X[:, 0]
    p1 = 1.0 / (1.0 + np.exp(-scores))
    return np.column_stack([1.0 - p1, p1])


class FakeEncoderTabPFN:
    X_train_ = np.zeros((4, 2), dtype=np.float32)

    def predict_proba_with_context(
        self,
        X_context,
        y_context,
        X_query,
        *,
        query_chunk_size=None,
    ):
        del X_context, y_context, query_chunk_size
        return _proba_from_first_feature(X_query.to_numpy(dtype=np.float32))


class FakeEncoderOnlyClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.fit_payload = (X_train, y_train, X_val, y_val)
        return self

    def predict_proba(self, X):
        return _proba_from_first_feature(X.to_numpy(dtype=np.float32))

    def metrics_from_proba(self, y, proba):
        pred = np.argmax(proba, axis=1)
        return {
            "accuracy": float(np.mean(pred == np.asarray(y))),
            "roc_auc": 1.0,
            "log_loss": 0.1,
        }


def test_nominal_benchmark_uses_shared_split_and_writes_outputs(tmp_path, monkeypatch) -> None:
    X_train = pd.DataFrame({"x0": [-2.0, -1.0, 1.0, 2.0], "x1": [0.0, 0.0, 0.0, 0.0]})
    X_val = pd.DataFrame({"x0": [-1.5, 1.5], "x1": [0.0, 0.0]})
    X_test = pd.DataFrame({"x0": [-3.0, 3.0, -2.5, 2.5], "x1": [0.0, 0.0, 0.0, 0.0]})
    bundle = DatasetBundle(
        X_train=X_train,
        y_train=np.array([0, 0, 1, 1]),
        X_val=X_val,
        y_val=np.array([0, 1]),
        X_test=X_test,
        y_test=np.array([0, 1, 0, 1]),
        feature_names=["x0", "x1"],
        medians=pd.Series({"x0": 0.0, "x1": 0.0}),
    )

    monkeypatch.setattr(
        benchmark,
        "_tabpfn_predict_proba",
        lambda X_context, y_context, X_query, query_chunk_size, device: _proba_from_first_feature(
            X_query
        ),
    )
    monkeypatch.setattr(benchmark, "EncoderOnlyClassifier", FakeEncoderOnlyClassifier)

    results = benchmark.run_nominal_benchmarks(
        dataset=bundle,
        trained_model=FakeEncoderTabPFN(),
        encoder_config=EncoderConfig(
            type="residual_mlp",
            hidden_dim=4,
            output_dim=2,
            batch_size=4,
            support_query_ratio=0.5,
            epochs=1,
        ),
        output_dir=tmp_path,
        device="cpu",
        random_state=7,
    )

    assert results["context_size"] == 2
    assert results["query_chunk_size"] == 2
    assert results["test_size"] == 4
    assert results["baseline_tabpfn"]["roc_auc"] == 1.0
    assert results["encoder_tabpfn"]["accuracy"] == 1.0
    assert results["encoder_only_classifier"]["accuracy"] == 1.0
    assert (tmp_path / "benchmark_metrics.json").exists()
    assert (tmp_path / "benchmark_baseline_tabpfn_proba.npy").exists()
    assert (tmp_path / "benchmark_encoder_tabpfn_proba.npy").exists()
    assert (tmp_path / "benchmark_encoder_only_proba.npy").exists()
