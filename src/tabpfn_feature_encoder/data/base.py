from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    feature_names: list[str]
    medians: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)
    graph_train: Any | None = None
    graph_val: Any | None = None
    graph_test: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "feature_names": self.feature_names,
            "medians": self.medians,
            "metadata": self.metadata,
            "graph_train": self.graph_train,
            "graph_val": self.graph_val,
            "graph_test": self.graph_test,
        }


def to_numpy_matrix(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame | pd.Series):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {arr.shape}.")
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Feature matrix contains NaN or inf.")
    return np.ascontiguousarray(arr)


def to_label_vector(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.int64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D label vector, got shape {arr.shape}.")
    return np.ascontiguousarray(arr)
