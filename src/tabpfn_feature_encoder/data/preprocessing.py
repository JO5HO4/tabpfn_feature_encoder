from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, X: Any) -> "Standardizer":
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std[std < self.eps] = 1.0
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer has not been fitted.")
        arr = np.asarray(X, dtype=np.float32)
        return ((arr - self.mean_) / self.std_).astype(np.float32, copy=False)

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class MedianImputer:
    medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        self.medians_ = (
            X.replace([np.inf, -np.inf], np.nan)
            .median(numeric_only=True)
            .fillna(0.0)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("MedianImputer has not been fitted.")
        return X.replace([np.inf, -np.inf], np.nan).fillna(self.medians_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def stratified_split_indices(
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(y):
        label_idx = np.flatnonzero(y == label)
        rng.shuffle(label_idx)
        n_test = int(round(len(label_idx) * test_size))
        if len(label_idx) > 1:
            n_test = min(max(1, n_test), len(label_idx) - 1)
        else:
            n_test = 0
        test_parts.append(label_idx[:n_test])
        train_parts.append(label_idx[n_test:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def stratified_sample_indices(
    y: np.ndarray,
    n_samples: int,
    *,
    random_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    y = np.asarray(y)
    n_samples = int(n_samples)
    if n_samples < 0 or n_samples > len(y):
        raise ValueError("n_samples must be in [0, len(y)].")
    if n_samples == 0:
        return np.array([], dtype=np.int64)

    rng = rng or np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if n_samples < len(classes):
        raise ValueError(
            "n_samples must be at least the number of classes for stratified sampling."
        )

    raw = n_samples * (counts / counts.sum())
    allocation = np.floor(raw).astype(int)
    allocation = np.minimum(allocation, counts)
    allocation[counts > 0] = np.maximum(allocation[counts > 0], 1)

    while allocation.sum() > n_samples:
        candidates = np.flatnonzero(allocation > 1)
        if len(candidates) == 0:
            break
        idx = candidates[np.argmin(raw[candidates] - np.floor(raw[candidates]))]
        allocation[idx] -= 1

    while allocation.sum() < n_samples:
        capacity = counts - allocation
        candidates = np.flatnonzero(capacity > 0)
        if len(candidates) == 0:
            break
        deficit_score = raw[candidates] - allocation[candidates]
        idx = candidates[np.argmax(deficit_score)]
        allocation[idx] += 1

    selected: list[np.ndarray] = []
    for label, n_take in zip(classes, allocation):
        label_idx = np.flatnonzero(y == label)
        selected.append(rng.choice(label_idx, size=int(n_take), replace=False))

    out = np.concatenate(selected).astype(np.int64)
    rng.shuffle(out)
    return out
