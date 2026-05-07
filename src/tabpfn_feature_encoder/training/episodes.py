from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tabpfn_feature_encoder.data.preprocessing import stratified_sample_indices


@dataclass(frozen=True)
class Episode:
    support_idx: np.ndarray
    query_idx: np.ndarray


@dataclass
class RatioEpisodeSampler:
    support_query_ratio: float = 0.5
    random_state: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.support_query_ratio < 1.0:
            raise ValueError("support_query_ratio must be between 0 and 1.")
        self.rng = np.random.default_rng(self.random_state)

    def sample(self, y: np.ndarray) -> Episode:
        y = np.asarray(y)
        n_samples = len(y)
        _, counts = np.unique(y, return_counts=True)
        n_classes = len(counts)
        if n_samples < 2 * n_classes:
            raise ValueError("Need at least two examples per class to form support/query episodes.")
        if np.min(counts) < 2:
            raise ValueError("Need at least two examples per class to split context/query.")

        support_size = int(round(n_samples * self.support_query_ratio))
        support_size = min(max(n_classes, support_size), n_samples - n_classes)
        support_idx = stratified_sample_indices(y, support_size, rng=self.rng)

        query_mask = np.ones(n_samples, dtype=bool)
        query_mask[support_idx] = False
        query_idx = np.flatnonzero(query_mask).astype(np.int64)
        self.rng.shuffle(query_idx)
        return Episode(support_idx=support_idx.astype(np.int64), query_idx=query_idx)
