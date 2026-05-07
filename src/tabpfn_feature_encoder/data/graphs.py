from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraphBatch:
    node_features: Any
    batch_index: Any
    global_features: Any
    n_graphs: int


@dataclass
class EventGraphDataset:
    nodes: list[np.ndarray]
    global_features: np.ndarray
    node_feature_names: list[str]
    global_feature_names: list[str]

    def __post_init__(self) -> None:
        n_events = len(self.nodes)
        globals_arr = np.asarray(self.global_features, dtype=np.float32)
        if globals_arr.ndim == 1:
            globals_arr = globals_arr.reshape(n_events, -1)
        if len(globals_arr) != n_events:
            raise ValueError("global_features length must match nodes length.")
        self.global_features = np.ascontiguousarray(globals_arr, dtype=np.float32)
        self.nodes = [
            np.ascontiguousarray(node, dtype=np.float32).reshape(-1, self.node_dim)
            for node in self.nodes
        ]

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def node_dim(self) -> int:
        return len(self.node_feature_names)

    @property
    def global_dim(self) -> int:
        return len(self.global_feature_names)

    def subset(self, indices: np.ndarray) -> "EventGraphDataset":
        idx = np.asarray(indices, dtype=np.int64)
        return EventGraphDataset(
            nodes=[self.nodes[int(i)] for i in idx],
            global_features=self.global_features[idx],
            node_feature_names=list(self.node_feature_names),
            global_feature_names=list(self.global_feature_names),
        )

    def to_batch(self, indices: np.ndarray, *, device: str) -> GraphBatch:
        from tabpfn_feature_encoder.models.encoders import require_torch

        torch_mod, _ = require_torch()
        idx = np.asarray(indices, dtype=np.int64)
        node_parts: list[Any] = []
        batch_parts: list[Any] = []
        for graph_idx, event_idx in enumerate(idx):
            node_arr = self.nodes[int(event_idx)]
            if len(node_arr) == 0:
                continue
            node_tensor = torch_mod.tensor(node_arr, dtype=torch_mod.float32, device=device)
            node_parts.append(node_tensor)
            batch_parts.append(
                torch_mod.full(
                    (len(node_arr),),
                    int(graph_idx),
                    dtype=torch_mod.long,
                    device=device,
                )
            )

        if node_parts:
            node_features = torch_mod.cat(node_parts, dim=0)
            batch_index = torch_mod.cat(batch_parts, dim=0)
        else:
            node_features = torch_mod.zeros(
                (0, self.node_dim),
                dtype=torch_mod.float32,
                device=device,
            )
            batch_index = torch_mod.zeros((0,), dtype=torch_mod.long, device=device)

        global_features = torch_mod.tensor(
            self.global_features[idx],
            dtype=torch_mod.float32,
            device=device,
        )
        return GraphBatch(
            node_features=node_features,
            batch_index=batch_index,
            global_features=global_features,
            n_graphs=int(len(idx)),
        )

    @classmethod
    def concat(cls, datasets: list["EventGraphDataset"]) -> "EventGraphDataset":
        if not datasets:
            raise ValueError("Need at least one EventGraphDataset to concatenate.")
        node_names = datasets[0].node_feature_names
        global_names = datasets[0].global_feature_names
        nodes: list[np.ndarray] = []
        globals_parts: list[np.ndarray] = []
        for dataset in datasets:
            if dataset.node_feature_names != node_names:
                raise ValueError("Cannot concatenate graph datasets with different node features.")
            if dataset.global_feature_names != global_names:
                raise ValueError("Cannot concatenate graph datasets with different global features.")
            nodes.extend(dataset.nodes)
            globals_parts.append(dataset.global_features)
        return cls(
            nodes=nodes,
            global_features=np.concatenate(globals_parts, axis=0),
            node_feature_names=list(node_names),
            global_feature_names=list(global_names),
        )


@dataclass
class GraphStandardizer:
    node_mean_: np.ndarray | None = None
    node_scale_: np.ndarray | None = None
    global_mean_: np.ndarray | None = None
    global_scale_: np.ndarray | None = None
    node_passthrough_mask_: np.ndarray | None = None

    def fit(self, dataset: EventGraphDataset) -> "GraphStandardizer":
        if dataset.nodes:
            nonempty = [node for node in dataset.nodes if len(node) > 0]
        else:
            nonempty = []
        if nonempty:
            stacked_nodes = np.concatenate(nonempty, axis=0).astype(np.float32)
            self.node_mean_ = np.nanmean(stacked_nodes, axis=0).astype(np.float32)
            self.node_scale_ = np.nanstd(stacked_nodes, axis=0).astype(np.float32)
        else:
            self.node_mean_ = np.zeros(dataset.node_dim, dtype=np.float32)
            self.node_scale_ = np.ones(dataset.node_dim, dtype=np.float32)

        if dataset.global_dim > 0:
            self.global_mean_ = np.nanmean(dataset.global_features, axis=0).astype(np.float32)
            self.global_scale_ = np.nanstd(dataset.global_features, axis=0).astype(np.float32)
        else:
            self.global_mean_ = np.zeros(0, dtype=np.float32)
            self.global_scale_ = np.ones(0, dtype=np.float32)

        self.node_scale_ = np.where(self.node_scale_ == 0.0, 1.0, self.node_scale_)
        self.global_scale_ = np.where(self.global_scale_ == 0.0, 1.0, self.global_scale_)
        self.node_passthrough_mask_ = np.asarray(
            [name.startswith("type_") for name in dataset.node_feature_names],
            dtype=bool,
        )
        self.node_mean_ = np.where(self.node_passthrough_mask_, 0.0, self.node_mean_)
        self.node_scale_ = np.where(self.node_passthrough_mask_, 1.0, self.node_scale_)
        return self

    def transform(self, dataset: EventGraphDataset) -> EventGraphDataset:
        if (
            self.node_mean_ is None
            or self.node_scale_ is None
            or self.global_mean_ is None
            or self.global_scale_ is None
        ):
            raise RuntimeError("GraphStandardizer.fit must be called before transform.")

        nodes = [
            np.nan_to_num((node - self.node_mean_) / self.node_scale_).astype(np.float32)
            for node in dataset.nodes
        ]
        globals_arr = np.nan_to_num(
            (dataset.global_features - self.global_mean_) / self.global_scale_
        ).astype(np.float32)
        return EventGraphDataset(
            nodes=nodes,
            global_features=globals_arr,
            node_feature_names=list(dataset.node_feature_names),
            global_feature_names=list(dataset.global_feature_names),
        )

    def fit_transform(self, dataset: EventGraphDataset) -> EventGraphDataset:
        return self.fit(dataset).transform(dataset)
