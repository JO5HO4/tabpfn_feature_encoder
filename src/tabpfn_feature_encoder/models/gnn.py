from __future__ import annotations

from typing import Any

from tabpfn_feature_encoder.models.torch_utils import TorchModule, require_torch


class LightweightGNNEncoder(TorchModule):
    """A small variable-node event encoder for particle graphs."""

    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 2,
    ) -> None:
        _, nn_mod = require_torch()
        super().__init__()
        self.encoder_type = "gnn"
        self.input_dim = int(node_dim)
        self.global_dim = int(global_dim)
        self.output_dim = int(output_dim)
        self.uses_identity_residual = False
        self.residual_scale = 1.0
        self.hidden_dim = int(hidden_dim)
        self.summary_dim = self._summary_dim(self.input_dim, self.global_dim)
        self.summary_direct = self.output_dim > self.summary_dim
        if self.summary_direct:
            self.learned_dim = self.output_dim - self.summary_dim
            self.summary_output_dim = self.summary_dim
        else:
            self.summary_output_dim = max(1, self.output_dim // 2)
            self.learned_dim = self.output_dim - self.summary_output_dim
        n_layers = max(1, int(layers))

        self.node_mlp = nn_mod.Sequential(
            nn_mod.Linear(self.input_dim, self.hidden_dim),
            nn_mod.GELU(),
            nn_mod.Linear(self.hidden_dim, self.hidden_dim),
            nn_mod.GELU(),
        )
        self.message_layers = nn_mod.ModuleList(
            [
                nn_mod.Sequential(
                    nn_mod.Linear(2 * self.hidden_dim, self.hidden_dim),
                    nn_mod.GELU(),
                    nn_mod.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.message_norms = nn_mod.ModuleList(
            [nn_mod.LayerNorm(self.hidden_dim) for _ in range(n_layers)]
        )
        self.global_mlp = (
            nn_mod.Sequential(
                nn_mod.Linear(self.global_dim, self.hidden_dim),
                nn_mod.GELU(),
            )
            if self.global_dim > 0
            else None
        )
        self.out = (
            nn_mod.Sequential(
                nn_mod.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn_mod.GELU(),
                nn_mod.Linear(self.hidden_dim, self.learned_dim),
            )
            if self.learned_dim > 0
            else None
        )
        self.summary_norm = nn_mod.Identity()
        self.summary_projector = (
            None
            if self.summary_direct
            else nn_mod.Sequential(
                nn_mod.Linear(self.summary_dim, self.summary_output_dim),
                nn_mod.GELU(),
                nn_mod.Linear(self.summary_output_dim, self.summary_output_dim),
            )
        )

    def forward(self, batch: Any) -> Any:
        torch_mod, _ = require_torch()
        node_features = batch.node_features
        batch_index = batch.batch_index
        n_graphs = int(batch.n_graphs)

        if node_features.shape[0] == 0:
            node_hidden = node_features.new_zeros((0, self.hidden_dim))
        else:
            node_hidden = self.node_mlp(node_features)

        for message_layer, norm in zip(self.message_layers, self.message_norms, strict=True):
            pooled = _scatter_mean(node_hidden, batch_index, n_graphs)
            messages = node_hidden if node_hidden.shape[0] == 0 else pooled[batch_index]
            update = message_layer(torch_mod.cat([node_hidden, messages], dim=1))
            node_hidden = norm(node_hidden + update)

        pooled_mean = _scatter_mean(node_hidden, batch_index, n_graphs)
        pooled_max = _scatter_max(node_hidden, batch_index, n_graphs)
        if self.global_mlp is None:
            global_hidden = pooled_mean.new_zeros((n_graphs, self.hidden_dim))
        else:
            global_hidden = self.global_mlp(batch.global_features)
        hidden = torch_mod.cat([pooled_mean, pooled_max, global_hidden], dim=1)
        if not hasattr(self, "summary_dim"):
            return self.out(hidden)

        summary = self._adjust_summary_width(self.summary_features(batch))
        summary = self.summary_norm(summary)
        if self.summary_projector is not None:
            summary = self.summary_projector(summary)
        parts = []
        if self.out is not None:
            parts.append(self.out(hidden))
        parts.append(summary)
        return torch_mod.cat(parts, dim=1)

    def summary_features(self, batch: Any) -> Any:
        torch_mod, _ = require_torch()
        node_features = batch.node_features
        batch_index = batch.batch_index
        n_graphs = int(batch.n_graphs)

        if node_features.shape[0] == 0:
            node_mean = node_features.new_zeros((n_graphs, self.input_dim))
            node_max = node_features.new_zeros((n_graphs, self.input_dim))
            node_abs_mean = node_features.new_zeros((n_graphs, self.input_dim))
            counts = node_features.new_zeros((n_graphs, 1))
            type_counts = node_features.new_zeros((n_graphs, self._type_dim()))
            per_type_summary = node_features.new_zeros(
                (n_graphs, self._per_type_summary_dim())
            )
        else:
            node_mean = _scatter_mean(node_features, batch_index, n_graphs)
            node_max = _scatter_max(node_features, batch_index, n_graphs)
            node_abs_mean = _scatter_mean(node_features.abs(), batch_index, n_graphs)
            counts = torch_mod.bincount(batch_index, minlength=n_graphs).to(
                node_features.dtype
            ).unsqueeze(1)
            if self._type_dim() > 0:
                type_values = node_features[:, self._type_start() :].clamp_min(0.0)
                type_counts = _scatter_sum(type_values, batch_index, n_graphs)
                per_type_summary = self._per_type_summary(
                    node_features,
                    batch_index,
                    n_graphs,
                )
            else:
                type_counts = node_features.new_zeros((n_graphs, 0))
                per_type_summary = node_features.new_zeros((n_graphs, 0))

        return torch_mod.cat(
            [
                batch.global_features,
                torch_mod.log1p(counts),
                torch_mod.log1p(type_counts),
                node_mean,
                node_max,
                node_abs_mean,
                per_type_summary,
            ],
            dim=1,
        )

    def _per_type_summary(self, node_features: Any, batch_index: Any, n_graphs: int) -> Any:
        torch_mod, _ = require_torch()
        base_dim = self._base_dim()
        if self._type_dim() == 0 or base_dim == 0:
            return node_features.new_zeros((n_graphs, 0))

        base_values = node_features[:, :base_dim]
        type_values = node_features[:, self._type_start() :]
        parts = []
        for type_idx in range(self._type_dim()):
            selected = type_values[:, type_idx] > 0.5
            if bool(selected.any()):
                selected_base = base_values[selected]
                selected_batch = batch_index[selected]
                parts.append(_scatter_mean(selected_base, selected_batch, n_graphs))
                parts.append(_scatter_max(selected_base, selected_batch, n_graphs))
            else:
                empty = node_features.new_zeros((n_graphs, base_dim))
                parts.extend([empty, empty])
        return torch_mod.cat(parts, dim=1)

    @staticmethod
    def _summary_dim(node_dim: int, global_dim: int) -> int:
        type_dim = max(0, int(node_dim) - LightweightGNNEncoder._type_start())
        base_dim = min(LightweightGNNEncoder._type_start(), int(node_dim))
        per_type_dim = type_dim * 2 * base_dim
        return int(global_dim) + 1 + type_dim + 3 * int(node_dim) + per_type_dim

    @staticmethod
    def _type_start() -> int:
        return 6

    def _type_dim(self) -> int:
        return max(0, self.input_dim - self._type_start())

    def _base_dim(self) -> int:
        return min(self._type_start(), self.input_dim)

    def _per_type_summary_dim(self) -> int:
        return self._type_dim() * 2 * self._base_dim()

    def _adjust_summary_width(self, summary: Any) -> Any:
        if summary.shape[1] == self.summary_dim:
            return summary
        if summary.shape[1] > self.summary_dim:
            return summary[:, : self.summary_dim]
        torch_mod, _ = require_torch()
        pad_width = self.summary_dim - summary.shape[1]
        return torch_mod.cat(
            [summary, summary.new_zeros((summary.shape[0], pad_width))],
            dim=1,
        )


def _scatter_mean(values: Any, batch_index: Any, n_graphs: int) -> Any:
    torch_mod, _ = require_torch()
    if values.shape[0] == 0:
        return values.new_zeros((n_graphs, values.shape[1]))
    out = values.new_zeros((n_graphs, values.shape[1]))
    out.index_add_(0, batch_index, values)
    counts = torch_mod.bincount(batch_index, minlength=n_graphs).clamp_min(1)
    return out / counts.to(values.dtype).unsqueeze(1)


def _scatter_sum(values: Any, batch_index: Any, n_graphs: int) -> Any:
    if values.shape[0] == 0:
        return values.new_zeros((n_graphs, values.shape[1]))
    out = values.new_zeros((n_graphs, values.shape[1]))
    out.index_add_(0, batch_index, values)
    return out


def _scatter_max(values: Any, batch_index: Any, n_graphs: int) -> Any:
    if values.shape[0] == 0:
        return values.new_zeros((n_graphs, values.shape[1]))
    out = values.new_full((n_graphs, values.shape[1]), float("-inf"))
    expanded_index = batch_index.unsqueeze(1).expand(-1, values.shape[1])
    out.scatter_reduce_(0, expanded_index, values, reduce="amax", include_self=True)
    return out.nan_to_num(neginf=0.0)
