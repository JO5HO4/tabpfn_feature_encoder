from __future__ import annotations

from typing import Any

from tabpfn_feature_encoder.models.gnn import LightweightGNNEncoder
from tabpfn_feature_encoder.models.torch_utils import TorchModule, require_torch


class ParticleTransformerEncoder(TorchModule):
    """A variable-particle event transformer with fixed event summaries."""

    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 3,
        attention_heads: int = 4,
    ) -> None:
        torch_mod, nn_mod = require_torch()
        super().__init__()
        self.encoder_type = "transformer"
        self.input_dim = int(node_dim)
        self.global_dim = int(global_dim)
        self.output_dim = int(output_dim)
        self.uses_identity_residual = False
        self.residual_scale = 1.0
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("encoder.hidden_dim must be divisible by encoder.attention_heads.")

        self.summary_dim = LightweightGNNEncoder._summary_dim(self.input_dim, self.global_dim)
        self.summary_direct = self.output_dim > self.summary_dim
        if self.summary_direct:
            self.learned_dim = self.output_dim - self.summary_dim
            self.summary_output_dim = self.summary_dim
        else:
            self.summary_output_dim = max(1, self.output_dim // 2)
            self.learned_dim = self.output_dim - self.summary_output_dim

        self.node_projection = nn_mod.Linear(self.input_dim, self.hidden_dim)
        self.cls_token = nn_mod.Parameter(torch_mod.zeros(1, 1, self.hidden_dim))
        self.global_projection = (
            nn_mod.Linear(self.global_dim, self.hidden_dim)
            if self.global_dim > 0
            else None
        )
        layer = nn_mod.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attention_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn_mod.TransformerEncoder(
            layer,
            num_layers=max(1, int(layers)),
            norm=nn_mod.LayerNorm(self.hidden_dim),
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
        padded_nodes, padding_mask = self._pad_nodes(node_features, batch_index, n_graphs)

        if padded_nodes.shape[1] == 0:
            node_tokens = padded_nodes.new_zeros((n_graphs, 0, self.hidden_dim))
        else:
            node_tokens = self.node_projection(padded_nodes)

        cls = self.cls_token.expand(n_graphs, -1, -1)
        if self.global_projection is not None:
            cls = cls + self.global_projection(batch.global_features).unsqueeze(1)
        tokens = torch_mod.cat([cls, node_tokens], dim=1)
        cls_mask = padding_mask.new_zeros((n_graphs, 1))
        mask = torch_mod.cat([cls_mask, padding_mask], dim=1)

        encoded = self.transformer(tokens, src_key_padding_mask=mask)
        cls_hidden = encoded[:, 0]
        node_hidden = encoded[:, 1:]
        pooled_mean = self._masked_mean(node_hidden, padding_mask)
        pooled_max = self._masked_max(node_hidden, padding_mask)
        hidden = torch_mod.cat([cls_hidden, pooled_mean, pooled_max], dim=1)

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
        return LightweightGNNEncoder.summary_features(self, batch)

    @staticmethod
    def _type_start() -> int:
        return LightweightGNNEncoder._type_start()

    def _type_dim(self) -> int:
        return max(0, self.input_dim - self._type_start())

    def _base_dim(self) -> int:
        return min(self._type_start(), self.input_dim)

    def _per_type_summary_dim(self) -> int:
        return self._type_dim() * 2 * self._base_dim()

    def _per_type_summary(self, node_features: Any, batch_index: Any, n_graphs: int) -> Any:
        return LightweightGNNEncoder._per_type_summary(
            self,
            node_features,
            batch_index,
            n_graphs,
        )

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

    def _pad_nodes(self, node_features: Any, batch_index: Any, n_graphs: int) -> tuple[Any, Any]:
        torch_mod, _ = require_torch()
        counts = torch_mod.bincount(batch_index, minlength=n_graphs)
        max_nodes = int(counts.max().item()) if len(counts) else 0
        padded = node_features.new_zeros((n_graphs, max_nodes, self.input_dim))
        mask = torch_mod.ones(
            (n_graphs, max_nodes),
            dtype=torch_mod.bool,
            device=node_features.device,
        )
        for graph_idx in range(n_graphs):
            selected = batch_index == graph_idx
            n_nodes = int(selected.sum().item())
            if n_nodes == 0:
                continue
            padded[graph_idx, :n_nodes] = node_features[selected]
            mask[graph_idx, :n_nodes] = False
        return padded, mask

    @staticmethod
    def _masked_mean(values: Any, mask: Any) -> Any:
        if values.shape[1] == 0:
            return values.new_zeros((values.shape[0], values.shape[2]))
        keep = (~mask).to(values.dtype).unsqueeze(-1)
        summed = (values * keep).sum(dim=1)
        counts = keep.sum(dim=1).clamp_min(1.0)
        return summed / counts

    @staticmethod
    def _masked_max(values: Any, mask: Any) -> Any:
        if values.shape[1] == 0:
            return values.new_zeros((values.shape[0], values.shape[2]))
        masked = values.masked_fill(mask.unsqueeze(-1), float("-inf"))
        return masked.max(dim=1).values.nan_to_num(neginf=0.0)
