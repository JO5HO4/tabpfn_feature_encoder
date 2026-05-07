from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for encoder training. Install with "
            "`python -m pip install -e '.[train]'`."
        )
    return torch, nn


TorchModule = nn.Module if nn is not None else object


class MLPEncoder(TorchModule):
    """A small encoder that maps raw tabular features into TabPFN input features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 4,
        residual_scale: float = 0.1,
    ) -> None:
        _, nn_mod = require_torch()
        super().__init__()
        self.encoder_type = "residual_mlp"
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.uses_identity_residual = self.input_dim == self.output_dim
        self.residual_scale = float(residual_scale) if self.uses_identity_residual else 1.0
        n_layers = max(2, int(layers))
        modules: list[Any] = [
            nn_mod.Linear(self.input_dim, int(hidden_dim)),
            nn_mod.GELU(),
        ]
        for _ in range(n_layers - 2):
            modules.extend(
                [
                    nn_mod.Linear(int(hidden_dim), int(hidden_dim)),
                    nn_mod.GELU(),
                ]
            )
        final = nn_mod.Linear(int(hidden_dim), self.output_dim)
        if self.uses_identity_residual:
            nn_mod.init.zeros_(final.weight)
            nn_mod.init.zeros_(final.bias)
        modules.append(final)
        self.net = nn_mod.Sequential(*modules)

    def forward(self, x: Any) -> Any:
        encoded = self.net(x)
        if self.uses_identity_residual:
            return x + self.residual_scale * encoded
        return encoded


class FeatureGateEncoder(TorchModule):
    """A stable per-feature gate initialized to the raw standardized features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        residual_scale: float = 0.1,
    ) -> None:
        torch_mod, nn_mod = require_torch()
        super().__init__()
        self.encoder_type = "feature_gate"
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.output_dim != self.input_dim:
            raise ValueError("feature_gate requires encoder.output_dim to equal input_dim.")
        self.uses_identity_residual = True
        self.residual_scale = float(residual_scale)
        self.raw_gate = nn_mod.Parameter(torch_mod.zeros(self.input_dim))

    def forward(self, x: Any) -> Any:
        torch_mod, _ = require_torch()
        gate = 1.0 + self.residual_scale * torch_mod.tanh(self.raw_gate)
        return x * gate


class FeatureMixerEncoder(TorchModule):
    """A linear residual feature mixer initialized to the raw standardized features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        residual_scale: float = 0.1,
    ) -> None:
        _, nn_mod = require_torch()
        super().__init__()
        self.encoder_type = "feature_mixer"
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.output_dim != self.input_dim:
            raise ValueError("feature_mixer requires encoder.output_dim to equal input_dim.")
        self.uses_identity_residual = True
        self.residual_scale = float(residual_scale)
        self.mixer = nn_mod.Linear(self.input_dim, self.output_dim)
        nn_mod.init.zeros_(self.mixer.weight)
        nn_mod.init.zeros_(self.mixer.bias)

    def forward(self, x: Any) -> Any:
        return x + self.residual_scale * self.mixer(x)


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


def build_encoder(
    *,
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    residual_scale: float,
    attention_heads: int = 4,
    global_dim: int = 0,
) -> Any:
    kind = encoder_type.lower()
    if kind in {"feature_gate", "gate"}:
        return FeatureGateEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            residual_scale=residual_scale,
        )
    if kind in {"feature_mixer", "linear_residual", "residual_linear"}:
        return FeatureMixerEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            residual_scale=residual_scale,
        )
    if kind in {"residual", "residual_mlp"}:
        if output_dim != input_dim:
            raise ValueError(
                "residual_mlp requires encoder.output_dim to equal the flat input feature count."
            )
        return MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layers=layers,
            residual_scale=residual_scale,
        )
    if kind == "mlp":
        return MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layers=layers,
            residual_scale=residual_scale,
        )
    if kind in {"gnn", "graph", "graph_gnn"}:
        return LightweightGNNEncoder(
            node_dim=input_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layers=layers,
        )
    if kind in {"transformer", "particle_transformer", "graph_transformer"}:
        return ParticleTransformerEncoder(
            node_dim=input_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layers=layers,
            attention_heads=attention_heads,
        )
    raise ValueError(
        "encoder.type must be `feature_gate`, `feature_mixer`, "
        "`residual_mlp`, `gnn`, or `transformer`."
    )
