from __future__ import annotations

from typing import Any

from tabpfn_feature_encoder.models.feature_gate import FeatureGateEncoder
from tabpfn_feature_encoder.models.feature_mixer import FeatureMixerEncoder
from tabpfn_feature_encoder.models.gnn import LightweightGNNEncoder
from tabpfn_feature_encoder.models.mlp import MLPEncoder
from tabpfn_feature_encoder.models.transformer import ParticleTransformerEncoder


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
