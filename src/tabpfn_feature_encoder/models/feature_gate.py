from __future__ import annotations

from typing import Any

from tabpfn_feature_encoder.models.torch_utils import TorchModule, require_torch


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
