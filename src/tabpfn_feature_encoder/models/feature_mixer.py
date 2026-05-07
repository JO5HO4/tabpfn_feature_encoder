from __future__ import annotations

from typing import Any

from tabpfn_feature_encoder.models.torch_utils import TorchModule, require_torch


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
