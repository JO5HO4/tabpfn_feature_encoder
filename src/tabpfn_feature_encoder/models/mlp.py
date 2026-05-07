from __future__ import annotations

from typing import Any

from tabpfn_feature_encoder.models.torch_utils import TorchModule, require_torch


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
