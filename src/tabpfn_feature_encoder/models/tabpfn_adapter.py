from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabpfn_feature_encoder.models.torch_utils import require_torch


@dataclass
class TabPFNPromptAdapter:
    """Small boundary around TabPFN's differentiable prompt API."""

    device: str
    model_path: str | Path | None = None
    random_state: int | None = None
    model: Any = None
    _prior_frozen: bool = False

    def build(self) -> "TabPFNPromptAdapter":
        if self.model is not None:
            return self
        torch_mod, _ = require_torch()
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError(
                "tabpfn is required for encoder training. Install with "
                "`python -m pip install -e '.[train]'`."
            ) from exc

        model_path = self.model_path or os.environ.get("TABPFN_MODEL_PATH") or "auto"
        self.model = TabPFNClassifier(
            device=self.device,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            ignore_pretraining_limits=True,
            n_estimators=1,
            inference_precision=torch_mod.float32,
            model_path=model_path,
            random_state=0 if self.random_state is None else int(self.random_state),
        )
        return self

    def fit_prompt(self, z_support: Any, y_support: Any) -> None:
        self._require_model()
        self.model.fit_with_differentiable_input(z_support.contiguous(), y_support.contiguous())
        self.freeze_prior()

    def forward_logits(self, z_query: Any) -> Any:
        self._require_model()
        return self.model.forward(
            z_query.contiguous(),
            use_inference_mode=True,
            return_logits=True,
        )

    def predict_proba(self, z_query: Any) -> Any:
        self._require_model()
        return self.model.predict_proba(z_query.contiguous())

    def clear_prompt(self) -> None:
        if self.model is None:
            return
        for attr in ("executor_", "ensemble_preprocessor_"):
            if hasattr(self.model, attr):
                delattr(self.model, attr)

    def freeze_prior(self) -> None:
        if self._prior_frozen or self.model is None or not hasattr(self.model, "models_"):
            return
        for base_model in self.model.models_:
            for param in base_model.parameters():
                param.requires_grad = False
        self._prior_frozen = True

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("TabPFNPromptAdapter.build() must be called before use.")
