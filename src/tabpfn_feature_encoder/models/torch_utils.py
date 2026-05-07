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
