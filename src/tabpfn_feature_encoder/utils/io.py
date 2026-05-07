from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    tmp_path = _tmp_path(path)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_pickle(payload: Any, path: str | Path) -> None:
    path = Path(path)
    tmp_path = _tmp_path(path)
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="list")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
