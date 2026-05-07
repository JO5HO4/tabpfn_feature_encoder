from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabpfn_feature_encoder.data.base import DatasetBundle
from tabpfn_feature_encoder.data.graphs import EventGraphDataset
from tabpfn_feature_encoder.data.parallel import (
    data_worker_count,
    detected_cpu_count,
    parallel_map,
)
from tabpfn_feature_encoder.data.preprocessing import (
    MedianImputer,
    stratified_split_indices,
)
from tabpfn_feature_encoder.utils.io import load_pickle, save_pickle


@dataclass(frozen=True)
class CPDatasetSettings:
    raw_dir: Path = Path("/global/cfs/projectdirs/atlas/joshua/gnn_data/stats_100K")
    tree_name: str = "output"
    even_file: str = "ttH_NLO.root"
    odd_file: str = "ttH_CPodd.root"
    train_fraction: float = 0.5
    val_fraction: float = 0.25
    test_fraction: float = 0.25
    cache_dir: Path | None = None
    use_cache: bool = True
    padding: str = "zero"
    scalar_cols: list[str] = field(default_factory=lambda: ["MET_met", "MET_phi"])
    jagged_maxlen: dict[str, int] = field(
        default_factory=lambda: {
            "jet_pt": 6,
            "jet_eta": 6,
            "jet_phi": 6,
            "jet_btag": 6,
            "ele_pt": 4,
            "ele_eta": 4,
            "ele_phi": 4,
            "ele_charge": 4,
            "mu_pt": 4,
            "mu_eta": 4,
            "mu_phi": 4,
            "mu_charge": 4,
            "ph_pt": 2,
            "ph_eta": 2,
            "ph_phi": 2,
        }
    )


def _require_uproot() -> Any:
    try:
        import uproot
    except ImportError as exc:
        raise ImportError(
            "uproot is required to read ROOT files. Install with "
            "`python -m pip install -e '.[atlas]'`."
        ) from exc
    return uproot


def _safe_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if np.isscalar(value):
        return [float(value)]
    try:
        return [float(item) for item in value]
    except TypeError:
        return []


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def _branch_suffix(branch: str) -> str:
    return str(branch).split("_")[-1].lower()


def _print_event_progress(
    progress_label: str | None,
    processed: int,
    total: int,
    progress_interval: int,
) -> None:
    if progress_label is None or progress_interval <= 0:
        return
    if processed % progress_interval == 0 or processed == total:
        print(
            f"{progress_label}: processed {processed}/{total} events",
            flush=True,
        )


def pad_jagged(series: pd.Series, max_len: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((len(series), int(max_len)), float(fill), dtype=np.float32)
    for row_idx, value in enumerate(series):
        values = _safe_sequence(value)
        if not values:
            continue
        clipped = np.asarray(values[:max_len], dtype=np.float32)
        out[row_idx, : len(clipped)] = clipped
    return out


@dataclass(frozen=True)
class RootEventLoader:
    tree_name: str = "output"

    def load(self, path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
        uproot = _require_uproot()
        with uproot.open(path) as root_file:
            return root_file[self.tree_name].arrays(columns, library="pd")

    def load_sample(
        self,
        path: str | Path,
        *,
        random_state: int,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        df = self.load(path, columns=columns)
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


@dataclass(frozen=True)
class AtlasFeatureBuilder:
    scalar_cols: list[str]
    jagged_maxlen: dict[str, int]
    padding: str = "zero"

    def input_columns(self) -> list[str]:
        return list(dict.fromkeys([*self.scalar_cols, *self.jagged_maxlen.keys()]))

    def build(self, df: pd.DataFrame, medians: pd.Series | None = None) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        padding_value = self._padding_value()

        for col in self.scalar_cols:
            if col in df.columns:
                part = pd.to_numeric(df[col], errors="coerce").to_frame(name=col)
            else:
                part = pd.DataFrame(
                    {col: np.zeros(len(df), dtype=np.float32)},
                    index=df.index,
                )
            parts.append(part)

        for col, max_len in self.jagged_maxlen.items():
            if col in df.columns:
                values = pad_jagged(df[col], max_len, fill=padding_value)
            else:
                values = np.full(
                    (len(df), int(max_len)),
                    padding_value,
                    dtype=np.float32,
                )
            names = [f"{col}_{idx}" for idx in range(int(max_len))]
            parts.append(pd.DataFrame(values, columns=names, index=df.index))

        if not parts:
            raise ValueError("No features were configured.")
        X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
        if medians is not None:
            X = X.fillna(medians)
        return X

    def _padding_value(self) -> float:
        padding = self.padding.lower()
        if padding == "zero":
            return 0.0
        if padding == "nan":
            return float("nan")
        raise ValueError("padding must be `zero` or `nan`.")


@dataclass(frozen=True)
class ParticleGraphSpec:
    name: str
    branches: list[str]


@dataclass(frozen=True)
class AtlasGraphBuilder:
    scalar_cols: list[str]
    particles: list[ParticleGraphSpec]

    def input_columns(self) -> list[str]:
        particle_cols = [
            branch
            for particle in self.particles
            for branch in particle.branches
        ]
        return list(dict.fromkeys([*self.scalar_cols, *particle_cols]))

    def node_feature_names(self) -> list[str]:
        type_features = [f"type_{particle.name}" for particle in self.particles]
        return [
            "log_pt",
            "eta",
            "sin_phi",
            "cos_phi",
            "charge",
            "btag",
            *type_features,
        ]

    def global_feature_names(self) -> list[str]:
        return list(self.scalar_cols)

    def build(
        self,
        df: pd.DataFrame,
        *,
        progress_label: str | None = None,
        progress_interval: int = 50_000,
    ) -> EventGraphDataset:
        node_names = self.node_feature_names()
        global_names = self.global_feature_names()
        global_features = self._build_globals(df, global_names)
        total = len(df)
        nodes: list[np.ndarray] = []
        for processed, (_, row) in enumerate(df.iterrows(), start=1):
            nodes.append(self._build_event_nodes(row, node_names))
            _print_event_progress(
                progress_label,
                processed,
                total,
                progress_interval,
            )
        return EventGraphDataset(
            nodes=nodes,
            global_features=global_features,
            node_feature_names=node_names,
            global_feature_names=global_names,
        )

    def _build_globals(self, df: pd.DataFrame, names: list[str]) -> np.ndarray:
        if not names:
            return np.zeros((len(df), 0), dtype=np.float32)
        parts = []
        for col in names:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            else:
                values = np.zeros(len(df), dtype=np.float32)
            parts.append(values.reshape(-1, 1))
        return np.nan_to_num(np.concatenate(parts, axis=1)).astype(np.float32)

    def _build_event_nodes(self, row: pd.Series, node_names: list[str]) -> np.ndarray:
        type_offset = 6
        out: list[np.ndarray] = []
        for type_idx, particle in enumerate(self.particles):
            sequences = {
                branch: _safe_sequence(row[branch]) if branch in row.index else []
                for branch in particle.branches
            }
            n_particles = max((len(values) for values in sequences.values()), default=0)
            for particle_idx in range(n_particles):
                node = np.zeros(len(node_names), dtype=np.float32)
                for branch, values in sequences.items():
                    if particle_idx >= len(values):
                        continue
                    value = _finite_float(values[particle_idx])
                    suffix = _branch_suffix(branch)
                    if suffix == "pt":
                        node[0] = np.log1p(max(value, 0.0))
                    elif suffix == "eta":
                        node[1] = value
                    elif suffix == "phi":
                        node[2] = np.sin(value)
                        node[3] = np.cos(value)
                    elif suffix == "charge":
                        node[4] = value
                    elif suffix in {"btag", "btagged"}:
                        node[5] = value
                node[type_offset + type_idx] = 1.0
                out.append(node)
        if not out:
            return np.zeros((0, len(node_names)), dtype=np.float32)
        return np.stack(out, axis=0).astype(np.float32)


@dataclass(frozen=True)
class _CPFileTask:
    raw_dir: Path
    filename: str
    label: int
    random_state: int
    tree_name: str
    columns: list[str]
    feature_builder: AtlasFeatureBuilder
    graph_builder: AtlasGraphBuilder | None


@dataclass(frozen=True)
class _CPFileResult:
    label: int
    filename: str
    features: pd.DataFrame
    graph: EventGraphDataset | None


def _load_cp_file_task(task: _CPFileTask) -> _CPFileResult:
    loader = RootEventLoader(tree_name=task.tree_name)
    path = task.raw_dir / task.filename
    print(f"Reading ROOT file: {path} (all rows)", flush=True)
    df = loader.load_sample(
        path,
        random_state=task.random_state,
        columns=task.columns,
    )
    features = task.feature_builder.build(df)
    graph = (
        task.graph_builder.build(
            df,
            progress_label=f"{task.filename} graph",
        )
        if task.graph_builder is not None
        else None
    )
    print(f"Prepared {len(features)} events from {task.filename}", flush=True)
    return _CPFileResult(
        label=task.label,
        filename=task.filename,
        features=features,
        graph=graph,
    )


@dataclass(frozen=True)
class CPDatasetBuilder:
    raw_dir: Path
    label_files: dict[int, list[str]]
    feature_builder: AtlasFeatureBuilder
    graph_builder: AtlasGraphBuilder | None
    train_fraction: float
    val_fraction: float
    test_fraction: float
    random_state: int
    tree_name: str = "output"
    cache_dir: Path | None = None
    use_cache: bool = True

    def build(self) -> DatasetBundle:
        cache_path = self._cache_path()
        if self.use_cache and cache_path is not None and cache_path.exists():
            print(f"Loading cached dataset: {cache_path}")
            cached = load_pickle(cache_path)
            if not isinstance(cached, DatasetBundle):
                raise TypeError(f"Cached object at {cache_path} is not a DatasetBundle.")
            return cached

        bundle = self._build_from_root()
        if self.use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(bundle, cache_path)
            print(f"Saved cached dataset: {cache_path}")
        return bundle

    def _build_from_root(self) -> DatasetBundle:
        X_parts: list[pd.DataFrame] = []
        graph_parts: list[EventGraphDataset] = []
        y_parts: list[np.ndarray] = []
        feature_names: list[str] | None = None

        print(f"Building dataset from ROOT files in {self.raw_dir}", flush=True)

        tasks: list[_CPFileTask] = []
        columns = self._input_columns()
        for label_idx, (label, files) in enumerate(sorted(self.label_files.items())):
            if not files:
                raise ValueError("Each label must have at least one input file.")
            for file_idx, filename in enumerate(files):
                tasks.append(
                    _CPFileTask(
                        raw_dir=self.raw_dir,
                        filename=filename,
                        label=int(label),
                        random_state=self.random_state + 100 * label_idx + file_idx,
                        tree_name=self.tree_name,
                        columns=columns,
                        feature_builder=self.feature_builder,
                        graph_builder=self.graph_builder,
                    )
                )

        workers = data_worker_count(len(tasks))
        print(
            "Preparing "
            f"{len(tasks)} ROOT files with {workers} CPU worker(s) "
            f"(os.cpu_count()={detected_cpu_count()})",
            flush=True,
        )
        results = parallel_map(_load_cp_file_task, tasks, workers=workers)

        for result in results:
            if feature_names is None:
                feature_names = [str(col) for col in result.features.columns]
            X_file = result.features.reindex(columns=feature_names, fill_value=np.nan)
            X_parts.append(X_file)
            if result.graph is not None:
                graph_parts.append(result.graph)
            y_parts.append(np.full(len(X_file), int(result.label), dtype=np.int64))

        if feature_names is None:
            raise ValueError("No label files were configured.")

        X_all = pd.concat(X_parts, ignore_index=True)
        graph_all = EventGraphDataset.concat(graph_parts) if graph_parts else None
        y_all = np.concatenate(y_parts)
        print(
            f"Loaded {len(y_all)} events across {len(feature_names)} features.",
            flush=True,
        )

        self._validate_split()
        trainval_idx, test_idx = stratified_split_indices(
            y_all,
            test_size=self.test_fraction,
            random_state=self.random_state,
        )
        val_relative_size = self.val_fraction / (self.train_fraction + self.val_fraction)
        train_rel_idx, val_rel_idx = stratified_split_indices(
            y_all[trainval_idx],
            test_size=val_relative_size,
            random_state=self.random_state + 11,
        )
        train_idx = trainval_idx[train_rel_idx]
        val_idx = trainval_idx[val_rel_idx]

        X_train = X_all.iloc[train_idx].reset_index(drop=True)
        y_train = y_all[train_idx]
        X_val = X_all.iloc[val_idx].reset_index(drop=True)
        y_val = y_all[val_idx]
        X_test = X_all.iloc[test_idx].reset_index(drop=True)
        y_test = y_all[test_idx]
        graph_train = graph_all.subset(train_idx) if graph_all is not None else None
        graph_val = graph_all.subset(val_idx) if graph_all is not None else None
        graph_test = graph_all.subset(test_idx) if graph_all is not None else None

        imputer = MedianImputer().fit(X_train)
        X_train = imputer.transform(X_train).reset_index(drop=True)
        X_val = imputer.transform(X_val).reset_index(drop=True)
        X_test = imputer.transform(X_test).reset_index(drop=True)

        print(
            "Dataset split: "
            f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}",
            flush=True,
        )

        return DatasetBundle(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            medians=(
                imputer.medians_
                if imputer.medians_ is not None
                else pd.Series(dtype=np.float32)
            ),
            metadata={
                "n_all": int(len(y_all)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "split": {
                    "train": self.train_fraction,
                    "val": self.val_fraction,
                    "test": self.test_fraction,
                },
                "label_files": self.label_files,
                "graph_features": None
                if graph_all is None
                else {
                    "node_features": graph_all.node_feature_names,
                    "global_features": graph_all.global_feature_names,
                },
            },
            graph_train=graph_train,
            graph_val=graph_val,
            graph_test=graph_test,
        )

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        fingerprint = self._cache_fingerprint()
        return self.cache_dir / f"cp_dataset_{fingerprint}.pkl"

    def _cache_fingerprint(self) -> str:
        payload = {
            "raw_dir": str(self.raw_dir),
            "label_files": self.label_files,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "random_state": self.random_state,
            "tree_name": self.tree_name,
            "scalar_cols": self.feature_builder.scalar_cols,
            "jagged_maxlen": self.feature_builder.jagged_maxlen,
            "padding": self.feature_builder.padding,
            "graph_builder": None
            if self.graph_builder is None
            else {
                "scalar_cols": self.graph_builder.scalar_cols,
                "particles": [
                    {"name": particle.name, "branches": particle.branches}
                    for particle in self.graph_builder.particles
                ],
            },
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:12]

    def _input_columns(self) -> list[str]:
        columns = self.feature_builder.input_columns()
        if self.graph_builder is not None:
            columns = [*columns, *self.graph_builder.input_columns()]
        return list(dict.fromkeys(columns))

    def _validate_split(self) -> None:
        values = {
            "train": self.train_fraction,
            "val": self.val_fraction,
            "test": self.test_fraction,
        }
        for name, value in values.items():
            if value <= 0.0:
                raise ValueError(f"{name}_fraction must be positive.")
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if not np.isclose(total, 1.0):
            raise ValueError("train/val/test fractions must sum to 1.0.")


def build_default_cp_dataset(
    random_state: int = 42,
    dataset_config: Any | None = None,
    build_graphs: bool = False,
    cache_dir: Path | None = None,
) -> DatasetBundle:
    settings = CPDatasetSettings()
    resolved_cache_dir = (
        cache_dir or settings.cache_dir or settings.raw_dir / ".tabpfn_feature_cache"
    )
    raw_dir = settings.raw_dir
    tree_name = settings.tree_name
    label_files = {0: [settings.even_file], 1: [settings.odd_file]}
    scalar_cols = settings.scalar_cols
    jagged_maxlen = settings.jagged_maxlen
    padding = settings.padding
    particle_specs: list[ParticleGraphSpec] = []
    if dataset_config is not None:
        if dataset_config.raw_dir is not None:
            raw_dir = dataset_config.raw_dir
            resolved_cache_dir = (
                cache_dir or settings.cache_dir or raw_dir / ".tabpfn_feature_cache"
            )
        if getattr(dataset_config, "cache_dir", None) is not None:
            resolved_cache_dir = dataset_config.cache_dir
        if getattr(dataset_config, "tree_name", None) is not None:
            tree_name = dataset_config.tree_name
        if dataset_config.labels:
            label_files = {
                label_config.label: label_config.files
                for label_config in dataset_config.labels
            }
        scalar_cols = list(dataset_config.scalars) or scalar_cols
        jagged_maxlen = _jagged_maxlen_from_config(dataset_config) or jagged_maxlen
        particle_specs = _particle_specs_from_config(dataset_config)
        padding = str(dataset_config.padding)
        train_fraction = dataset_config.split.train
        val_fraction = dataset_config.split.val
        test_fraction = dataset_config.split.test
    else:
        train_fraction = settings.train_fraction
        val_fraction = settings.val_fraction
        test_fraction = settings.test_fraction
    feature_builder = AtlasFeatureBuilder(
        scalar_cols=scalar_cols,
        jagged_maxlen=jagged_maxlen,
        padding=padding,
    )
    graph_builder = (
        AtlasGraphBuilder(scalar_cols=scalar_cols, particles=particle_specs)
        if build_graphs
        else None
    )
    return CPDatasetBuilder(
        raw_dir=raw_dir,
        label_files=label_files,
        feature_builder=feature_builder,
        graph_builder=graph_builder,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
        tree_name=tree_name,
        cache_dir=resolved_cache_dir,
        use_cache=settings.use_cache,
    ).build()


def _jagged_maxlen_from_config(dataset_config: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for particle in dataset_config.particles:
        for branch in particle.branches:
            out[str(branch)] = int(particle.max_particles)
    return out


def _particle_specs_from_config(dataset_config: Any) -> list[ParticleGraphSpec]:
    return [
        ParticleGraphSpec(
            name=str(particle.name),
            branches=[str(branch) for branch in particle.branches],
        )
        for particle in dataset_config.particles
    ]
