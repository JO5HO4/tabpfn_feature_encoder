from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabpfn_feature_encoder.data.atlas_root import (
    RootEventLoader,
    _finite_float,
    _print_event_progress,
    _safe_sequence,
    pad_jagged,
)
from tabpfn_feature_encoder.data.base import DatasetBundle
from tabpfn_feature_encoder.data.graphs import EventGraphDataset
from tabpfn_feature_encoder.data.parallel import (
    data_worker_count,
    detected_cpu_count,
    parallel_map,
)
from tabpfn_feature_encoder.data.preprocessing import MedianImputer, stratified_split_indices
from tabpfn_feature_encoder.utils.io import load_pickle, save_pickle


@dataclass(frozen=True)
class GamGamMode:
    label: int
    name: str
    files: list[str]


@dataclass(frozen=True)
class GamGamDatasetBuilder:
    raw_dir: Path
    modes: list[GamGamMode]
    train_fraction: float
    val_fraction: float
    test_fraction: float
    random_state: int
    tree_name: str = "mini"
    cache_dir: Path | None = None
    use_cache: bool = True
    build_graphs: bool = True

    def build(self) -> DatasetBundle:
        cache_path = self._cache_path()
        if self.use_cache and cache_path is not None and cache_path.exists():
            print(f"Loading cached GamGam dataset: {cache_path}")
            cached = load_pickle(cache_path)
            if not isinstance(cached, DatasetBundle):
                raise TypeError(f"Cached object at {cache_path} is not a DatasetBundle.")
            return cached

        bundle = self._build_from_root()
        if self.use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(bundle, cache_path)
            print(f"Saved cached GamGam dataset: {cache_path}")
        return bundle

    def _build_from_root(self) -> DatasetBundle:
        flat_builder = GamGamCPFeatureBuilder()
        graph_builder = GamGamGraphBuilder() if self.build_graphs else None
        graph_columns = [] if graph_builder is None else graph_builder.input_columns()
        columns = list(dict.fromkeys([*flat_builder.input_columns(), *graph_columns]))

        X_parts: list[pd.DataFrame] = []
        graph_parts: list[EventGraphDataset] = []
        y_parts: list[np.ndarray] = []
        feature_names: list[str] | None = None
        label_names: dict[int, str] = {}

        print(f"Building GamGam dataset from ROOT files in {self.raw_dir}", flush=True)
        tasks: list[_GamGamFileTask] = []
        for mode_idx, mode in enumerate(self.modes):
            label_names[int(mode.label)] = mode.name
            if not mode.files:
                raise ValueError(f"GamGam mode {mode.name} must have at least one file.")
            for file_idx, filename in enumerate(mode.files):
                tasks.append(
                    _GamGamFileTask(
                        raw_dir=self.raw_dir,
                        filename=filename,
                        label=int(mode.label),
                        mode_name=mode.name,
                        random_state=self.random_state + 100 * mode_idx + file_idx,
                        tree_name=self.tree_name,
                        columns=columns,
                        flat_builder=flat_builder,
                        graph_builder=graph_builder,
                    )
                )

        workers = data_worker_count(len(tasks))
        print(
            "Preparing "
            f"{len(tasks)} GamGam ROOT files with {workers} CPU worker(s) "
            f"(os.cpu_count()={detected_cpu_count()})",
            flush=True,
        )
        results = parallel_map(_load_gamgam_file_task, tasks, workers=workers)

        for result in results:
            if feature_names is None:
                feature_names = [str(col) for col in result.features.columns]
            X_file = result.features.reindex(columns=feature_names, fill_value=np.nan)
            X_parts.append(X_file)
            if result.graph is not None:
                graph_parts.append(result.graph)
            y_parts.append(np.full(len(X_file), int(result.label), dtype=np.int64))

        if feature_names is None:
            raise ValueError("No GamGam modes were configured.")

        X_all = pd.concat(X_parts, ignore_index=True)
        graph_all = EventGraphDataset.concat(graph_parts) if graph_parts else None
        y_all = np.concatenate(y_parts)
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
            "GamGam split: "
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
                "label_names": label_names,
                "split": {
                    "train": self.train_fraction,
                    "val": self.val_fraction,
                    "test": self.test_fraction,
                },
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
        return self.cache_dir / f"gamgam_dataset_{self._cache_fingerprint()}.pkl"

    def _cache_fingerprint(self) -> str:
        payload = {
            "raw_dir": str(self.raw_dir),
            "modes": [
                {"label": mode.label, "name": mode.name, "files": mode.files}
                for mode in self.modes
            ],
            "tree_name": self.tree_name,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "random_state": self.random_state,
            "build_graphs": self.build_graphs,
            "flat_schema": GamGamCPFeatureBuilder().feature_names(),
            "graph_schema": None
            if not self.build_graphs
            else GamGamGraphBuilder().node_feature_names(),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]

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


@dataclass(frozen=True)
class GamGamCPFeatureBuilder:
    """Build a 72-column flat feature view matching the CP source schema."""

    jet_max: int = 10
    lepton_max: int = 3
    photon_max: int = 2

    def input_columns(self) -> list[str]:
        return [
            "met_et",
            "met_phi",
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_MV2c10",
            "photon_pt",
            "photon_eta",
            "photon_phi",
            "lep_pt",
            "lep_eta",
            "lep_phi",
            "lep_charge",
            "lep_type",
        ]

    def feature_names(self) -> list[str]:
        names = ["MET_met", "MET_phi"]
        for branch in ("jet_pt", "jet_eta", "jet_phi", "jet_btag"):
            names.extend(f"{branch}_{idx}" for idx in range(self.jet_max))
        for prefix in ("ele", "mu"):
            for suffix in ("pt", "eta", "phi", "charge"):
                names.extend(f"{prefix}_{suffix}_{idx}" for idx in range(self.lepton_max))
        for branch in ("ph_pt", "ph_eta", "ph_phi"):
            names.extend(f"{branch}_{idx}" for idx in range(self.photon_max))
        return names

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = [
            pd.DataFrame(
                {
                    "MET_met": self._scalar(df, "met_et"),
                    "MET_phi": self._scalar(df, "met_phi"),
                },
                index=df.index,
            )
        ]
        parts.extend(
            [
                self._jagged(df, "jet_pt", "jet_pt", self.jet_max),
                self._jagged(df, "jet_eta", "jet_eta", self.jet_max),
                self._jagged(df, "jet_phi", "jet_phi", self.jet_max),
                self._jagged(df, "jet_MV2c10", "jet_btag", self.jet_max),
            ]
        )
        for particle_code, prefix in ((11, "ele"), (13, "mu")):
            split = self._split_leptons(df, particle_code)
            for suffix in ("pt", "eta", "phi", "charge"):
                parts.append(
                    pd.DataFrame(
                        pad_jagged(pd.Series(split[suffix], index=df.index), self.lepton_max),
                        columns=[f"{prefix}_{suffix}_{idx}" for idx in range(self.lepton_max)],
                        index=df.index,
                    )
                )
        parts.extend(
            [
                self._jagged(df, "photon_pt", "ph_pt", self.photon_max),
                self._jagged(df, "photon_eta", "ph_eta", self.photon_max),
                self._jagged(df, "photon_phi", "ph_phi", self.photon_max),
            ]
        )
        features = pd.concat(parts, axis=1).reindex(columns=self.feature_names(), fill_value=0.0)
        return features.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _scalar(df: pd.DataFrame, column: str) -> np.ndarray:
        if column not in df.columns:
            return np.zeros(len(df), dtype=np.float32)
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    @staticmethod
    def _jagged(df: pd.DataFrame, source: str, target: str, max_len: int) -> pd.DataFrame:
        values = (
            df[source]
            if source in df.columns
            else pd.Series([[] for _ in range(len(df))], index=df.index)
        )
        return pd.DataFrame(
            pad_jagged(values, max_len),
            columns=[f"{target}_{idx}" for idx in range(max_len)],
            index=df.index,
        )

    @staticmethod
    def _split_leptons(df: pd.DataFrame, particle_code: int) -> dict[str, list[list[float]]]:
        out = {key: [] for key in ("pt", "eta", "phi", "charge")}
        for _, row in df.iterrows():
            lep_type = _safe_sequence(row["lep_type"]) if "lep_type" in row.index else []
            values = {
                "pt": _safe_sequence(row["lep_pt"]) if "lep_pt" in row.index else [],
                "eta": _safe_sequence(row["lep_eta"]) if "lep_eta" in row.index else [],
                "phi": _safe_sequence(row["lep_phi"]) if "lep_phi" in row.index else [],
                "charge": (
                    _safe_sequence(row["lep_charge"]) if "lep_charge" in row.index else []
                ),
            }
            selected = {key: [] for key in out}
            for idx, type_value in enumerate(lep_type):
                if abs(int(round(type_value))) != particle_code:
                    continue
                for key, sequence in values.items():
                    selected[key].append(_value_at(sequence, idx))
            for key in out:
                out[key].append(selected[key])
        return out


@dataclass(frozen=True)
class GamGamGraphBuilder:
    """Build the same node schema as the CP GNN encoder, using all open-data particles."""

    global_cols: tuple[str, str] = ("met_et", "met_phi")

    def input_columns(self) -> list[str]:
        return [
            "met_et",
            "met_phi",
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_MV2c10",
            "photon_pt",
            "photon_eta",
            "photon_phi",
            "lep_pt",
            "lep_eta",
            "lep_phi",
            "lep_charge",
            "lep_type",
        ]

    def node_feature_names(self) -> list[str]:
        return [
            "log_pt",
            "eta",
            "sin_phi",
            "cos_phi",
            "charge",
            "btag",
            "type_jet",
            "type_electron",
            "type_muon",
            "type_photon",
        ]

    def global_feature_names(self) -> list[str]:
        return ["met_et", "met_phi"]

    def build(
        self,
        df: pd.DataFrame,
        *,
        progress_label: str | None = None,
        progress_interval: int = 50_000,
    ) -> EventGraphDataset:
        total = len(df)
        nodes: list[np.ndarray] = []
        for processed, (_, row) in enumerate(df.iterrows(), start=1):
            nodes.append(self._build_event_nodes(row))
            _print_event_progress(
                progress_label,
                processed,
                total,
                progress_interval,
            )
        return EventGraphDataset(
            nodes=nodes,
            global_features=self._build_globals(df),
            node_feature_names=self.node_feature_names(),
            global_feature_names=self.global_feature_names(),
        )

    def _build_globals(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col in self.global_cols:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            else:
                values = np.zeros(len(df), dtype=np.float32)
            parts.append(values.reshape(-1, 1))
        return np.nan_to_num(np.concatenate(parts, axis=1)).astype(np.float32)

    def _build_event_nodes(self, row: pd.Series) -> np.ndarray:
        out: list[np.ndarray] = []
        self._append_nodes(
            out,
            row=row,
            pt_branch="jet_pt",
            eta_branch="jet_eta",
            phi_branch="jet_phi",
            btag_branch="jet_MV2c10",
            type_index=6,
        )
        self._append_lepton_nodes(out, row=row)
        self._append_nodes(
            out,
            row=row,
            pt_branch="photon_pt",
            eta_branch="photon_eta",
            phi_branch="photon_phi",
            type_index=9,
        )
        if not out:
            return np.zeros((0, len(self.node_feature_names())), dtype=np.float32)
        return np.stack(out, axis=0).astype(np.float32)

    def _append_lepton_nodes(self, out: list[np.ndarray], *, row: pd.Series) -> None:
        pt = _safe_sequence(row["lep_pt"]) if "lep_pt" in row.index else []
        eta = _safe_sequence(row["lep_eta"]) if "lep_eta" in row.index else []
        phi = _safe_sequence(row["lep_phi"]) if "lep_phi" in row.index else []
        charge = _safe_sequence(row["lep_charge"]) if "lep_charge" in row.index else []
        lep_type = _safe_sequence(row["lep_type"]) if "lep_type" in row.index else []
        n_particles = max(len(pt), len(eta), len(phi), len(charge), len(lep_type))
        for idx in range(n_particles):
            type_value = abs(int(round(_value_at(lep_type, idx))))
            if type_value == 11:
                type_index = 7
            elif type_value == 13:
                type_index = 8
            else:
                continue
            node = self._base_node(
                pt=_value_at(pt, idx),
                eta=_value_at(eta, idx),
                phi=_value_at(phi, idx),
                charge=_value_at(charge, idx),
                btag=0.0,
                type_index=type_index,
            )
            out.append(node)

    def _append_nodes(
        self,
        out: list[np.ndarray],
        *,
        row: pd.Series,
        pt_branch: str,
        eta_branch: str,
        phi_branch: str,
        type_index: int,
        btag_branch: str | None = None,
    ) -> None:
        pt = _safe_sequence(row[pt_branch]) if pt_branch in row.index else []
        eta = _safe_sequence(row[eta_branch]) if eta_branch in row.index else []
        phi = _safe_sequence(row[phi_branch]) if phi_branch in row.index else []
        btag = _safe_sequence(row[btag_branch]) if btag_branch and btag_branch in row.index else []
        n_particles = max(len(pt), len(eta), len(phi), len(btag))
        for idx in range(n_particles):
            out.append(
                self._base_node(
                    pt=_value_at(pt, idx),
                    eta=_value_at(eta, idx),
                    phi=_value_at(phi, idx),
                    charge=0.0,
                    btag=_value_at(btag, idx),
                    type_index=type_index,
                )
            )

    def _base_node(
        self,
        *,
        pt: float,
        eta: float,
        phi: float,
        charge: float,
        btag: float,
        type_index: int,
    ) -> np.ndarray:
        node = np.zeros(len(self.node_feature_names()), dtype=np.float32)
        node[0] = np.log1p(max(_finite_float(pt), 0.0))
        node[1] = _finite_float(eta)
        phi_value = _finite_float(phi)
        node[2] = np.sin(phi_value)
        node[3] = np.cos(phi_value)
        node[4] = _finite_float(charge)
        node[5] = _finite_float(btag)
        node[type_index] = 1.0
        return node


def _value_at(values: list[float], idx: int) -> float:
    if idx >= len(values):
        return 0.0
    return _finite_float(values[idx])


@dataclass(frozen=True)
class _GamGamFileTask:
    raw_dir: Path
    filename: str
    label: int
    mode_name: str
    random_state: int
    tree_name: str
    columns: list[str]
    flat_builder: GamGamCPFeatureBuilder
    graph_builder: GamGamGraphBuilder | None


@dataclass(frozen=True)
class _GamGamFileResult:
    label: int
    filename: str
    features: pd.DataFrame
    graph: EventGraphDataset | None


def _load_gamgam_file_task(task: _GamGamFileTask) -> _GamGamFileResult:
    loader = RootEventLoader(tree_name=task.tree_name)
    path = task.raw_dir / task.filename
    print(f"Reading GamGam ROOT file: {path}", flush=True)
    df = loader.load_sample(
        path,
        random_state=task.random_state,
        columns=task.columns,
    )
    features = task.flat_builder.build(df)
    graph = (
        None
        if task.graph_builder is None
        else task.graph_builder.build(
            df,
            progress_label=f"{task.mode_name}/{task.filename} graph",
        )
    )
    print(f"Prepared {len(df)} {task.mode_name} events from {task.filename}", flush=True)
    return _GamGamFileResult(
        label=task.label,
        filename=task.filename,
        features=features,
        graph=graph,
    )


def build_gamgam_dataset(
    *,
    random_state: int,
    transfer_config: Any,
    cache_dir: Path | None,
    build_graphs: bool = True,
) -> DatasetBundle:
    modes = [
        GamGamMode(
            label=int(label_config.label),
            name=str(label_config.name),
            files=list(label_config.files),
        )
        for label_config in transfer_config.labels
    ]
    resolved_cache_dir = transfer_config.cache_dir or cache_dir
    return GamGamDatasetBuilder(
        raw_dir=transfer_config.raw_dir,
        modes=modes,
        train_fraction=transfer_config.split.train,
        val_fraction=transfer_config.split.val,
        test_fraction=transfer_config.split.test,
        random_state=random_state,
        tree_name=transfer_config.tree_name,
        cache_dir=resolved_cache_dir,
        build_graphs=build_graphs,
    ).build()
