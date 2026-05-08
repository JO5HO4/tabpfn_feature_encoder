from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EncoderConfig:
    type: str = "residual_mlp"
    layers: int = 4
    hidden_dim: int = 64
    attention_heads: int = 4
    output_dim: int = 72
    epochs: int = 20
    learning_rate: float = 5e-5
    batch_size: int = 2048
    support_query_ratio: float = 0.5
    residual_scale: float = 0.1
    grad_clip_norm: float = 0.1
    early_stopping_patience: int = 8
    min_delta: float = 0.001
    many_class_redundancy: int = 4
    tabpfn_max_classes: int = 10
    validation_episodes: int = 8

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EncoderConfig:
        encoder_type = str(payload.get("type", payload.get("kind", "residual_mlp")))
        support_query_ratio = float(payload.get("support_query_ratio", 0.5))
        residual_scale = float(payload.get("residual_scale", 0.1))
        grad_clip_norm = float(payload.get("grad_clip_norm", 0.1))
        early_stopping_patience = int(payload.get("early_stopping_patience", 8))
        min_delta = float(payload.get("min_delta", 0.001))
        attention_heads = int(payload.get("attention_heads", payload.get("heads", 4)))
        many_class_redundancy = int(payload.get("many_class_redundancy", 4))
        tabpfn_max_classes = int(payload.get("tabpfn_max_classes", 10))
        validation_episodes = int(payload.get("validation_episodes", 8))
        if not 0.0 < support_query_ratio < 1.0:
            raise ValueError("encoder.support_query_ratio must be between 0 and 1.")
        if attention_heads <= 0:
            raise ValueError("encoder.attention_heads must be positive.")
        if residual_scale < 0.0:
            raise ValueError("encoder.residual_scale must be non-negative.")
        if grad_clip_norm < 0.0:
            raise ValueError("encoder.grad_clip_norm must be non-negative.")
        if early_stopping_patience < 0:
            raise ValueError("encoder.early_stopping_patience must be non-negative.")
        if min_delta < 0.0:
            raise ValueError("encoder.min_delta must be non-negative.")
        if many_class_redundancy <= 0:
            raise ValueError("encoder.many_class_redundancy must be positive.")
        if tabpfn_max_classes <= 1:
            raise ValueError("encoder.tabpfn_max_classes must be greater than one.")
        if validation_episodes <= 0:
            raise ValueError("encoder.validation_episodes must be positive.")
        return cls(
            type=encoder_type,
            layers=int(payload.get("layers", 4)),
            hidden_dim=int(payload.get("hidden_dim", 64)),
            attention_heads=attention_heads,
            output_dim=int(payload.get("output_dim", 72)),
            epochs=int(payload.get("epochs", 20)),
            learning_rate=float(payload.get("learning_rate", 5e-5)),
            batch_size=int(payload.get("batch_size", 2048)),
            support_query_ratio=support_query_ratio,
            residual_scale=residual_scale,
            grad_clip_norm=grad_clip_norm,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            many_class_redundancy=many_class_redundancy,
            tabpfn_max_classes=tabpfn_max_classes,
            validation_episodes=validation_episodes,
        )


@dataclass(frozen=True)
class ParticleBranchConfig:
    name: str
    branches: list[str]
    max_particles: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParticleBranchConfig:
        if "name" not in payload:
            raise KeyError("Missing required config key: dataset.particles[].name")
        if "branches" not in payload:
            raise KeyError("Missing required config key: dataset.particles[].branches")
        if "max" not in payload:
            raise KeyError("Missing required config key: dataset.particles[].max")
        branches = payload["branches"]
        if not isinstance(branches, list):
            raise TypeError("dataset.particles[].branches must be a list.")
        return cls(
            name=str(payload["name"]),
            branches=[str(branch) for branch in branches],
            max_particles=int(payload["max"]),
        )


@dataclass(frozen=True)
class LabelFileConfig:
    label: int
    files: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LabelFileConfig:
        if "label" not in payload:
            raise KeyError("Missing required config key: dataset.labels[].label")
        if "files" not in payload:
            raise KeyError("Missing required config key: dataset.labels[].files")
        files = payload["files"]
        if not isinstance(files, list):
            raise TypeError("dataset.labels[].files must be a list.")
        if not files:
            raise ValueError("dataset.labels[].files must not be empty.")
        return cls(
            label=int(payload["label"]),
            files=[str(filename) for filename in files],
        )


@dataclass(frozen=True)
class NamedLabelFileConfig:
    label: int
    name: str
    files: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NamedLabelFileConfig:
        if "label" not in payload:
            raise KeyError("Missing required config key: transfer.labels[].label")
        if "name" not in payload:
            raise KeyError("Missing required config key: transfer.labels[].name")
        if "files" not in payload:
            raise KeyError("Missing required config key: transfer.labels[].files")
        files = payload["files"]
        if not isinstance(files, list):
            raise TypeError("transfer.labels[].files must be a list.")
        if not files:
            raise ValueError("transfer.labels[].files must not be empty.")
        return cls(
            label=int(payload["label"]),
            name=str(payload["name"]),
            files=[str(filename) for filename in files],
        )


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.5
    val: float = 0.25
    test: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitConfig:
        split = cls(
            train=float(payload.get("train", 0.5)),
            val=float(payload.get("val", 0.25)),
            test=float(payload.get("test", 0.25)),
        )
        split.validate()
        return split

    def validate(self) -> None:
        values = {"train": self.train, "val": self.val, "test": self.test}
        for name, value in values.items():
            if value <= 0.0:
                raise ValueError(f"dataset.split.{name} must be positive.")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("dataset.split train/val/test values must sum to 1.0.")


@dataclass(frozen=True)
class DatasetConfig:
    raw_dir: Path | None = None
    cache_dir: Path | None = None
    tree_name: str | None = None
    split: SplitConfig = field(default_factory=SplitConfig)
    labels: list[LabelFileConfig] = field(default_factory=list)
    scalars: list[str] = field(default_factory=list)
    particles: list[ParticleBranchConfig] = field(default_factory=list)
    padding: str = "zero"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetConfig:
        scalars = payload.get("scalars", [])
        if not isinstance(scalars, list):
            raise TypeError("dataset.scalars must be a list.")
        particles = payload.get("particles", [])
        if not isinstance(particles, list):
            raise TypeError("dataset.particles must be a list.")
        labels = payload.get("labels", [])
        if not isinstance(labels, list):
            raise TypeError("dataset.labels must be a list.")
        split_payload = payload.get("split", {})
        if not isinstance(split_payload, dict):
            raise TypeError("dataset.split must be a mapping.")
        padding = str(payload.get("padding", "zero")).lower()
        if padding not in {"zero", "nan"}:
            raise ValueError("dataset.padding must be `zero` or `nan`.")
        return cls(
            raw_dir=Path(payload["raw_dir"]) if "raw_dir" in payload else None,
            cache_dir=Path(payload["cache_dir"]) if "cache_dir" in payload else None,
            tree_name=str(payload["tree_name"]) if "tree_name" in payload else None,
            split=SplitConfig.from_dict(split_payload),
            labels=[LabelFileConfig.from_dict(item) for item in labels],
            scalars=[str(branch) for branch in scalars],
            particles=[ParticleBranchConfig.from_dict(item) for item in particles],
            padding=padding,
        )


@dataclass(frozen=True)
class TransferConfig:
    raw_dir: Path = Path("/global/cfs/projectdirs/atlas/haichen/opendata/GamGam_data")
    cache_dir: Path | None = None
    output_dir: Path | None = None
    tree_name: str = "mini"
    split: SplitConfig = field(default_factory=SplitConfig)
    labels: list[NamedLabelFileConfig] = field(
        default_factory=lambda: [
            NamedLabelFileConfig(0, "ttH", ["mc_341081.ttH125_gamgam.GamGam.root"]),
            NamedLabelFileConfig(1, "ggF", ["mc_343981.ggH125_gamgam.GamGam.root"]),
            NamedLabelFileConfig(2, "VBF", ["mc_345041.VBFH125_gamgam.GamGam.root"]),
            NamedLabelFileConfig(3, "WH", ["mc_345318.WpH125J_Wincl_gamgam.GamGam.root"]),
            NamedLabelFileConfig(4, "ZH", ["mc_345319.ZH125J_Zincl_gamgam.GamGam.root"]),
        ]
    )
    context_size: int | None = None
    context_min_per_class: int = 100
    context_scan_points: int = 16
    context_repeats: int = 5
    query_chunk_size: int = 1024
    encoder_model: Path | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransferConfig:
        split_payload = payload.get("split", {})
        if not isinstance(split_payload, dict):
            raise TypeError("transfer.split must be a mapping.")
        labels = payload.get("labels")
        if labels is not None and not isinstance(labels, list):
            raise TypeError("transfer.labels must be a list.")
        context_size = (
            int(payload["context_size"])
            if payload.get("context_size") is not None
            else None
        )
        context_min_per_class = int(payload.get("context_min_per_class", 100))
        context_scan_points = int(payload.get("context_scan_points", 16))
        context_repeats = int(payload.get("context_repeats", 5))
        query_chunk_size = int(payload.get("query_chunk_size", 1024))
        if context_size is not None and context_size <= 0:
            raise ValueError("transfer.context_size must be positive.")
        if context_min_per_class <= 0:
            raise ValueError("transfer.context_min_per_class must be positive.")
        if context_scan_points <= 0:
            raise ValueError("transfer.context_scan_points must be positive.")
        if context_repeats <= 0:
            raise ValueError("transfer.context_repeats must be positive.")
        if query_chunk_size <= 0:
            raise ValueError("transfer.query_chunk_size must be positive.")
        return cls(
            raw_dir=Path(payload.get("raw_dir", cls.raw_dir)),
            cache_dir=Path(payload["cache_dir"]) if "cache_dir" in payload else None,
            output_dir=Path(payload["output_dir"]) if "output_dir" in payload else None,
            tree_name=str(payload.get("tree_name", "mini")),
            split=SplitConfig.from_dict(split_payload),
            labels=(
                [NamedLabelFileConfig.from_dict(item) for item in labels]
                if labels is not None
                else cls().labels
            ),
            context_size=context_size,
            context_min_per_class=context_min_per_class,
            context_scan_points=context_scan_points,
            context_repeats=context_repeats,
            query_chunk_size=query_chunk_size,
            encoder_model=Path(payload["encoder_model"]) if "encoder_model" in payload else None,
        )


@dataclass(frozen=True)
class ProjectConfig:
    output_dir: Path
    cache_dir: Path | None = None
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    seed: int = 42
    device: str = "cuda"
    metrics: list[str] = field(default_factory=lambda: ["roc_auc", "accuracy"])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectConfig:
        if "output_dir" not in payload:
            raise KeyError("Missing required config key: output_dir")
        encoder_payload = payload.get("encoder", {})
        if not isinstance(encoder_payload, dict):
            raise TypeError("encoder must be a mapping.")
        dataset_payload = payload.get("dataset", {})
        if not isinstance(dataset_payload, dict):
            raise TypeError("dataset must be a mapping.")
        transfer_payload = payload.get("transfer", {})
        if not isinstance(transfer_payload, dict):
            raise TypeError("transfer must be a mapping.")
        metrics = payload.get("metrics", ["roc_auc", "accuracy"])
        if not isinstance(metrics, list):
            raise TypeError("metrics must be a list.")
        return cls(
            output_dir=Path(payload["output_dir"]),
            cache_dir=Path(payload["cache_dir"]) if "cache_dir" in payload else None,
            encoder=EncoderConfig.from_dict(encoder_payload),
            dataset=DatasetConfig.from_dict(dataset_payload),
            transfer=TransferConfig.from_dict(transfer_payload),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "cuda")),
            metrics=[str(metric) for metric in metrics],
        )


def load_project_config(path: str | Path) -> ProjectConfig:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Config root must be a mapping.")
    return ProjectConfig.from_dict(payload)
