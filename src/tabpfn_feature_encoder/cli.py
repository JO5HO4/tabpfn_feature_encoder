from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from tabpfn_feature_encoder.config import DatasetConfig, LabelFileConfig, load_project_config
from tabpfn_feature_encoder.data.atlas_root import build_default_cp_dataset
from tabpfn_feature_encoder.data.gamgam_root import build_gamgam_dataset
from tabpfn_feature_encoder.evaluation.transfer import (
    print_transfer_summary,
    run_encoder_context_scan_evaluation,
)
from tabpfn_feature_encoder.evaluation.plots import save_encoder_comparison_plots
from tabpfn_feature_encoder.training.artifacts import save_training_artifacts
from tabpfn_feature_encoder.training.encoder_classifier import EncoderOnlyClassifier
from tabpfn_feature_encoder.utils.io import load_pickle, save_json, save_pickle
from tabpfn_feature_encoder.utils.seed import set_global_seed


def _log_progress(message: str) -> None:
    print(f"[tabpfn-feature-encoder] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a TabPFN feature encoder.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Run encoder training from a YAML config.")
    train.add_argument("--config", required=True, help="Path to YAML config.")
    source_transfer = subparsers.add_parser(
        "transfer-source",
        help="Evaluate a frozen source-trained encoder on the 12-class source task.",
    )
    source_transfer.add_argument("--config", required=True, help="Path to YAML config.")
    source_transfer.add_argument(
        "--model",
        default=None,
        help="Path to saved encoder checkpoint. Defaults to output_dir/encoder_classifier.pkl.",
    )
    transfer = subparsers.add_parser(
        "transfer",
        help="Evaluate a frozen source-trained encoder on GamGam production modes.",
    )
    transfer.add_argument("--config", required=True, help="Path to YAML config.")
    transfer.add_argument(
        "--model",
        default=None,
        help=(
            "Path to saved encoder checkpoint. Defaults to transfer.encoder_model, "
            "then output_dir/encoder_classifier.pkl."
        ),
    )
    cp_transfer = subparsers.add_parser(
        "transfer-cp",
        help="Evaluate a frozen source-trained encoder on held-out CP even vs odd.",
    )
    cp_transfer.add_argument("--config", required=True, help="Path to YAML config.")
    cp_transfer.add_argument(
        "--model",
        default=None,
        help="Path to saved encoder checkpoint. Defaults to output_dir/encoder_classifier.pkl.",
    )
    plot = subparsers.add_parser(
        "plot-context-comparison",
        help="Plot context-scan comparison curves across trained encoder runs.",
    )
    plot.add_argument(
        "--output-dir",
        default=None,
        help="Directory for comparison PDFs. Defaults to ../runs/context_scan_comparison.",
    )
    plot.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        default=None,
        help="Run label and output directory. May be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        run_train(Path(args.config))
    elif args.command == "transfer-source":
        run_source_transfer(
            Path(args.config),
            model_path=None if args.model is None else Path(args.model),
        )
    elif args.command == "transfer":
        run_transfer(Path(args.config), model_path=None if args.model is None else Path(args.model))
    elif args.command == "transfer-cp":
        run_cp_transfer(
            Path(args.config),
            model_path=None if args.model is None else Path(args.model),
        )
    elif args.command == "plot-context-comparison":
        run_plot_context_comparison(
            output_dir=None if args.output_dir is None else Path(args.output_dir),
            runs=args.run,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


def run_train(config_path: Path) -> None:
    cfg = load_project_config(config_path)
    set_global_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config_path": str(config_path)}, cfg.output_dir / "run_metadata.json")
    _log_progress(
        "Starting full train/evaluate workflow: "
        f"config={config_path}, output_dir={cfg.output_dir}, "
        f"encoder={cfg.encoder.type}, device={cfg.device}"
    )

    use_graph_encoder = cfg.encoder.type.lower() in {
        "gnn",
        "graph",
        "graph_gnn",
        "transformer",
        "particle_transformer",
        "graph_transformer",
    }
    cache_dir = _cache_subdir(cfg.cache_dir, "source_multiclass")
    _log_progress(
        "Stage 1/6 loading source dataset: "
        f"raw_dir={cfg.dataset.raw_dir}, cache_dir={cache_dir}, "
        f"graph_features={use_graph_encoder}"
    )
    dataset = build_default_cp_dataset(
        random_state=cfg.seed,
        dataset_config=cfg.dataset,
        build_graphs=use_graph_encoder,
        cache_dir=cache_dir,
    )
    _log_progress(
        "Loaded source dataset: "
        f"train={len(dataset.y_train)}, val={len(dataset.y_val)}, "
        f"test={len(dataset.y_test)}, flat_features={dataset.X_train.shape[1]}"
    )
    X_train = dataset.graph_train if use_graph_encoder else dataset.X_train
    X_val = dataset.graph_val if use_graph_encoder else dataset.X_val
    if X_train is None or X_val is None:
        raise RuntimeError("Graph encoder requested, but graph dataset was not built.")
    model = EncoderOnlyClassifier(
        encoder=cfg.encoder,
        device=cfg.device,
        random_state=cfg.seed,
    )
    _log_progress("Stage 2/6 training source encoder through frozen TabPFN.")
    model.fit(
        X_train,
        dataset.y_train,
        X_val=X_val,
        y_val=dataset.y_val,
    )
    source_val_metrics = model.evaluate(X_val, dataset.y_val)
    X_test = dataset.graph_test if use_graph_encoder else dataset.X_test
    if X_test is None:
        raise RuntimeError("Graph encoder requested, but graph test dataset was not built.")
    _log_progress("Evaluating best source encoder checkpoint on validation and test splits.")
    source_test_metrics = model.evaluate(X_test, dataset.y_test)
    print(
        "source_12_class val: "
        + ", ".join(f"{key}={value:.3f}" for key, value in source_val_metrics.items())
    )
    print(
        "source_12_class test: "
        + ", ".join(f"{key}={value:.3f}" for key, value in source_test_metrics.items())
    )

    _log_progress("Stage 3/6 running source 12-class TabPFN context scan.")
    source_generalization = _run_source_generalization(
        cfg=cfg,
        model=model,
        dataset=dataset,
    )
    _log_progress("Stage 4/6 running held-out CP even/odd TabPFN context scan.")
    cp_generalization = _run_cp_generalization(
        cfg=cfg,
        model=model,
        use_graph_encoder=use_graph_encoder,
    )
    _log_progress("Stage 5/6 running open-data GamGam TabPFN context scan.")
    open_data_generalization = _run_open_data_generalization(
        cfg=cfg,
        model=model,
    )

    metrics = {}
    for metric_name, metric_value in source_val_metrics.items():
        metrics[f"source_val_{metric_name}"] = metric_value
    for metric_name, metric_value in source_test_metrics.items():
        metrics[f"source_test_{metric_name}"] = metric_value
    for family in ("baseline_tabpfn", "frozen_encoder_tabpfn", "delta"):
        for metric_name, metric_value in source_generalization[family].items():
            metrics[f"source_generalization_{family}_{metric_name}"] = metric_value
        for metric_name, metric_value in cp_generalization[family].items():
            metrics[f"cp_generalization_{family}_{metric_name}"] = metric_value
        for metric_name, metric_value in open_data_generalization[family].items():
            metrics[f"open_data_generalization_{family}_{metric_name}"] = metric_value
    metrics.update(
        {
            "n_train": int(len(dataset.y_train)),
            "n_val": int(len(dataset.y_val)),
            "n_test": int(len(dataset.y_test)),
            "n_source_classes": 0 if model.classes_ is None else int(len(model.classes_)),
            "n_features": int(
                cfg.encoder.output_dim if use_graph_encoder else dataset.X_train.shape[1]
            ),
            "n_flat_features": int(dataset.X_train.shape[1]),
            "n_tabpfn_features": int(cfg.encoder.output_dim),
            "trainable_encoder_params": model._count_trainable_parameters(model.encoder_model_),
        }
    )
    if use_graph_encoder and dataset.graph_train is not None:
        metrics["n_node_features"] = int(dataset.graph_train.node_dim)
        metrics["n_global_features"] = int(dataset.graph_train.global_dim)
    if model.best_epoch_ is not None:
        metrics["best_epoch"] = int(model.best_epoch_)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    _log_progress("Stage 6/6 saving model and metric artifacts.")
    model.prepare_for_serialization()
    best_checkpoint_path = cfg.output_dir / "encoder_classifier.pkl"
    save_pickle(model, best_checkpoint_path)
    save_json(
        {
            "checkpoint": best_checkpoint_path,
            "best_epoch": model.best_epoch_,
            "source_val_roc_auc": metrics.get("source_val_roc_auc"),
            "source_val_accuracy": metrics.get("source_val_accuracy"),
            "source_val_log_loss": metrics.get("source_val_log_loss"),
            "source_training_task": "frozen_tabpfn_support_query_encoder_training",
            "tabpfn_used_for_source_training": True,
            "tabpfn_trainable": False,
        },
        cfg.output_dir / "best_checkpoint.json",
    )
    saved = save_training_artifacts(
        output_dir=cfg.output_dir,
        metrics=metrics,
        training_summary=model.get_training_summary(),
        model=model,
        model_artifact="encoder_classifier.pkl",
        save_model=True,
        save_metrics=True,
    )
    epoch_log_path = cfg.output_dir / "epoch_metrics.csv"
    pd.DataFrame([asdict(row) for row in model.history_]).to_csv(
        epoch_log_path,
        index=False,
    )
    saved["epoch_metrics"] = epoch_log_path
    saved["best_checkpoint"] = best_checkpoint_path
    saved["best_checkpoint_metadata"] = cfg.output_dir / "best_checkpoint.json"
    for name, path in saved.items():
        print(f"saved {name}: {path}")
    _log_progress("Full train/evaluate workflow finished.")


def run_source_transfer(config_path: Path, model_path: Path | None = None) -> dict[str, Any]:
    cfg = load_project_config(config_path)
    set_global_seed(cfg.seed)
    resolved_model_path, trained = _load_encoder_checkpoint(
        cfg=cfg,
        model_path=model_path,
        allow_transfer_config=False,
        command_name="transfer-source",
    )
    dataset = build_default_cp_dataset(
        random_state=cfg.seed,
        dataset_config=cfg.dataset,
        build_graphs=trained.is_graph_input_,
        cache_dir=_cache_subdir(cfg.cache_dir, "source_multiclass"),
    )
    print(f"Using frozen encoder: {resolved_model_path}")
    metrics = _run_source_generalization(
        cfg=cfg,
        model=trained,
        dataset=dataset,
    )
    output_dir = cfg.output_dir / "source_generalization"
    print(f"saved transfer metrics: {output_dir / 'source_12_class_generalization_metrics.json'}")
    return metrics


def run_transfer(config_path: Path, model_path: Path | None = None) -> None:
    cfg = load_project_config(config_path)
    set_global_seed(cfg.seed)
    output_dir = cfg.transfer.output_dir or cfg.output_dir / "open_data_generalization"
    cache_dir = cfg.transfer.cache_dir or _cache_subdir(cfg.cache_dir, "gamgam_production_modes")
    resolved_model_path, trained = _load_encoder_checkpoint(
        cfg=cfg,
        model_path=model_path,
        allow_transfer_config=True,
        command_name="transfer",
    )
    dataset = build_gamgam_dataset(
        random_state=cfg.seed,
        transfer_config=cfg.transfer,
        cache_dir=cache_dir,
        build_graphs=trained.is_graph_input_,
    )
    print(f"Using frozen encoder: {resolved_model_path}")
    print(f"Using GamGam cache dir: {cache_dir}")
    print(
        "Transfer TabPFN context scan: "
        f"context_split=val, min_per_class={cfg.transfer.context_min_per_class}, "
        f"points={cfg.transfer.context_scan_points}, "
        f"repeats={cfg.transfer.context_repeats}, "
        f"max_context={cfg.transfer.context_size or 'full_val'}, "
        f"query_chunk={cfg.transfer.query_chunk_size}, "
        "query_split=test"
    )
    metrics = run_encoder_context_scan_evaluation(
        trained=trained,
        dataset=dataset,
        output_dir=output_dir,
        context_min_per_class=cfg.transfer.context_min_per_class,
        context_scan_points=cfg.transfer.context_scan_points,
        context_repeats=cfg.transfer.context_repeats,
        max_context_size=cfg.transfer.context_size,
        query_chunk_size=cfg.transfer.query_chunk_size,
        device=cfg.device,
        random_state=cfg.seed,
        name="open_data_generalization",
    )
    print_transfer_summary("open_data_generalization", metrics)
    print(f"saved transfer metrics: {output_dir / 'open_data_generalization_metrics.json'}")


def run_cp_transfer(config_path: Path, model_path: Path | None = None) -> dict[str, Any]:
    cfg = load_project_config(config_path)
    set_global_seed(cfg.seed)
    resolved_model_path, trained = _load_encoder_checkpoint(
        cfg=cfg,
        model_path=model_path,
        allow_transfer_config=False,
        command_name="transfer-cp",
    )
    print(f"Using frozen encoder: {resolved_model_path}")
    metrics = _run_cp_generalization(
        cfg=cfg,
        model=trained,
        use_graph_encoder=trained.is_graph_input_,
    )
    output_dir = cfg.output_dir / "cp_generalization"
    print(f"saved transfer metrics: {output_dir / 'cp_even_odd_generalization_metrics.json'}")
    return metrics


def run_plot_context_comparison(
    *,
    output_dir: Path | None,
    runs: list[list[str]] | None,
) -> dict[str, list[Path]]:
    run_specs = (
        [(str(label), Path(path)) for label, path in runs]
        if runs
        else _default_comparison_runs()
    )
    resolved_output_dir = output_dir or _default_runs_root() / "context_scan_comparison"
    saved = save_encoder_comparison_plots(run_specs, resolved_output_dir)
    if not saved:
        print("No context-scan CSV files found for comparison plotting.")
        return saved
    for task_name, paths in saved.items():
        for path in paths:
            print(f"saved {task_name} comparison plot: {path}")
    return saved


def _cache_subdir(cache_dir: Path | None, name: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / name


def _split_size(dataset: Any, split_name: str) -> int | str:
    values = getattr(dataset, split_name, None)
    return "unknown" if values is None else len(values)


def _load_encoder_checkpoint(
    *,
    cfg: Any,
    model_path: Path | None,
    allow_transfer_config: bool,
    command_name: str,
) -> tuple[Path, EncoderOnlyClassifier]:
    resolved_model_path = (
        model_path
        or (cfg.transfer.encoder_model if allow_transfer_config else None)
        or _default_encoder_checkpoint(cfg.output_dir)
    )
    trained = load_pickle(resolved_model_path)
    if not isinstance(trained, EncoderOnlyClassifier):
        raise TypeError(f"{command_name} --model must point to a saved EncoderOnlyClassifier.")
    return Path(resolved_model_path), trained


def _default_encoder_checkpoint(output_dir: Path) -> Path:
    return output_dir / "encoder_classifier.pkl"


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parents[3] / "runs"


def _default_comparison_runs() -> list[tuple[str, Path]]:
    root = _default_runs_root()
    return [
        ("MLP encoder", root / "source_residual_mlp"),
        ("GNN encoder", root / "source_gnn"),
        ("Transformer encoder", root / "source_transformer"),
    ]


def _run_source_generalization(
    *,
    cfg: Any,
    model: EncoderOnlyClassifier,
    dataset: Any,
) -> dict[str, Any]:
    output_dir = cfg.output_dir / "source_generalization"
    _log_progress("Running source 12-class TabPFN context scan.")
    print(
        "Source 12-class TabPFN context scan: "
        f"context_split=val, min_per_class={cfg.transfer.context_min_per_class}, "
        f"points={cfg.transfer.context_scan_points}, "
        f"repeats={cfg.transfer.context_repeats}, "
        f"max_context={cfg.transfer.context_size or 'full_val'}, "
        f"query_chunk={cfg.transfer.query_chunk_size}, "
        "query_split=test"
    )
    metrics = run_encoder_context_scan_evaluation(
        trained=model,
        dataset=dataset,
        output_dir=output_dir,
        context_min_per_class=cfg.transfer.context_min_per_class,
        context_scan_points=cfg.transfer.context_scan_points,
        context_repeats=cfg.transfer.context_repeats,
        max_context_size=cfg.transfer.context_size,
        query_chunk_size=cfg.transfer.query_chunk_size,
        device=cfg.device,
        random_state=cfg.seed,
        name="source_12_class_generalization",
    )
    print_transfer_summary("source_12_class_generalization", metrics)
    return metrics


def _run_cp_generalization(
    *,
    cfg: Any,
    model: EncoderOnlyClassifier,
    use_graph_encoder: bool,
) -> dict[str, Any]:
    cp_dataset_config = _cp_generalization_dataset_config(cfg.dataset)
    cache_dir = _cache_subdir(cfg.cache_dir, "cp_even_odd_generalization")
    _log_progress(
        "Loading held-out CP even/odd dataset: "
        f"raw_dir={cp_dataset_config.raw_dir}, cache_dir={cache_dir}, "
        f"graph_features={use_graph_encoder}"
    )
    dataset = build_default_cp_dataset(
        random_state=cfg.seed,
        dataset_config=cp_dataset_config,
        build_graphs=use_graph_encoder,
        cache_dir=cache_dir,
    )
    _log_progress(
        "Running held-out CP even/odd TabPFN context scan: "
        f"context_pool={_split_size(dataset, 'y_val')}, "
        f"query={_split_size(dataset, 'y_test')}"
    )
    output_dir = cfg.output_dir / "cp_generalization"
    print(
        "CP transfer TabPFN context scan: "
        f"context_split=val, min_per_class={cfg.transfer.context_min_per_class}, "
        f"points={cfg.transfer.context_scan_points}, "
        f"repeats={cfg.transfer.context_repeats}, "
        f"max_context={cfg.transfer.context_size or 'full_val'}, "
        f"query_chunk={cfg.transfer.query_chunk_size}, "
        "query_split=test"
    )
    metrics = run_encoder_context_scan_evaluation(
        trained=model,
        dataset=dataset,
        output_dir=output_dir,
        context_min_per_class=cfg.transfer.context_min_per_class,
        context_scan_points=cfg.transfer.context_scan_points,
        context_repeats=cfg.transfer.context_repeats,
        max_context_size=cfg.transfer.context_size,
        query_chunk_size=cfg.transfer.query_chunk_size,
        device=cfg.device,
        random_state=cfg.seed,
        name="cp_even_odd_generalization",
    )
    print_transfer_summary("cp_even_odd_generalization", metrics)
    return metrics


def _run_open_data_generalization(
    *,
    cfg: Any,
    model: EncoderOnlyClassifier,
) -> dict[str, Any]:
    output_dir = cfg.transfer.output_dir or cfg.output_dir / "open_data_generalization"
    cache_dir = cfg.transfer.cache_dir or _cache_subdir(cfg.cache_dir, "gamgam_production_modes")
    _log_progress(
        "Loading open-data GamGam dataset: "
        f"raw_dir={cfg.transfer.raw_dir}, cache_dir={cache_dir}, "
        f"graph_features={model.is_graph_input_}"
    )
    dataset = build_gamgam_dataset(
        random_state=cfg.seed,
        transfer_config=cfg.transfer,
        cache_dir=cache_dir,
        build_graphs=model.is_graph_input_,
    )
    _log_progress(
        "Running open-data GamGam TabPFN context scan: "
        f"context_pool={_split_size(dataset, 'y_val')}, "
        f"query={_split_size(dataset, 'y_test')}"
    )
    print(
        "Open-data GamGam TabPFN context scan: "
        f"context_split=val, min_per_class={cfg.transfer.context_min_per_class}, "
        f"points={cfg.transfer.context_scan_points}, "
        f"repeats={cfg.transfer.context_repeats}, "
        f"max_context={cfg.transfer.context_size or 'full_val'}, "
        f"query_chunk={cfg.transfer.query_chunk_size}, "
        "query_split=test"
    )
    metrics = run_encoder_context_scan_evaluation(
        trained=model,
        dataset=dataset,
        output_dir=output_dir,
        context_min_per_class=cfg.transfer.context_min_per_class,
        context_scan_points=cfg.transfer.context_scan_points,
        context_repeats=cfg.transfer.context_repeats,
        max_context_size=cfg.transfer.context_size,
        query_chunk_size=cfg.transfer.query_chunk_size,
        device=cfg.device,
        random_state=cfg.seed,
        name="open_data_generalization",
    )
    print_transfer_summary("open_data_generalization", metrics)
    return metrics


def _cp_generalization_dataset_config(source: DatasetConfig) -> DatasetConfig:
    raw_dir = source.raw_dir or Path("/global/cfs/projectdirs/atlas/joshua/gnn_data/stats_100K")
    odd_filename = "ttH_CPodd_NLO.root"
    if not (raw_dir / odd_filename).exists():
        odd_filename = "ttH_CPodd.root"
    return replace(
        source,
        labels=[
            LabelFileConfig(label=0, files=["ttH_NLO.root"]),
            LabelFileConfig(label=1, files=[odd_filename]),
        ],
    )


if __name__ == "__main__":
    main()
