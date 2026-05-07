from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from tabpfn_feature_encoder.config import load_project_config
from tabpfn_feature_encoder.data.atlas_root import build_default_cp_dataset
from tabpfn_feature_encoder.data.gamgam_root import build_gamgam_dataset
from tabpfn_feature_encoder.evaluation.transfer import run_gnn_transfer_evaluation
from tabpfn_feature_encoder.training.artifacts import save_training_artifacts
from tabpfn_feature_encoder.training.trainer import EncoderTabPFNClassifier
from tabpfn_feature_encoder.utils.io import save_json, save_pickle
from tabpfn_feature_encoder.utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple TabPFN feature encoder.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Run encoder training from a YAML config.")
    train.add_argument("--config", required=True, help="Path to YAML config.")
    transfer = subparsers.add_parser(
        "transfer",
        help="Evaluate a frozen CP-trained GNN encoder on GamGam production modes.",
    )
    transfer.add_argument("--config", required=True, help="Path to YAML config.")
    transfer.add_argument(
        "--model",
        default=None,
        help=(
            "Path to saved encoder checkpoint. Defaults to transfer.encoder_model, "
            "then output_dir/encoder_best_val_auc.pkl, then output_dir/encoder_tabpfn.pkl."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        run_train(Path(args.config))
    elif args.command == "transfer":
        run_transfer(Path(args.config), model_path=None if args.model is None else Path(args.model))
    else:
        parser.error(f"Unknown command: {args.command}")


def run_train(config_path: Path) -> None:
    cfg = load_project_config(config_path)
    set_global_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config_path": str(config_path)}, cfg.output_dir / "run_metadata.json")

    use_graph_encoder = cfg.encoder.type.lower() in {"gnn", "graph", "graph_gnn"}
    cache_dir = _cache_subdir(cfg.cache_dir, "cp_encoder")
    dataset = build_default_cp_dataset(
        random_state=cfg.seed,
        dataset_config=cfg.dataset,
        build_graphs=use_graph_encoder,
        cache_dir=cache_dir,
    )
    X_train = dataset.graph_train if use_graph_encoder else dataset.X_train
    X_val = dataset.graph_val if use_graph_encoder else dataset.X_val
    if X_train is None or X_val is None:
        raise RuntimeError("Graph encoder requested, but graph dataset was not built.")
    model = EncoderTabPFNClassifier(
        encoder=cfg.encoder,
        device=cfg.device,
        random_state=cfg.seed,
    )
    model.fit(
        X_train,
        dataset.y_train,
        X_eval=X_val,
        y_eval=dataset.y_val,
        eval_metrics=cfg.metrics,
    )

    metrics = {f"val_{key}": value for key, value in (model.latest_evaluation_ or {}).items()}
    metrics.update(
        {
            "n_train": int(len(dataset.y_train)),
            "n_val": int(len(dataset.y_val)),
            "n_test_held_out": int(len(dataset.y_test)),
            "n_features": int(
                cfg.encoder.output_dim if use_graph_encoder else dataset.X_train.shape[1]
            ),
            "n_flat_features": int(dataset.X_train.shape[1]),
            "n_tabpfn_features": int(cfg.encoder.output_dim),
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

    model.prepare_for_serialization()
    best_checkpoint_path = cfg.output_dir / "encoder_best_val_auc.pkl"
    save_pickle(model, best_checkpoint_path)
    save_json(
        {
            "checkpoint": best_checkpoint_path,
            "best_epoch": model.best_epoch_,
            "val_roc_auc": metrics.get("val_roc_auc"),
            "val_accuracy": metrics.get("val_accuracy"),
            "val_log_loss": metrics.get("val_log_loss"),
        },
        cfg.output_dir / "best_checkpoint.json",
    )
    saved = save_training_artifacts(
        output_dir=cfg.output_dir,
        metrics=metrics,
        training_summary=model.get_training_summary(),
        model=model,
        model_artifact="encoder_tabpfn.pkl",
        save_model=True,
        save_metrics=True,
    )
    epoch_log_path = cfg.output_dir / "epoch_metrics.csv"
    pd.DataFrame([asdict(row) for row in model.epoch_metrics_]).to_csv(
        epoch_log_path,
        index=False,
    )
    saved["epoch_metrics"] = epoch_log_path
    saved["best_checkpoint"] = best_checkpoint_path
    saved["best_checkpoint_metadata"] = cfg.output_dir / "best_checkpoint.json"
    for name, path in saved.items():
        print(f"saved {name}: {path}")


def run_transfer(config_path: Path, model_path: Path | None = None) -> None:
    cfg = load_project_config(config_path)
    set_global_seed(cfg.seed)
    output_dir = cfg.transfer.output_dir or cfg.output_dir / "gamgam_transfer"
    cache_dir = cfg.transfer.cache_dir or _cache_subdir(cfg.cache_dir, "gamgam_production_modes")
    dataset = build_gamgam_dataset(
        random_state=cfg.seed,
        transfer_config=cfg.transfer,
        cache_dir=cache_dir,
    )
    resolved_model_path = (
        model_path
        or cfg.transfer.encoder_model
        or _default_encoder_checkpoint(cfg.output_dir)
    )
    print(f"Using frozen encoder: {resolved_model_path}")
    print(f"Using GamGam cache dir: {cache_dir}")
    print(
        "Transfer TabPFN batch: "
        f"context={cfg.transfer.context_size}, "
        f"query_chunk={cfg.transfer.query_chunk_size}, "
        f"total={cfg.transfer.context_size + cfg.transfer.query_chunk_size}"
    )
    metrics = run_gnn_transfer_evaluation(
        encoder_model_path=resolved_model_path,
        dataset=dataset,
        output_dir=output_dir,
        context_size=cfg.transfer.context_size,
        query_chunk_size=cfg.transfer.query_chunk_size,
        device=cfg.device,
        random_state=cfg.seed,
    )
    for family in (
        "frozen_gnn_tabpfn",
        "frozen_gnn_plus_flat_tabpfn",
        "baseline_tabpfn",
        "delta",
        "plus_flat_delta",
    ):
        values = metrics[family]
        text = ", ".join(f"{key}={value:.3f}" for key, value in values.items())
        print(f"{family}: {text}")
    print(f"saved transfer metrics: {output_dir / 'transfer_metrics.json'}")


def _cache_subdir(cache_dir: Path | None, name: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / name


def _default_encoder_checkpoint(output_dir: Path) -> Path:
    best = output_dir / "encoder_best_val_auc.pkl"
    if best.exists():
        return best
    return output_dir / "encoder_tabpfn.pkl"


if __name__ == "__main__":
    main()
