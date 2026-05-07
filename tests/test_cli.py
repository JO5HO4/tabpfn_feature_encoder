from pathlib import Path

from tabpfn_feature_encoder import cli
from tabpfn_feature_encoder.cli import build_parser
from tabpfn_feature_encoder.config import ProjectConfig


def test_cli_accepts_cp_transfer_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["transfer-cp", "--config", "configs/source_residual_mlp.yaml"])

    assert args.command == "transfer-cp"
    assert args.config == "configs/source_residual_mlp.yaml"
    assert args.model is None


def test_cli_accepts_cp_transfer_model_path() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "transfer-cp",
            "--config",
            "configs/source_residual_mlp.yaml",
            "--model",
            "/tmp/encoder_classifier.pkl",
        ]
    )

    assert args.command == "transfer-cp"
    assert Path(args.model) == Path("/tmp/encoder_classifier.pkl")


def test_cli_accepts_source_transfer_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "transfer-source",
            "--config",
            "configs/source_residual_mlp.yaml",
            "--model",
            "/tmp/encoder_classifier.pkl",
        ]
    )

    assert args.command == "transfer-source"
    assert args.config == "configs/source_residual_mlp.yaml"
    assert Path(args.model) == Path("/tmp/encoder_classifier.pkl")


def test_cli_accepts_context_comparison_plot_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "plot-context-comparison",
            "--output-dir",
            "/tmp/plots",
            "--run",
            "MLP encoder",
            "/tmp/mlp",
        ]
    )

    assert args.command == "plot-context-comparison"
    assert args.output_dir == "/tmp/plots"
    assert args.run == [["MLP encoder", "/tmp/mlp"]]


def test_source_generalization_uses_transfer_context_scan_settings(monkeypatch, tmp_path) -> None:
    cfg = ProjectConfig.from_dict(
        {
            "output_dir": str(tmp_path / "run"),
            "device": "cpu",
            "transfer": {
                "context_min_per_class": 33,
                "context_scan_points": 4,
                "context_repeats": 2,
                "context_size": 999,
                "query_chunk_size": 44,
            },
        }
    )
    captured = {}

    def fake_run_encoder_context_scan_evaluation(**kwargs):
        captured["output_dir"] = kwargs["output_dir"]
        captured["context_min_per_class"] = kwargs["context_min_per_class"]
        captured["context_scan_points"] = kwargs["context_scan_points"]
        captured["context_repeats"] = kwargs["context_repeats"]
        captured["max_context_size"] = kwargs["max_context_size"]
        captured["query_chunk_size"] = kwargs["query_chunk_size"]
        captured["name"] = kwargs["name"]
        return {
            "baseline_tabpfn": {"accuracy": 0.5, "log_loss": 0.7, "roc_auc": 0.5},
            "frozen_encoder_tabpfn": {"accuracy": 0.6, "log_loss": 0.6, "roc_auc": 0.6},
            "delta": {"accuracy": 0.1, "log_loss": -0.1, "roc_auc": 0.1},
        }

    monkeypatch.setattr(
        cli,
        "run_encoder_context_scan_evaluation",
        fake_run_encoder_context_scan_evaluation,
    )

    cli._run_source_generalization(cfg=cfg, model=object(), dataset=object())

    assert captured["output_dir"] == tmp_path / "run" / "source_generalization"
    assert captured["context_min_per_class"] == 33
    assert captured["context_scan_points"] == 4
    assert captured["context_repeats"] == 2
    assert captured["max_context_size"] == 999
    assert captured["query_chunk_size"] == 44
    assert captured["name"] == "source_12_class_generalization"


def test_cp_generalization_uses_transfer_context_scan_settings(monkeypatch, tmp_path) -> None:
    cfg = ProjectConfig.from_dict(
        {
            "output_dir": str(tmp_path / "run"),
            "cache_dir": str(tmp_path / "cache"),
            "device": "cpu",
            "dataset": {"raw_dir": str(tmp_path / "raw")},
            "transfer": {
                "context_min_per_class": 77,
                "context_scan_points": 5,
                "context_repeats": 3,
                "query_chunk_size": 55,
            },
        }
    )
    captured = {}

    def fake_build_default_cp_dataset(**kwargs):
        captured["build_graphs"] = kwargs["build_graphs"]
        return object()

    def fake_run_encoder_context_scan_evaluation(**kwargs):
        captured["context_min_per_class"] = kwargs["context_min_per_class"]
        captured["context_scan_points"] = kwargs["context_scan_points"]
        captured["context_repeats"] = kwargs["context_repeats"]
        captured["max_context_size"] = kwargs["max_context_size"]
        captured["query_chunk_size"] = kwargs["query_chunk_size"]
        captured["name"] = kwargs["name"]
        return {
            "baseline_tabpfn": {"accuracy": 0.5, "log_loss": 0.7, "roc_auc": 0.5},
            "frozen_encoder_tabpfn": {"accuracy": 0.6, "log_loss": 0.6, "roc_auc": 0.6},
            "delta": {"accuracy": 0.1, "log_loss": -0.1, "roc_auc": 0.1},
        }

    monkeypatch.setattr(cli, "build_default_cp_dataset", fake_build_default_cp_dataset)
    monkeypatch.setattr(
        cli,
        "run_encoder_context_scan_evaluation",
        fake_run_encoder_context_scan_evaluation,
    )

    cli._run_cp_generalization(cfg=cfg, model=object(), use_graph_encoder=True)

    assert captured["build_graphs"] is True
    assert captured["context_min_per_class"] == 77
    assert captured["context_scan_points"] == 5
    assert captured["context_repeats"] == 3
    assert captured["max_context_size"] is None
    assert captured["query_chunk_size"] == 55
    assert captured["name"] == "cp_even_odd_generalization"
