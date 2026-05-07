from pathlib import Path

from tabpfn_feature_encoder.cli import build_parser


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
