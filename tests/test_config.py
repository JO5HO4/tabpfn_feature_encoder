from pathlib import Path

from tabpfn_feature_encoder.config import ProjectConfig, load_project_config


def test_project_config_from_dict() -> None:
    cfg = ProjectConfig.from_dict(
        {
            "output_dir": "/tmp/demo",
            "cache_dir": "/tmp/cache",
            "seed": 7,
            "device": "cpu",
            "dataset": {
                "raw_dir": "/tmp/root",
                "split": {"train": 0.5, "val": 0.25, "test": 0.25},
                "labels": [
                    {"label": 0, "files": ["even.root"]},
                    {"label": 1, "files": ["odd_a.root", "odd_b.root"]},
                ],
                "padding": "nan",
                "scalars": ["MET_met"],
                "particles": [
                    {
                        "name": "jet",
                        "max": 4,
                        "branches": ["jet_pt", "jet_eta"],
                    }
                ],
            },
            "encoder": {
                "type": "residual_mlp",
                "layers": 4,
                "hidden_dim": 32,
                "attention_heads": 2,
                "output_dim": 16,
                "epochs": 3,
                "learning_rate": 0.01,
                "batch_size": 2048,
                "support_query_ratio": 0.5,
                "residual_scale": 0.05,
                "grad_clip_norm": 0.2,
                "early_stopping_patience": 4,
                "min_delta": 0.002,
                "validation_episodes": 6,
            },
            "transfer": {
                "raw_dir": "/tmp/gamgam",
                "cache_dir": "/tmp/gamgam-cache",
                "output_dir": "/tmp/transfer-out",
                "tree_name": "mini",
                "context_size": 128,
                "context_min_per_class": 10,
                "context_scan_points": 5,
                "context_repeats": 3,
                "query_chunk_size": 64,
                "labels": [
                    {"label": 0, "name": "ttH", "files": ["ttH.root"]},
                    {"label": 1, "name": "ggF", "files": ["ggF.root"]},
                ],
            },
        }
    )

    assert cfg.output_dir == Path("/tmp/demo")
    assert cfg.cache_dir == Path("/tmp/cache")
    assert cfg.seed == 7
    assert cfg.dataset.raw_dir == Path("/tmp/root")
    assert cfg.dataset.split.train == 0.5
    assert cfg.dataset.split.val == 0.25
    assert cfg.dataset.split.test == 0.25
    assert cfg.dataset.labels[0].label == 0
    assert cfg.dataset.labels[0].files == ["even.root"]
    assert cfg.dataset.labels[1].label == 1
    assert cfg.dataset.labels[1].files == ["odd_a.root", "odd_b.root"]
    assert cfg.dataset.padding == "nan"
    assert cfg.dataset.scalars == ["MET_met"]
    assert cfg.dataset.particles[0].name == "jet"
    assert cfg.dataset.particles[0].max_particles == 4
    assert cfg.dataset.particles[0].branches == ["jet_pt", "jet_eta"]
    assert cfg.encoder.type == "residual_mlp"
    assert cfg.encoder.layers == 4
    assert cfg.encoder.hidden_dim == 32
    assert cfg.encoder.attention_heads == 2
    assert cfg.encoder.output_dim == 16
    assert cfg.encoder.epochs == 3
    assert cfg.encoder.learning_rate == 0.01
    assert cfg.encoder.batch_size == 2048
    assert cfg.encoder.support_query_ratio == 0.5
    assert cfg.encoder.residual_scale == 0.05
    assert cfg.encoder.grad_clip_norm == 0.2
    assert cfg.encoder.early_stopping_patience == 4
    assert cfg.encoder.min_delta == 0.002
    assert cfg.encoder.validation_episodes == 6
    assert cfg.transfer.raw_dir == Path("/tmp/gamgam")
    assert cfg.transfer.cache_dir == Path("/tmp/gamgam-cache")
    assert cfg.transfer.output_dir == Path("/tmp/transfer-out")
    assert cfg.transfer.tree_name == "mini"
    assert cfg.transfer.context_size == 128
    assert cfg.transfer.context_min_per_class == 10
    assert cfg.transfer.context_scan_points == 5
    assert cfg.transfer.context_repeats == 3
    assert cfg.transfer.query_chunk_size == 64
    assert cfg.transfer.labels[0].name == "ttH"
    assert cfg.transfer.labels[1].files == ["ggF.root"]


def test_encoder_defaults_match_main_training_config() -> None:
    cfg = ProjectConfig.from_dict({"output_dir": "/tmp/demo"})

    assert cfg.device == "cuda"
    assert cfg.encoder.type == "residual_mlp"
    assert cfg.encoder.layers == 4
    assert cfg.encoder.hidden_dim == 64
    assert cfg.encoder.attention_heads == 4
    assert cfg.encoder.output_dim == 72
    assert cfg.encoder.learning_rate == 5e-5
    assert cfg.encoder.batch_size == 2048
    assert cfg.encoder.early_stopping_patience == 8
    assert cfg.encoder.min_delta == 0.001
    assert cfg.encoder.validation_episodes == 8
    assert cfg.transfer.context_size is None
    assert cfg.transfer.context_min_per_class == 100
    assert cfg.transfer.context_scan_points == 16
    assert cfg.transfer.context_repeats == 5
    assert cfg.transfer.query_chunk_size == 1024


def test_main_config_uses_12_class_source_task_and_holds_out_cp_files() -> None:
    cfg = load_project_config(Path("configs/source_residual_mlp.yaml"))
    configured_files = {
        filename
        for label_config in cfg.dataset.labels
        for filename in label_config.files
    }

    assert len(cfg.dataset.labels) == 12
    assert "ttH_NLO.root" not in configured_files
    assert "ttH_CPodd.root" not in configured_files
    assert cfg.encoder.type == "residual_mlp"


def test_source_encoder_configs_have_clear_output_names() -> None:
    expected = {
        "source_residual_mlp.yaml": ("residual_mlp", "source_residual_mlp"),
        "source_gnn.yaml": ("gnn", "source_gnn"),
        "source_transformer.yaml": ("transformer", "source_transformer"),
    }

    for filename, (encoder_type, run_name) in expected.items():
        cfg = load_project_config(Path("configs") / filename)

        assert cfg.encoder.type == encoder_type
        assert cfg.output_dir.name == run_name
        assert cfg.encoder.output_dim == 72
        assert cfg.encoder.learning_rate == 2e-4
        assert cfg.encoder.grad_clip_norm == 1.0
        assert cfg.encoder.tabpfn_max_classes == 2
        assert cfg.encoder.validation_episodes == 8
        assert cfg.transfer.output_dir == cfg.output_dir / "open_data_generalization"
        assert cfg.transfer.context_size is None
        assert cfg.transfer.context_min_per_class == 100
        assert cfg.transfer.context_scan_points == 16
        assert cfg.transfer.context_repeats == 5
        assert len(cfg.dataset.labels) == 12
