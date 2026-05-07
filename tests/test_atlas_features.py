import numpy as np
import pandas as pd

from tabpfn_feature_encoder.data.atlas_root import (
    AtlasFeatureBuilder,
    AtlasGraphBuilder,
    CPDatasetBuilder,
    CPDatasetSettings,
    ParticleGraphSpec,
    _jagged_maxlen_from_config,
    pad_jagged,
)
from tabpfn_feature_encoder.data.base import DatasetBundle
from tabpfn_feature_encoder.utils.io import save_pickle


def test_pad_jagged_clips_and_fills() -> None:
    series = pd.Series([[1.0, 2.0, 3.0], [], [4.0]])

    out = pad_jagged(series, max_len=2, fill=-1.0)

    assert np.allclose(out, [[1.0, 2.0], [-1.0, -1.0], [4.0, -1.0]])


def test_atlas_feature_builder_creates_stable_columns() -> None:
    df = pd.DataFrame(
        {
            "MET_met": [100.0, 120.0],
            "jet_pt": [[40.0, 30.0], [50.0]],
            "jet_eta": [[0.1, -0.2], [0.3]],
        }
    )
    builder = AtlasFeatureBuilder(
        scalar_cols=["MET_met", "missing_scalar"],
        jagged_maxlen={"jet_pt": 2, "jet_eta": 2},
    )

    X = builder.build(df)

    assert list(X.columns) == [
        "MET_met",
        "missing_scalar",
        "jet_pt_0",
        "jet_pt_1",
        "jet_eta_0",
        "jet_eta_1",
    ]
    assert np.allclose(X["missing_scalar"], 0.0)


def test_atlas_feature_builder_can_nan_pad() -> None:
    df = pd.DataFrame({"jet_pt": [[40.0], []]})
    builder = AtlasFeatureBuilder(
        scalar_cols=[],
        jagged_maxlen={"jet_pt": 2},
        padding="nan",
    )

    X = builder.build(df)

    assert np.isnan(X.loc[0, "jet_pt_1"])
    assert np.isnan(X.loc[1, "jet_pt_0"])


def test_cp_dataset_builder_loads_existing_cache(tmp_path) -> None:
    bundle = DatasetBundle(
        X_train=pd.DataFrame({"x": [1.0]}),
        y_train=np.asarray([0]),
        X_val=pd.DataFrame({"x": [2.0]}),
        y_val=np.asarray([0]),
        X_test=pd.DataFrame({"x": [3.0]}),
        y_test=np.asarray([0]),
        feature_names=["x"],
        medians=pd.Series({"x": 1.0}),
    )
    builder = CPDatasetBuilder(
        raw_dir=tmp_path,
        label_files={0: ["even.root"], 1: ["odd.root"]},
        feature_builder=AtlasFeatureBuilder(scalar_cols=["x"], jagged_maxlen={}),
        graph_builder=None,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        random_state=1,
        cache_dir=tmp_path / "cache",
    )
    cache_path = builder._cache_path()
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True)
    save_pickle(bundle, cache_path)

    loaded = builder.build()

    assert loaded.feature_names == ["x"]
    assert loaded.X_train.equals(bundle.X_train)


def test_atlas_graph_builder_uses_all_particles() -> None:
    df = pd.DataFrame(
        {
            "MET_met": [100.0],
            "MET_phi": [0.5],
            "jet_pt": [[40.0, 30.0, 20.0]],
            "jet_eta": [[0.1, -0.2, 0.3]],
            "jet_phi": [[0.0, 1.0, 2.0]],
            "jet_btag": [[1.0, 0.0, 1.0]],
            "mu_pt": [[25.0]],
            "mu_eta": [[-1.0]],
            "mu_phi": [[-0.5]],
            "mu_charge": [[-1.0]],
        }
    )
    builder = AtlasGraphBuilder(
        scalar_cols=["MET_met", "MET_phi"],
        particles=[
            ParticleGraphSpec(
                name="jet",
                branches=["jet_pt", "jet_eta", "jet_phi", "jet_btag"],
            ),
            ParticleGraphSpec(
                name="muon",
                branches=["mu_pt", "mu_eta", "mu_phi", "mu_charge"],
            ),
        ],
    )

    graphs = builder.build(df)

    assert len(graphs) == 1
    assert graphs.nodes[0].shape == (4, len(graphs.node_feature_names))
    assert graphs.global_features.shape == (1, 2)
    assert "type_jet" in graphs.node_feature_names
    assert "type_muon" in graphs.node_feature_names


def test_atlas_graph_builder_prints_progress(capsys) -> None:
    df = pd.DataFrame(
        {
            "MET_met": [100.0, 110.0, 120.0],
            "MET_phi": [0.5, 0.6, 0.7],
            "jet_pt": [[40.0], [30.0], [20.0]],
            "jet_eta": [[0.1], [0.2], [0.3]],
            "jet_phi": [[0.0], [1.0], [2.0]],
            "jet_btag": [[1.0], [0.0], [1.0]],
        }
    )
    builder = AtlasGraphBuilder(
        scalar_cols=["MET_met", "MET_phi"],
        particles=[
            ParticleGraphSpec(
                name="jet",
                branches=["jet_pt", "jet_eta", "jet_phi", "jet_btag"],
            )
        ],
    )

    builder.build(df, progress_label="cp-file", progress_interval=2)

    captured = capsys.readouterr()
    assert "cp-file: processed 2/3 events" in captured.out
    assert "cp-file: processed 3/3 events" in captured.out


def test_default_cp_split_is_50_25_25() -> None:
    settings = CPDatasetSettings()

    assert settings.train_fraction == 0.5
    assert settings.val_fraction == 0.25
    assert settings.test_fraction == 0.25
    assert np.isclose(
        settings.train_fraction + settings.val_fraction + settings.test_fraction,
        1.0,
    )


def test_jagged_maxlen_from_dataset_config() -> None:
    class Particle:
        branches = ["jet_pt", "jet_eta"]
        max_particles = 10

    class Dataset:
        particles = [Particle()]

    assert _jagged_maxlen_from_config(Dataset()) == {"jet_pt": 10, "jet_eta": 10}
