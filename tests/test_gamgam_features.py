import numpy as np
import pandas as pd

from tabpfn_feature_encoder.data.gamgam_root import GamGamCPFeatureBuilder, GamGamGraphBuilder


def test_gamgam_cp_feature_builder_matches_cp_flat_schema() -> None:
    df = pd.DataFrame(
        {
            "met_et": [100.0],
            "met_phi": [0.5],
            "jet_pt": [[40.0]],
            "jet_eta": [[0.1]],
            "jet_phi": [[0.0]],
            "jet_MV2c10": [[0.8]],
            "photon_pt": [[50.0, 45.0, 35.0]],
            "photon_eta": [[0.3, -0.4, 0.2]],
            "photon_phi": [[2.0, -2.0, 1.0]],
            "lep_pt": [[25.0, 20.0]],
            "lep_eta": [[1.0, -1.0]],
            "lep_phi": [[0.2, -0.2]],
            "lep_charge": [[-1.0, 1.0]],
            "lep_type": [[11.0, 13.0]],
        }
    )

    features = GamGamCPFeatureBuilder().build(df)

    assert features.shape == (1, 72)
    assert list(features.columns) == GamGamCPFeatureBuilder().feature_names()
    assert features.loc[0, "MET_met"] == 100.0
    assert features.loc[0, "jet_btag_0"] == 0.8
    assert features.loc[0, "ele_pt_0"] == 25.0
    assert features.loc[0, "mu_pt_0"] == 20.0
    assert features.loc[0, "ph_pt_0"] == 50.0
    assert features.loc[0, "ph_pt_1"] == 45.0


def test_gamgam_graph_builder_splits_leptons_and_uses_all_particles() -> None:
    df = pd.DataFrame(
        {
            "met_et": [100.0],
            "met_phi": [0.5],
            "jet_pt": [[40.0, 30.0]],
            "jet_eta": [[0.1, -0.2]],
            "jet_phi": [[0.0, 1.0]],
            "jet_MV2c10": [[0.8, 0.2]],
            "photon_pt": [[50.0, 45.0]],
            "photon_eta": [[0.3, -0.4]],
            "photon_phi": [[2.0, -2.0]],
            "lep_pt": [[25.0, 20.0]],
            "lep_eta": [[1.0, -1.0]],
            "lep_phi": [[0.2, -0.2]],
            "lep_charge": [[-1.0, 1.0]],
            "lep_type": [[11.0, 13.0]],
        }
    )

    graphs = GamGamGraphBuilder().build(df)
    node_names = graphs.node_feature_names

    assert len(graphs) == 1
    assert graphs.nodes[0].shape == (6, len(node_names))
    assert graphs.global_features.shape == (1, 2)
    assert np.count_nonzero(graphs.nodes[0][:, node_names.index("type_jet")]) == 2
    assert np.count_nonzero(graphs.nodes[0][:, node_names.index("type_electron")]) == 1
    assert np.count_nonzero(graphs.nodes[0][:, node_names.index("type_muon")]) == 1
    assert np.count_nonzero(graphs.nodes[0][:, node_names.index("type_photon")]) == 2


def test_gamgam_graph_builder_prints_progress(capsys) -> None:
    df = pd.DataFrame(
        {
            "met_et": [100.0, 110.0, 120.0],
            "met_phi": [0.5, 0.6, 0.7],
            "jet_pt": [[40.0], [30.0], [20.0]],
            "jet_eta": [[0.1], [0.2], [0.3]],
            "jet_phi": [[0.0], [1.0], [2.0]],
            "jet_MV2c10": [[0.8], [0.2], [0.1]],
            "photon_pt": [[], [], []],
            "photon_eta": [[], [], []],
            "photon_phi": [[], [], []],
        }
    )

    GamGamGraphBuilder().build(df, progress_label="gamgam-file", progress_interval=2)

    captured = capsys.readouterr()
    assert "gamgam-file: processed 2/3 events" in captured.out
    assert "gamgam-file: processed 3/3 events" in captured.out
