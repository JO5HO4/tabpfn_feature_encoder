import numpy as np

from tabpfn_feature_encoder.data.graphs import EventGraphDataset, GraphStandardizer


def test_event_graph_dataset_batches_variable_nodes() -> None:
    graphs = EventGraphDataset(
        nodes=[
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[5.0, 6.0]], dtype=np.float32),
        ],
        global_features=np.asarray([[10.0], [20.0]], dtype=np.float32),
        node_feature_names=["a", "b"],
        global_feature_names=["g"],
    )

    batch = graphs.to_batch(np.asarray([1, 0]), device="cpu")

    assert batch.node_features.shape == (3, 2)
    assert batch.batch_index.tolist() == [0, 1, 1]
    assert batch.global_features.shape == (2, 1)
    assert batch.n_graphs == 2


def test_graph_standardizer_preserves_shapes() -> None:
    graphs = EventGraphDataset(
        nodes=[
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[5.0, 6.0]], dtype=np.float32),
        ],
        global_features=np.asarray([[10.0], [20.0]], dtype=np.float32),
        node_feature_names=["a", "b"],
        global_feature_names=["g"],
    )

    transformed = GraphStandardizer().fit_transform(graphs)

    assert len(transformed) == 2
    assert transformed.nodes[0].shape == (2, 2)
    assert transformed.global_features.shape == (2, 1)


def test_graph_standardizer_preserves_type_indicators() -> None:
    graphs = EventGraphDataset(
        nodes=[
            np.asarray(
                [
                    [10.0, 1.0, 0.0],
                    [20.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.asarray([[30.0, 1.0, 0.0]], dtype=np.float32),
        ],
        global_features=np.asarray([[10.0], [20.0]], dtype=np.float32),
        node_feature_names=["log_pt", "type_jet", "type_photon"],
        global_feature_names=["met"],
    )

    transformed = GraphStandardizer().fit_transform(graphs)

    assert np.allclose(transformed.nodes[0][:, 1:], graphs.nodes[0][:, 1:])
    assert np.allclose(transformed.nodes[1][:, 1:], graphs.nodes[1][:, 1:])
