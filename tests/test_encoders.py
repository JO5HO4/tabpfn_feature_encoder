import torch

from tabpfn_feature_encoder.data.graphs import EventGraphDataset
from tabpfn_feature_encoder.models.encoders import (
    FeatureGateEncoder,
    FeatureMixerEncoder,
    LightweightGNNEncoder,
    MLPEncoder,
    ParticleTransformerEncoder,
    build_encoder,
)


def test_mlp_encoder_starts_as_identity_when_dims_match() -> None:
    encoder = MLPEncoder(input_dim=4, hidden_dim=8, output_dim=4, layers=3)
    x = torch.randn(5, 4)

    out = encoder(x)

    assert encoder.uses_identity_residual
    assert encoder.residual_scale == 0.1
    assert torch.allclose(out, x)


def test_mlp_encoder_projects_when_dims_differ() -> None:
    encoder = MLPEncoder(input_dim=4, hidden_dim=8, output_dim=3, layers=3)
    x = torch.randn(5, 4)

    out = encoder(x)

    assert not encoder.uses_identity_residual
    assert encoder.residual_scale == 1.0
    assert out.shape == (5, 3)


def test_feature_gate_encoder_starts_as_identity() -> None:
    encoder = FeatureGateEncoder(input_dim=4, output_dim=4, residual_scale=0.2)
    x = torch.randn(5, 4)

    out = encoder(x)

    assert encoder.encoder_type == "feature_gate"
    assert encoder.uses_identity_residual
    assert encoder.residual_scale == 0.2
    assert torch.allclose(out, x)


def test_feature_mixer_encoder_starts_as_identity() -> None:
    encoder = FeatureMixerEncoder(input_dim=4, output_dim=4, residual_scale=0.2)
    x = torch.randn(5, 4)

    out = encoder(x)

    assert encoder.encoder_type == "feature_mixer"
    assert encoder.uses_identity_residual
    assert encoder.residual_scale == 0.2
    assert torch.allclose(out, x)


def test_build_encoder_selects_feature_gate() -> None:
    encoder = build_encoder(
        encoder_type="feature_gate",
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        layers=2,
        residual_scale=0.1,
    )

    assert isinstance(encoder, FeatureGateEncoder)


def test_build_encoder_selects_feature_mixer() -> None:
    encoder = build_encoder(
        encoder_type="feature_mixer",
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        layers=2,
        residual_scale=0.1,
    )

    assert isinstance(encoder, FeatureMixerEncoder)


def test_lightweight_gnn_encoder_outputs_event_embeddings() -> None:
    graphs = EventGraphDataset(
        nodes=[
            torch.randn(3, 4).numpy(),
            torch.randn(1, 4).numpy(),
        ],
        global_features=torch.randn(2, 2).numpy(),
        node_feature_names=["a", "b", "c", "d"],
        global_feature_names=["g0", "g1"],
    )
    batch = graphs.to_batch(torch.tensor([0, 1]).numpy(), device="cpu")
    encoder = LightweightGNNEncoder(
        node_dim=4,
        global_dim=2,
        hidden_dim=8,
        output_dim=5,
        layers=2,
    )

    out = encoder(batch)

    assert encoder.encoder_type == "gnn"
    assert out.shape == (2, 5)


def test_particle_transformer_encoder_outputs_event_embeddings() -> None:
    graphs = EventGraphDataset(
        nodes=[
            torch.randn(3, 4).numpy(),
            torch.randn(1, 4).numpy(),
        ],
        global_features=torch.randn(2, 2).numpy(),
        node_feature_names=["a", "b", "c", "d"],
        global_feature_names=["g0", "g1"],
    )
    batch = graphs.to_batch(torch.tensor([0, 1]).numpy(), device="cpu")
    encoder = ParticleTransformerEncoder(
        node_dim=4,
        global_dim=2,
        hidden_dim=8,
        output_dim=5,
        layers=2,
        attention_heads=2,
    )

    out = encoder(batch)

    assert encoder.encoder_type == "transformer"
    assert encoder.attention_heads == 2
    assert out.shape == (2, 5)


def test_build_encoder_selects_particle_transformer() -> None:
    encoder = build_encoder(
        encoder_type="transformer",
        input_dim=4,
        global_dim=2,
        hidden_dim=8,
        output_dim=5,
        layers=2,
        residual_scale=0.1,
        attention_heads=2,
    )

    assert isinstance(encoder, ParticleTransformerEncoder)


def test_lightweight_gnn_encoder_appends_direct_summary_features() -> None:
    graphs = EventGraphDataset(
        nodes=[
            torch.tensor(
                [
                    [1.0, 0.1, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0],
                    [2.0, -0.2, 1.0, 0.0, 0.0, 0.1, 0.0, 1.0],
                ]
            ).numpy(),
            torch.tensor([[3.0, 0.3, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0]]).numpy(),
        ],
        global_features=torch.randn(2, 2).numpy(),
        node_feature_names=[
            "log_pt",
            "eta",
            "sin_phi",
            "cos_phi",
            "charge",
            "btag",
            "type_jet",
            "type_photon",
        ],
        global_feature_names=["met", "phi"],
    )
    batch = graphs.to_batch(torch.tensor([0, 1]).numpy(), device="cpu")
    encoder = LightweightGNNEncoder(
        node_dim=8,
        global_dim=2,
        hidden_dim=8,
        output_dim=64,
        layers=2,
    )

    out = encoder(batch)
    expected_summary = encoder.summary_norm(encoder.summary_features(batch))

    assert encoder.summary_direct
    assert encoder.summary_dim == 2 + 1 + 2 + 3 * 8 + 2 * 2 * 6
    assert out.shape == (2, 64)
    assert torch.allclose(out[:, -encoder.summary_dim :], expected_summary)
