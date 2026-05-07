from __future__ import annotations

import numpy as np

from tabpfn_feature_encoder.config import EncoderConfig
from tabpfn_feature_encoder.training.encoder_classifier import EncoderOnlyClassifier


def test_encoder_only_classifier_uses_encoder_config_and_predicts_probabilities() -> None:
    X_train = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1])
    cfg = EncoderConfig(
        type="residual_mlp",
        layers=2,
        hidden_dim=4,
        output_dim=2,
        epochs=2,
        learning_rate=0.01,
        batch_size=2,
        early_stopping_patience=0,
    )

    classifier = EncoderOnlyClassifier(encoder=cfg, device="cpu", random_state=3).fit(
        X_train,
        y_train,
        X_val=X_train,
        y_val=y_train,
    )
    proba = classifier.predict_proba(X_train)

    assert classifier.encoder_model_.output_dim == cfg.output_dim
    assert classifier.encoder.hidden_dim == cfg.hidden_dim
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_encoder_only_classifier_maps_labels_to_contiguous_training_indices() -> None:
    X_train = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([10, 10, 20, 20])
    cfg = EncoderConfig(
        type="residual_mlp",
        layers=2,
        hidden_dim=4,
        output_dim=2,
        epochs=1,
        learning_rate=0.01,
        batch_size=2,
        early_stopping_patience=0,
    )

    classifier = EncoderOnlyClassifier(encoder=cfg, device="cpu", random_state=3).fit(
        X_train,
        y_train,
    )

    assert classifier.classes_.tolist() == [10, 20]
    assert classifier.predict_proba(X_train).shape == (4, 2)
