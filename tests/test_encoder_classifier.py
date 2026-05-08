from __future__ import annotations

import numpy as np

from tabpfn_feature_encoder.config import EncoderConfig
from tabpfn_feature_encoder.training import encoder_classifier as encoder_classifier_mod
from tabpfn_feature_encoder.training.encoder_classifier import EncoderOnlyClassifier


class FakeTabPFNPromptAdapter:
    def __init__(self, device: str, random_state: int | None = None) -> None:
        self.device = device
        self.random_state = random_state
        self.model = type("FakeModel", (), {"max_num_classes_": 10})()
        self._centroids = None

    def build(self):
        return self

    def fit_prompt(self, z_support, y_support) -> None:
        import torch

        classes = torch.unique(y_support, sorted=True)
        self._centroids = torch.stack(
            [z_support[y_support == label].mean(dim=0) for label in classes],
            dim=0,
        )

    def forward_logits(self, z_query):
        import torch

        if self._centroids is None:
            raise RuntimeError("Fake prompt was not fitted.")
        return -torch.cdist(z_query, self._centroids).pow(2)

    def predict_proba(self, z_query):
        import torch

        return torch.softmax(self.forward_logits(z_query), dim=1).detach().cpu().numpy()

    def clear_prompt(self) -> None:
        self._centroids = None


def test_encoder_only_classifier_uses_encoder_config_and_predicts_probabilities(monkeypatch) -> None:
    monkeypatch.setattr(encoder_classifier_mod, "TabPFNPromptAdapter", FakeTabPFNPromptAdapter)
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
    assert classifier.classifier_head_ is None
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert classifier.get_training_summary()["tabpfn_used_for_training"] is True


def test_encoder_only_classifier_maps_labels_to_contiguous_training_indices(monkeypatch) -> None:
    monkeypatch.setattr(encoder_classifier_mod, "TabPFNPromptAdapter", FakeTabPFNPromptAdapter)
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


def test_encoder_only_classifier_builds_ecoc_for_many_classes(monkeypatch) -> None:
    monkeypatch.setattr(encoder_classifier_mod, "TabPFNPromptAdapter", FakeTabPFNPromptAdapter)
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(24, 3)).astype(np.float32)
    y_train = np.repeat(np.arange(12), 2)
    cfg = EncoderConfig(
        type="mlp",
        layers=2,
        hidden_dim=8,
        output_dim=4,
        epochs=1,
        learning_rate=0.01,
        batch_size=24,
        early_stopping_patience=0,
    )

    classifier = EncoderOnlyClassifier(encoder=cfg, device="cpu", random_state=3).fit(
        X_train,
        y_train,
    )

    assert classifier.ecoc_codebook_ is not None
    assert classifier.ecoc_codebook_.shape[0] == 12
    assert classifier.ecoc_codebook_.max() < classifier.tabpfn_max_classes_
    assert len(np.unique(classifier.ecoc_codebook_, axis=0)) == 12


def test_ecoc_codebook_binary_columns_are_balanced() -> None:
    codebook = EncoderOnlyClassifier._make_ecoc_codebook(
        n_classes=12,
        alphabet_size=2,
        redundancy=4,
        random_state=3,
    )

    assert codebook is not None
    assert codebook.shape == (12, 16)
    for column_idx in range(codebook.shape[1]):
        assert np.bincount(codebook[:, column_idx], minlength=2).tolist() == [6, 6]


def test_rotating_episode_indices_do_not_repeat_before_wrap() -> None:
    labels = np.repeat(np.arange(3), 10)
    rng = np.random.default_rng(1)
    class_orders, cursors = EncoderOnlyClassifier._make_rotation_state(labels, rng)

    support_idx, query_idx = EncoderOnlyClassifier._rotating_episode_indices(
        labels=labels,
        rng=rng,
        batch_size=12,
        support_query_ratio=0.5,
        class_orders=class_orders,
        cursors=cursors,
    )

    selected = np.concatenate([support_idx, query_idx])
    assert len(selected) == len(np.unique(selected))
    assert len(support_idx) == 6
    assert len(query_idx) == 6
    assert np.bincount(labels[support_idx], minlength=3).tolist() == [2, 2, 2]
    assert np.bincount(labels[query_idx], minlength=3).tolist() == [2, 2, 2]
