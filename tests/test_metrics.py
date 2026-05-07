import numpy as np

from tabpfn_feature_encoder.evaluation.metrics import accuracy, binary_roc_auc, log_loss


def test_accuracy() -> None:
    assert accuracy([0, 1, 1], [0, 1, 0]) == 2 / 3


def test_binary_roc_auc_perfect() -> None:
    y = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])

    assert binary_roc_auc(y, scores) == 1.0


def test_log_loss_is_finite() -> None:
    y = np.asarray([0, 1])
    proba = np.asarray([[0.9, 0.1], [0.2, 0.8]])

    assert np.isfinite(log_loss(y, proba))
