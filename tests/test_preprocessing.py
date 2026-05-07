import numpy as np

from tabpfn_feature_encoder.data.preprocessing import (
    Standardizer,
    stratified_sample_indices,
    stratified_split_indices,
)


def test_standardizer_handles_constant_columns() -> None:
    X = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]], dtype=np.float32)
    standardizer = Standardizer().fit(X)

    transformed = standardizer.transform(X)

    assert np.allclose(transformed[:, 0], 0.0)
    assert np.isfinite(transformed).all()


def test_stratified_split_preserves_classes() -> None:
    y = np.asarray([0] * 10 + [1] * 10)

    train_idx, test_idx = stratified_split_indices(y, test_size=0.2, random_state=1)

    assert len(train_idx) == 16
    assert len(test_idx) == 4
    assert set(np.unique(y[train_idx])) == {0, 1}
    assert set(np.unique(y[test_idx])) == {0, 1}


def test_stratified_sample_is_exact_size() -> None:
    y = np.asarray([0] * 8 + [1] * 12)

    idx = stratified_sample_indices(y, 6, random_state=2)

    assert len(idx) == 6
    assert set(np.unique(y[idx])) == {0, 1}
