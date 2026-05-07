from tabpfn_feature_encoder.data import parallel


def test_data_worker_count_uses_detected_cpu_count(monkeypatch) -> None:
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 8)

    assert parallel.detected_cpu_count() == 8
    assert parallel.data_worker_count(2) == 2
    assert parallel.data_worker_count(12) == 8


def test_data_worker_count_handles_missing_cpu_count(monkeypatch) -> None:
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: None)

    assert parallel.detected_cpu_count() == 1
    assert parallel.data_worker_count(0) == 0
    assert parallel.data_worker_count(3) == 1


def test_parallel_map_preserves_order_with_one_worker(monkeypatch) -> None:
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 1)

    assert parallel.parallel_map(str, [3, 2, 1]) == ["3", "2", "1"]
