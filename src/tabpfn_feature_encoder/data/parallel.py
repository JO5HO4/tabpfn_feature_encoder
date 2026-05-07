from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def detected_cpu_count() -> int:
    return os.cpu_count() or 1


def data_worker_count(n_tasks: int) -> int:
    if n_tasks <= 0:
        return 0
    return min(detected_cpu_count(), int(n_tasks))


def parallel_map(
    fn: Callable[[T], U],
    tasks: Iterable[T],
    *,
    workers: int | None = None,
) -> list[U]:
    task_list = list(tasks)
    if not task_list:
        return []

    worker_count = data_worker_count(len(task_list)) if workers is None else int(workers)
    worker_count = max(1, min(worker_count, len(task_list)))
    if worker_count == 1:
        return [fn(task) for task in task_list]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fn, task_list))
