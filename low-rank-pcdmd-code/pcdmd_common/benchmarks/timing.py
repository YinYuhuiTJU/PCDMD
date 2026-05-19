from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Generic, TypeVar

ArtifactsT = TypeVar("ArtifactsT")


@dataclass(frozen=True)
class TimingSummary:
    offline_setup_s: float
    online_run_s: float
    total_s: float
    avg_step_s: float
    n_repeats: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def benchmark_workflow(
    *,
    setup_callable: Callable[[], ArtifactsT],
    online_callable: Callable[[ArtifactsT], object],
    n_online_steps: int,
    n_repeats: int = 3,
) -> TimingSummary:
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    if n_online_steps <= 0:
        raise ValueError("n_online_steps must be positive.")

    offline_times: list[float] = []
    online_times: list[float] = []

    for _ in range(n_repeats):
        start = perf_counter()
        artifacts = setup_callable()
        offline_times.append(perf_counter() - start)

        start = perf_counter()
        online_callable(artifacts)
        online_times.append(perf_counter() - start)

    offline_mean = sum(offline_times) / n_repeats
    online_mean = sum(online_times) / n_repeats
    total_mean = offline_mean + online_mean
    return TimingSummary(
        offline_setup_s=offline_mean,
        online_run_s=online_mean,
        total_s=total_mean,
        avg_step_s=online_mean / n_online_steps,
        n_repeats=n_repeats,
    )
