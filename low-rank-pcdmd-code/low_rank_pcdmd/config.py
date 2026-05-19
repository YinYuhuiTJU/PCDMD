from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_ROOT.parent / "outputs" / "low_rank_diffusion_1d"


@dataclass(frozen=True)
class DiffusionConfig:
    x_min: float = 0.0
    x_max: float = 2.0
    t_end: float = 2.0
    nx: int = 200
    n_steps: int = 500
    nu: float = 0.01

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dt(self) -> float:
        return self.t_end / self.n_steps


@dataclass(frozen=True)
class RunConfig:
    train_stop: int = 200
    rank: int = 5
    correction_rank: int | None = None
    alpha: float = 0.01
    correction_damping: float = 0.01
    pressure_measurement_weight: float = 1.0
    artifact_residual_sample_count: int | None = None
    use_prior_term: bool = True
    use_correction_basis: bool = True
    benchmark_repeats: int = 3
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def active_correction_rank(self) -> int:
        return self.rank if self.correction_rank is None else self.correction_rank
