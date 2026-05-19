from __future__ import annotations

from typing import Protocol

import numpy as np

from pcdmd_common.cases.base_case import FullSpaceCase


class DiffusionGridConfig(Protocol):
    x_min: float
    x_max: float
    t_end: float
    nx: int
    n_steps: int
    nu: float

    @property
    def dx(self) -> float:
        ...

    @property
    def dt(self) -> float:
        ...


def initial_condition(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.exp(-((x - 1.0) ** 2) / (0.05**2))


def solve_tridiagonal_system(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal linear system with the Thomas algorithm."""

    upper_work = np.array(upper, copy=True, dtype=float)
    diagonal_work = np.array(diagonal, copy=True, dtype=float)
    rhs_work = np.array(rhs, copy=True, dtype=float)

    for idx in range(1, diagonal_work.size):
        multiplier = lower[idx - 1] / diagonal_work[idx - 1]
        diagonal_work[idx] -= multiplier * upper_work[idx - 1]
        rhs_work[idx] -= multiplier * rhs_work[idx - 1]

    solution = np.empty_like(rhs_work)
    solution[-1] = rhs_work[-1] / diagonal_work[-1]
    for idx in range(diagonal_work.size - 2, -1, -1):
        solution[idx] = (rhs_work[idx] - upper_work[idx] * solution[idx + 1]) / diagonal_work[idx]

    return solution


class Diffusion1DCase(FullSpaceCase):
    def __init__(self, config: DiffusionGridConfig) -> None:
        self.config = config
        self._x = np.linspace(config.x_min, config.x_max, config.nx + 1)
        self._t = np.linspace(0.0, config.t_end, config.n_steps + 1)
        self._jacobian = self._build_diffusion_jacobian()
        self._lower_diagonal, self._main_diagonal, self._upper_diagonal = (
            self._build_implicit_step_diagonals()
        )

    @property
    def state_size(self) -> int:
        return self._x.size

    def simulate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        snapshots = np.zeros((self._x.size, self._t.size), dtype=float)
        snapshots[:, 0] = initial_condition(self._x)

        for k in range(self.config.n_steps):
            u = snapshots[:, k].copy()
            u_next = u.copy()
            u_next[1:-1] = solve_tridiagonal_system(
                self._lower_diagonal,
                self._main_diagonal,
                self._upper_diagonal,
                u[1:-1],
            )
            snapshots[:, k + 1] = self.enforce_constraints(u_next)

        return self._x.copy(), self._t.copy(), snapshots

    def residual(self, u_next: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        laplace_u = (u_next[2:] - 2.0 * u_next[1:-1] + u_next[:-2]) / (self.config.dx**2)
        return (u_next[1:-1] - u_prev[1:-1]) / self.config.dt - self.config.nu * laplace_u

    def jacobian(self, u_next: np.ndarray | None = None) -> np.ndarray:
        return self._jacobian.copy()

    def enforce_constraints(self, state: np.ndarray) -> np.ndarray:
        constrained = np.array(state, copy=True)
        constrained[0] = 0.0
        constrained[-1] = 0.0
        return constrained

    def default_snapshot_indices(self) -> tuple[int, int]:
        return self.config.n_steps // 2, self.config.n_steps

    def _build_diffusion_jacobian(self) -> np.ndarray:
        n_interior = self.state_size - 2
        jacobian = np.zeros((n_interior, self.state_size), dtype=float)
        laplace_coeff = self.config.nu / (self.config.dx**2)
        for row, idx in enumerate(range(1, self.state_size - 1)):
            jacobian[row, idx - 1] = -laplace_coeff
            jacobian[row, idx] = 1.0 / self.config.dt + 2.0 * laplace_coeff
            jacobian[row, idx + 1] = -laplace_coeff
        return jacobian

    def _build_implicit_step_diagonals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_interior = self.state_size - 2
        coeff = self.config.nu * self.config.dt / (self.config.dx**2)
        lower = np.full(n_interior - 1, -coeff, dtype=float)
        main = np.full(n_interior, 1.0 + 2.0 * coeff, dtype=float)
        upper = np.full(n_interior - 1, -coeff, dtype=float)
        return lower, main, upper
