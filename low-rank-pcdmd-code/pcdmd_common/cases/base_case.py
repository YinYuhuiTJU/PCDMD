from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FullSpaceCase(ABC):
    @property
    @abstractmethod
    def state_size(self) -> int:
        """Number of state variables in the full-space system."""

    @abstractmethod
    def simulate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return spatial grid, time grid, and state snapshots."""

    @abstractmethod
    def residual(self, u_next: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        """Compute the physics residual used in the PCDMD correction step."""

    @abstractmethod
    def jacobian(self, u_next: np.ndarray | None = None) -> np.ndarray:
        """Return the residual Jacobian, optionally state-dependent."""

    @abstractmethod
    def enforce_constraints(self, state: np.ndarray) -> np.ndarray:
        """Project a state back onto simple admissibility constraints."""

    @abstractmethod
    def default_snapshot_indices(self) -> tuple[int, int]:
        """Return default snapshot indices for comparison plots."""
