from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_basis(basis_like: Any) -> np.ndarray:
    return basis_like.basis if hasattr(basis_like, "basis") else np.asarray(basis_like, dtype=float)


def fit_reduced_coordinates(full_state: np.ndarray, basis_like: Any) -> np.ndarray:
    basis = _resolve_basis(basis_like)
    targets = np.asarray(full_state, dtype=float)
    coordinates, _, _, _ = np.linalg.lstsq(basis, targets, rcond=None)
    return np.real_if_close(coordinates)


def reconstruct_full_state(reduced_state: np.ndarray, basis_like: Any) -> np.ndarray:
    basis = _resolve_basis(basis_like)
    return np.real_if_close(basis @ np.asarray(reduced_state, dtype=float))


def project_to_reduced(full_state: np.ndarray, basis_like: Any) -> np.ndarray:
    return fit_reduced_coordinates(full_state, basis_like)
