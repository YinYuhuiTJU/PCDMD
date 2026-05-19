from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from low_rank_pcdmd.models.state_mapping import fit_reduced_coordinates, reconstruct_full_state


@dataclass(frozen=True)
class DMDModelArtifacts:
    basis: np.ndarray
    operator: np.ndarray
    effective_rank: int


def _realify_dmd_modes(
    eigenvalues: np.ndarray,
    modes: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    basis_columns: list[np.ndarray] = []
    used_indices: set[int] = set()

    for index, eigenvalue in enumerate(eigenvalues):
        if index in used_indices:
            continue

        if abs(float(np.imag(eigenvalue))) <= tol:
            basis_columns.append(np.real_if_close(modes[:, index]).astype(float))
            used_indices.add(index)
            continue

        if float(np.imag(eigenvalue)) < 0.0:
            continue

        pair_index = None
        for candidate in range(index + 1, eigenvalues.size):
            if candidate in used_indices:
                continue
            if np.allclose(eigenvalues[candidate], np.conj(eigenvalue), atol=1e-8, rtol=1e-5):
                pair_index = candidate
                break

        if pair_index is None:
            basis_columns.append(np.real(modes[:, index]).astype(float))
            basis_columns.append(-np.imag(modes[:, index]).astype(float))
            used_indices.add(index)
            continue

        mode = modes[:, index]
        basis_columns.append(np.real(mode).astype(float))
        basis_columns.append(-np.imag(mode).astype(float))
        used_indices.add(index)
        used_indices.add(pair_index)

    if not basis_columns:
        raise ValueError("Unable to build a real-valued DMD basis from the provided snapshots.")

    return np.column_stack(basis_columns)


def _fit_operator_for_basis(snapshot_sequence: np.ndarray, basis: np.ndarray) -> np.ndarray:
    reduced_snapshots = fit_reduced_coordinates(snapshot_sequence, basis)
    x1 = reduced_snapshots[:, :-1]
    x2 = reduced_snapshots[:, 1:]
    operator = x2 @ np.linalg.pinv(x1)
    return np.real_if_close(operator).astype(float)


def fit_dmd_model(snapshot_sequence: np.ndarray, rank: int) -> DMDModelArtifacts:
    if snapshot_sequence.shape[1] < 2:
        raise ValueError("At least two snapshots are required to fit a DMD model.")

    x1 = snapshot_sequence[:, :-1]
    x2 = snapshot_sequence[:, 1:]
    u, singular_values, vh = np.linalg.svd(x1, full_matrices=False)
    effective_rank = min(rank, int(np.count_nonzero(singular_values > 1e-12)))
    if effective_rank < 1:
        raise ValueError("The requested DMD rank produced an empty reduced model.")

    u_r = u[:, :effective_rank]
    s_r = singular_values[:effective_rank]
    v_r = vh[:effective_rank, :].T
    a_tilde = u_r.T @ x2 @ v_r @ np.diag(1.0 / s_r)
    eigenvalues, eigenvectors = np.linalg.eig(a_tilde)
    modes = x2 @ v_r @ np.diag(1.0 / s_r) @ eigenvectors

    basis = _realify_dmd_modes(eigenvalues, modes)
    operator = _fit_operator_for_basis(snapshot_sequence, basis)
    return DMDModelArtifacts(
        basis=np.real_if_close(basis).astype(float),
        operator=operator,
        effective_rank=basis.shape[1],
    )


def fit_prediction_dmd(
    snapshots: np.ndarray,
    train_stop: int,
    rank: int,
) -> DMDModelArtifacts:
    return fit_dmd_model(snapshots[:, : train_stop + 1], rank)


def rollout_reduced_coordinates(
    operator: np.ndarray,
    reduced_initial_state: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    rollout = np.zeros((reduced_initial_state.size, n_steps + 1), dtype=float)
    rollout[:, 0] = np.real_if_close(reduced_initial_state)
    for step in range(n_steps):
        rollout[:, step + 1] = np.real_if_close(operator @ rollout[:, step])
    return rollout


def predict_next_state(case, model: DMDModelArtifacts, current_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_coordinates = fit_reduced_coordinates(current_state, model.basis)
    next_coordinates = np.real_if_close(model.operator @ current_coordinates)
    predicted_state = reconstruct_full_state(next_coordinates, model.basis)
    predicted_state = case.enforce_constraints(np.real_if_close(predicted_state))
    return predicted_state, current_coordinates, next_coordinates


def rollout_prediction_states(
    case,
    model: DMDModelArtifacts,
    initial_state: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    reduced_initial_state = fit_reduced_coordinates(initial_state, model.basis)
    reduced_rollout = rollout_reduced_coordinates(model.operator, reduced_initial_state, n_steps)
    full_prediction = reconstruct_full_state(reduced_rollout, model.basis)
    constrained = np.array(full_prediction, copy=True)
    constrained[:, 0] = case.enforce_constraints(np.asarray(initial_state, dtype=float).copy())
    for step in range(constrained.shape[1]):
        constrained[:, step] = case.enforce_constraints(np.real_if_close(constrained[:, step]))
    return np.real_if_close(constrained), reduced_rollout
