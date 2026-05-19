from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from full_space_pcdmd.config import RunConfig
from pcdmd_common.cases.base_case import FullSpaceCase

_MIN_COVARIANCE_EIGENVALUE = 1e-12
_COVARIANCE_JITTER_SCALE = 1e-10
_MAX_INNOVATION_CONDITION_NUMBER = 1e12


def _measurement_dimension(case: FullSpaceCase, reference_state: np.ndarray) -> int:
    measurement_size = getattr(case, "measurement_size", None)
    if callable(measurement_size):
        return int(measurement_size())
    return int(case.jacobian(reference_state).shape[0])


def _set_case_transition_times(
    case: FullSpaceCase,
    time_next: float | None,
    time_prev: float | None,
) -> None:
    setter = getattr(case, "set_transition_times", None)
    if not callable(setter):
        return
    if time_next is None or time_prev is None:
        clear = getattr(case, "clear_transition_times", None)
        if callable(clear):
            clear()
        return
    setter(float(time_next), float(time_prev))


def _evaluate_residual_and_jacobian(
    case: FullSpaceCase,
    predicted_state: np.ndarray,
    previous_state: np.ndarray,
    *,
    time_next: float | None = None,
    time_prev: float | None = None,
):
    _set_case_transition_times(case, time_next, time_prev)
    combined_evaluator = getattr(case, "residual_and_jacobian", None)
    if callable(combined_evaluator):
        return combined_evaluator(predicted_state, previous_state)
    return case.residual(predicted_state, previous_state), case.jacobian(predicted_state)


def _as_dense_array(matrix_like) -> np.ndarray:
    if hasattr(matrix_like, "toarray"):
        matrix_like = matrix_like.toarray()
    return np.asarray(np.real_if_close(matrix_like), dtype=float)


def _symmetrize_covariance(covariance: np.ndarray) -> np.ndarray:
    return np.real_if_close(0.5 * (covariance + covariance.T))


def _covariance_scale(covariance: np.ndarray) -> float:
    diagonal = np.abs(np.diag(covariance))
    trace_scale = abs(float(np.trace(covariance))) / max(covariance.shape[0], 1)
    diagonal_scale = float(np.max(diagonal)) if diagonal.size else 0.0
    return max(trace_scale, diagonal_scale, 1.0)


def _regularize_covariance(
    covariance: np.ndarray,
    *,
    min_eigenvalue: float = _MIN_COVARIANCE_EIGENVALUE,
    jitter_scale: float = _COVARIANCE_JITTER_SCALE,
) -> np.ndarray:
    regularized = _symmetrize_covariance(np.asarray(np.real_if_close(covariance)))
    scale = _covariance_scale(regularized)
    regularized = regularized + np.eye(regularized.shape[0]) * (jitter_scale * scale)
    min_observed_eigenvalue = float(np.min(np.linalg.eigvalsh(regularized)))
    if min_observed_eigenvalue < min_eigenvalue:
        regularized = regularized + np.eye(regularized.shape[0]) * (
            min_eigenvalue - min_observed_eigenvalue
        )
    return _symmetrize_covariance(regularized)


def _solve_innovation_system(innovation_covariance: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    symmetric_innovation = _symmetrize_covariance(np.asarray(np.real_if_close(innovation_covariance)))
    regularized_innovation = _regularize_covariance(symmetric_innovation)
    condition_number = float(np.linalg.cond(regularized_innovation))
    if not np.isfinite(condition_number) or condition_number > _MAX_INNOVATION_CONDITION_NUMBER:
        return np.linalg.pinv(regularized_innovation) @ rhs
    try:
        cholesky_factor = np.linalg.cholesky(regularized_innovation)
        triangular_solution = np.linalg.solve(cholesky_factor, rhs)
        return np.linalg.solve(cholesky_factor.T, triangular_solution)
    except np.linalg.LinAlgError:
        pass
    try:
        return np.linalg.solve(regularized_innovation, rhs)
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(regularized_innovation) @ rhs


def _assemble_innovation_terms(
    predicted_covariance: np.ndarray,
    jacobian,
    measurement_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dense_jacobian = _as_dense_array(jacobian)
    hp_product = _as_dense_array(jacobian @ predicted_covariance)
    innovation_covariance = _regularize_covariance(
        hp_product @ dense_jacobian.T + measurement_error
    )
    cross_covariance = hp_product.T
    return innovation_covariance, cross_covariance


def _joseph_covariance_update(
    predicted_covariance: np.ndarray,
    jacobian,
    kalman_gain: np.ndarray,
    measurement_error: np.ndarray,
) -> np.ndarray:
    dense_jacobian = _as_dense_array(jacobian)
    identity = np.eye(predicted_covariance.shape[0])
    residual_factor = identity - kalman_gain @ dense_jacobian
    posterior_covariance = (
        residual_factor @ predicted_covariance @ residual_factor.T
        + kalman_gain @ measurement_error @ kalman_gain.T
    )
    return _symmetrize_covariance(posterior_covariance)


def estimate_filter_scales(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    a_matrix: np.ndarray,
    train_stop: int,
    time_values: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    train_snapshots = snapshots[:, : train_stop + 1]
    covariance = _regularize_covariance(np.cov(train_snapshots))

    state_scale = max(float(np.var(train_snapshots)), 1e-10)

    residual_samples = []
    for k in range(train_stop):
        u_prev = train_snapshots[:, k]
        u_pred = np.real_if_close(a_matrix @ u_prev)
        u_pred = case.enforce_constraints(u_pred)
        _set_case_transition_times(
            case,
            None if time_values is None else float(time_values[k + 1]),
            None if time_values is None else float(time_values[k]),
        )
        residual_samples.append(case.residual(u_pred, u_prev))

    residual_array = np.column_stack(residual_samples)
    residual_scale = max(float(np.var(residual_array)), 1e-12)
    return covariance, state_scale, residual_scale


@dataclass(frozen=True)
class PreparedFilterArtifacts:
    initial_covariance: np.ndarray
    process_error: np.ndarray
    measurement_error: np.ndarray


def prepare_filter_artifacts(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    a_matrix: np.ndarray,
    run_config: RunConfig,
    alpha: float,
    *,
    time_values: np.ndarray | None = None,
) -> PreparedFilterArtifacts:
    n_state = snapshots.shape[0]
    measurement_state = case.enforce_constraints(np.real_if_close(snapshots[:, 0]))
    measurement_dim = _measurement_dimension(case, measurement_state)
    state_covariance, state_scale, residual_scale = estimate_filter_scales(
        case,
        snapshots,
        a_matrix,
        run_config.train_stop,
        time_values,
    )

    q_weight = max(1.0 - alpha, 1e-8)
    r_weight = max(alpha, 1e-8)
    process_error = np.eye(n_state) * state_scale * q_weight
    measurement_error = np.eye(measurement_dim) * residual_scale * r_weight
    return PreparedFilterArtifacts(
        initial_covariance=_regularize_covariance(state_covariance),
        process_error=process_error,
        measurement_error=measurement_error,
    )


def run_pcdmd(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    a_matrix: np.ndarray,
    run_config: RunConfig,
    alpha: float,
    *,
    time_values: np.ndarray | None = None,
    prepared_artifacts: PreparedFilterArtifacts | None = None,
) -> np.ndarray:
    active_artifacts = prepared_artifacts
    if active_artifacts is None:
        active_artifacts = prepare_filter_artifacts(
            case,
            snapshots,
            a_matrix,
            run_config,
            alpha,
            time_values=time_values,
        )

    prediction = np.zeros_like(snapshots)
    prediction[:, 0] = snapshots[:, 0]

    current_state = prediction[:, 0].copy()
    current_covariance = active_artifacts.initial_covariance.copy()

    for k in range(snapshots.shape[1] - 1):
        predicted_state = np.real_if_close(a_matrix @ current_state)
        predicted_state = case.enforce_constraints(predicted_state)

        predicted_covariance = _regularize_covariance(
            a_matrix @ current_covariance @ a_matrix.T + active_artifacts.process_error
        )
        residual, jacobian = _evaluate_residual_and_jacobian(
            case,
            predicted_state,
            current_state,
            time_next=None if time_values is None else float(time_values[k + 1]),
            time_prev=None if time_values is None else float(time_values[k]),
        )
        innovation_covariance, cross_covariance = _assemble_innovation_terms(
            predicted_covariance,
            jacobian,
            active_artifacts.measurement_error,
        )
        kalman_gain = _solve_innovation_system(
            innovation_covariance,
            cross_covariance.T,
        ).T

        corrected_state = predicted_state - kalman_gain @ residual
        corrected_state = np.real_if_close(corrected_state)
        corrected_state = case.enforce_constraints(corrected_state)

        current_covariance = _regularize_covariance(
            _joseph_covariance_update(
                predicted_covariance,
                jacobian,
                kalman_gain,
                active_artifacts.measurement_error,
            )
        )
        current_state = corrected_state
        prediction[:, k + 1] = corrected_state

    return prediction
