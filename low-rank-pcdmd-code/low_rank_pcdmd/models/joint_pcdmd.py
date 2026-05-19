from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from low_rank_pcdmd.models.correction_dmd import (
    CorrectionDMDArtifacts,
    build_disabled_correction_dmd,
)
from low_rank_pcdmd.models.prediction_dmd import DMDModelArtifacts
from low_rank_pcdmd.models.state_mapping import fit_reduced_coordinates, reconstruct_full_state
from pcdmd_common.cases.base_case import FullSpaceCase

_COVARIANCE_JITTER = 1e-10


def _symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.real_if_close(0.5 * (matrix + matrix.T))


def _regularization_scale(matrix: np.ndarray) -> float:
    diagonal = np.abs(np.diag(matrix))
    trace_scale = abs(float(np.trace(matrix))) / max(matrix.shape[0], 1)
    diagonal_scale = float(np.max(diagonal)) if diagonal.size else 0.0
    return max(trace_scale, diagonal_scale, 1.0)


def _regularize_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    symmetric = _symmetrize_matrix(np.asarray(np.real_if_close(matrix), dtype=float))
    scale = _regularization_scale(symmetric)
    return _symmetrize_matrix(symmetric + np.eye(symmetric.shape[0]) * (_COVARIANCE_JITTER * scale))


def _solve_regularized_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    regularized = _regularize_symmetric_matrix(matrix)
    try:
        cholesky_factor = np.linalg.cholesky(regularized)
        triangular_solution = np.linalg.solve(cholesky_factor, rhs)
        return np.linalg.solve(cholesky_factor.T, triangular_solution)
    except np.linalg.LinAlgError:
        pass
    try:
        return np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized) @ rhs


def _invert_regularized_system(matrix: np.ndarray) -> np.ndarray:
    return _solve_regularized_system(matrix, np.eye(matrix.shape[0], dtype=float))


def _stable_covariance(samples: np.ndarray) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(samples))
    return _regularize_symmetric_matrix(covariance)


def _block_diagonal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [left, np.zeros((left.shape[0], right.shape[1]))],
            [np.zeros((right.shape[0], left.shape[1])), right],
        ]
    )


def _fit_joint_coordinates(
    state: np.ndarray,
    prediction_basis: np.ndarray,
    correction_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    combined_basis = np.column_stack((prediction_basis, correction_basis))
    joint_coordinates = fit_reduced_coordinates(state, combined_basis)
    split_index = prediction_basis.shape[1]
    return joint_coordinates[:split_index], joint_coordinates[split_index:]


def _measurement_dimension(case: FullSpaceCase, reference_state: np.ndarray) -> int:
    measurement_size = getattr(case, "measurement_size", None)
    if callable(measurement_size):
        return int(measurement_size())
    return int(case.jacobian(reference_state).shape[0])


def _measurement_precision_vector(
    case: FullSpaceCase,
    measurement_dim: int,
    measurement_variance: float,
    run_config,
) -> np.ndarray:
    base_precision = 1.0 / max(measurement_variance, 1e-12)
    precision = np.full(measurement_dim, base_precision, dtype=float)
    pressure_weight = float(getattr(run_config, "pressure_measurement_weight", 1.0))
    if np.isclose(pressure_weight, 1.0):
        return precision
    block_slices = getattr(case, "measurement_block_slices", None)
    if not callable(block_slices):
        return precision
    slices = block_slices()
    pressure_slice = slices.get("p")
    if pressure_slice is not None:
        precision[pressure_slice] *= pressure_weight
    return precision


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
        residual, jacobian = combined_evaluator(predicted_state, previous_state)
        return np.asarray(np.real_if_close(residual), dtype=float), jacobian

    residual = case.residual(predicted_state, previous_state)
    jacobian = case.jacobian(predicted_state)
    return residual, jacobian


def _project_jacobian_to_basis(jacobian, basis: np.ndarray) -> np.ndarray:
    projected = jacobian @ basis
    return np.asarray(np.real_if_close(projected), dtype=float)


def _build_augmented_operator(
    prediction_model: DMDModelArtifacts,
    correction_model: CorrectionDMDArtifacts,
) -> np.ndarray:
    return _block_diagonal(prediction_model.operator, correction_model.model.operator)


def _resolve_correction_model(
    snapshots: np.ndarray,
    correction_model: CorrectionDMDArtifacts,
    train_stop: int,
    use_correction_basis: bool,
) -> CorrectionDMDArtifacts:
    if use_correction_basis:
        return correction_model
    return build_disabled_correction_dmd(snapshots.shape[0], train_stop)


def _select_residual_sample_steps(
    train_stop: int,
    residual_sample_count: int | None,
) -> np.ndarray:
    if train_stop <= 0:
        return np.zeros(0, dtype=int)
    if residual_sample_count is None or residual_sample_count >= train_stop:
        return np.arange(train_stop, dtype=int)
    sample_count = max(int(residual_sample_count), 1)
    sampled_steps = np.linspace(0, train_stop - 1, sample_count, dtype=int)
    return np.unique(sampled_steps)


def estimate_joint_filter_scales(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    prediction_model: DMDModelArtifacts,
    correction_model: CorrectionDMDArtifacts,
    train_stop: int,
    time_values: np.ndarray | None = None,
    residual_sample_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    prediction_coordinates = fit_reduced_coordinates(
        snapshots[:, : train_stop + 1],
        prediction_model.basis,
    )
    correction_coordinates = fit_reduced_coordinates(
        correction_model.processed_snapshots,
        correction_model.model.basis,
    )

    prediction_covariance = _stable_covariance(prediction_coordinates)
    correction_covariance = _stable_covariance(correction_coordinates)
    initial_covariance = _block_diagonal(prediction_covariance, correction_covariance)

    prediction_model_errors = (
        prediction_coordinates[:, 1:]
        - prediction_model.operator @ prediction_coordinates[:, :-1]
    )
    correction_model_errors = (
        correction_coordinates[:, 1:]
        - correction_model.model.operator @ correction_coordinates[:, :-1]
    )
    prediction_process_covariance = _stable_covariance(prediction_model_errors)
    correction_process_covariance = _stable_covariance(correction_model_errors)
    process_covariance = _block_diagonal(
        prediction_process_covariance,
        correction_process_covariance,
    )

    residual_samples = []
    residual_sample_steps = _select_residual_sample_steps(train_stop, residual_sample_count)
    for step in residual_sample_steps:
        current_state = snapshots[:, step]
        predicted_z = fit_reduced_coordinates(current_state, prediction_model.basis)
        predicted_z = np.real_if_close(prediction_model.operator @ predicted_z)
        predicted_state = reconstruct_full_state(predicted_z, prediction_model.basis)
        predicted_state = case.enforce_constraints(np.real_if_close(predicted_state))
        time_prev = None if time_values is None else float(time_values[step])
        time_next = None if time_values is None else float(time_values[step + 1])
        _set_case_transition_times(case, time_next, time_prev)
        residual_samples.append(case.residual(predicted_state, current_state))

    residual_array = np.column_stack(residual_samples)
    residual_scale = max(float(np.var(residual_array)), 1e-12)
    return initial_covariance, process_covariance, residual_scale


@dataclass(frozen=True)
class PreparedReducedPCDMDArtifacts:
    active_correction_model: CorrectionDMDArtifacts
    initial_covariance: np.ndarray
    process_error: np.ndarray
    measurement_precision: np.ndarray
    combined_basis: np.ndarray
    augmented_operator: np.ndarray


def prepare_joint_reduced_pcdmd_artifacts(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    prediction_model: DMDModelArtifacts,
    correction_model: CorrectionDMDArtifacts,
    run_config,
    *,
    time_values: np.ndarray | None = None,
) -> PreparedReducedPCDMDArtifacts:
    active_correction_model = _resolve_correction_model(
        snapshots,
        correction_model,
        run_config.train_stop,
        run_config.use_correction_basis,
    )
    initial_covariance, process_covariance_base, residual_scale = estimate_joint_filter_scales(
        case,
        snapshots,
        prediction_model,
        active_correction_model,
        run_config.train_stop,
        time_values,
        getattr(run_config, "artifact_residual_sample_count", None),
    )
    augmented_operator = _build_augmented_operator(prediction_model, active_correction_model)
    combined_basis = np.column_stack((prediction_model.basis, active_correction_model.model.basis))

    q_weight = max(1.0 - run_config.alpha, 1e-8)
    r_weight = max(run_config.alpha, 1e-8)
    process_error = process_covariance_base * q_weight
    measurement_state = case.enforce_constraints(np.real_if_close(snapshots[:, 0]))
    measurement_dim = _measurement_dimension(case, measurement_state)
    measurement_variance = max(residual_scale * r_weight, 1e-12)
    measurement_precision = _measurement_precision_vector(
        case,
        measurement_dim,
        measurement_variance,
        run_config,
    )
    return PreparedReducedPCDMDArtifacts(
        active_correction_model=active_correction_model,
        initial_covariance=_regularize_symmetric_matrix(initial_covariance),
        process_error=process_error,
        measurement_precision=measurement_precision,
        combined_basis=combined_basis,
        augmented_operator=augmented_operator,
    )


@dataclass(frozen=True)
class ReducedPCDMDDiagnostics:
    mean_delta_z_norm: float
    max_delta_z_norm: float
    mean_delta_eta_norm: float
    max_delta_eta_norm: float
    mean_predicted_residual_norm: float
    mean_corrected_residual_norm: float
    residual_reduction_fraction: float
    mean_predicted_covariance_trace: float
    mean_corrected_covariance_trace: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_delta_z_norm": self.mean_delta_z_norm,
            "max_delta_z_norm": self.max_delta_z_norm,
            "mean_delta_eta_norm": self.mean_delta_eta_norm,
            "max_delta_eta_norm": self.max_delta_eta_norm,
            "mean_predicted_residual_norm": self.mean_predicted_residual_norm,
            "mean_corrected_residual_norm": self.mean_corrected_residual_norm,
            "residual_reduction_fraction": self.residual_reduction_fraction,
            "mean_predicted_covariance_trace": self.mean_predicted_covariance_trace,
            "mean_corrected_covariance_trace": self.mean_corrected_covariance_trace,
        }


def _run_joint_reduced_pcdmd(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    prediction_model: DMDModelArtifacts,
    correction_model: CorrectionDMDArtifacts,
    run_config,
    *,
    time_values: np.ndarray | None,
    initial_eta: np.ndarray | None,
    prepared_artifacts: PreparedReducedPCDMDArtifacts | None,
    collect_diagnostics: bool,
) -> tuple[np.ndarray, dict[str, float] | None]:
    if not 0.0 < run_config.correction_damping <= 1.0:
        raise ValueError("correction_damping must be in (0, 1].")

    active_artifacts = prepared_artifacts
    if active_artifacts is None:
        active_artifacts = prepare_joint_reduced_pcdmd_artifacts(
            case,
            snapshots,
            prediction_model,
            correction_model,
            run_config,
            time_values=time_values,
        )

    active_correction_model = active_artifacts.active_correction_model
    combined_basis = active_artifacts.combined_basis
    augmented_operator = active_artifacts.augmented_operator

    prediction = np.zeros_like(snapshots)
    prediction[:, 0] = snapshots[:, 0]

    current_state = snapshots[:, 0].copy()
    current_z = fit_reduced_coordinates(current_state, prediction_model.basis)
    if initial_eta is None:
        current_eta = np.zeros(active_correction_model.model.effective_rank, dtype=float)
    else:
        current_eta = np.asarray(np.real_if_close(initial_eta), dtype=float).reshape(
            active_correction_model.model.effective_rank
        )
    current_covariance = active_artifacts.initial_covariance.copy()

    delta_z_norms: list[float] = []
    delta_eta_norms: list[float] = []
    predicted_residual_norms: list[float] = []
    corrected_residual_norms: list[float] = []
    predicted_covariance_traces: list[float] = []
    corrected_covariance_traces: list[float] = []
    residual_reductions = 0

    for step in range(snapshots.shape[1] - 1):
        previous_state = current_state
        current_joint_state = np.concatenate((current_z, current_eta))
        predicted_joint_state = np.real_if_close(augmented_operator @ current_joint_state)
        predicted_covariance = _regularize_symmetric_matrix(
            augmented_operator @ current_covariance @ augmented_operator.T
            + active_artifacts.process_error
        )

        predicted_state = reconstruct_full_state(predicted_joint_state, combined_basis)
        predicted_state = case.enforce_constraints(np.real_if_close(predicted_state))

        residual, jacobian = _evaluate_residual_and_jacobian(
            case,
            predicted_state,
            previous_state,
            time_next=None if time_values is None else float(time_values[step + 1]),
            time_prev=None if time_values is None else float(time_values[step]),
        )
        reduced_jacobian = _project_jacobian_to_basis(jacobian, combined_basis)

        weighted_jacobian = active_artifacts.measurement_precision[:, None] * reduced_jacobian
        measurement_matrix = reduced_jacobian.T @ weighted_jacobian
        normal_matrix = measurement_matrix
        if run_config.use_prior_term:
            prior_precision = _invert_regularized_system(predicted_covariance)
            normal_matrix = prior_precision + measurement_matrix
        normal_matrix = _regularize_symmetric_matrix(normal_matrix)
        normal_rhs = -(reduced_jacobian.T @ (active_artifacts.measurement_precision * residual))
        joint_delta = _solve_regularized_system(normal_matrix, normal_rhs)
        split_index = prediction_model.effective_rank
        delta_z = joint_delta[:split_index]
        delta_eta = joint_delta[split_index:]

        corrected_joint_state = np.real_if_close(
            predicted_joint_state + run_config.correction_damping * joint_delta
        )
        reconstructed_corrected_state = np.asarray(
            np.real_if_close(reconstruct_full_state(corrected_joint_state, combined_basis)),
            dtype=float,
        )
        corrected_state = case.enforce_constraints(reconstructed_corrected_state)

        current_covariance = _regularize_symmetric_matrix(
            _invert_regularized_system(normal_matrix)
        )

        if np.array_equal(corrected_state, reconstructed_corrected_state):
            current_z = np.asarray(np.real_if_close(corrected_joint_state[:split_index]), dtype=float)
            current_eta = np.asarray(np.real_if_close(corrected_joint_state[split_index:]), dtype=float)
        else:
            current_z, current_eta = _fit_joint_coordinates(
                corrected_state,
                prediction_model.basis,
                active_correction_model.model.basis,
            )
        current_state = corrected_state
        prediction[:, step + 1] = corrected_state

        if collect_diagnostics:
            corrected_residual = case.residual(corrected_state, previous_state)
            predicted_residual_norm = float(np.linalg.norm(residual))
            corrected_residual_norm = float(np.linalg.norm(corrected_residual))
            predicted_residual_norms.append(predicted_residual_norm)
            corrected_residual_norms.append(corrected_residual_norm)
            predicted_covariance_traces.append(float(np.trace(predicted_covariance)))
            corrected_covariance_traces.append(float(np.trace(current_covariance)))
            delta_z_norms.append(float(np.linalg.norm(delta_z)))
            delta_eta_norms.append(float(np.linalg.norm(delta_eta)))
            if corrected_residual_norm < predicted_residual_norm:
                residual_reductions += 1

    if not collect_diagnostics:
        return prediction, None

    diagnostics = ReducedPCDMDDiagnostics(
        mean_delta_z_norm=float(np.mean(delta_z_norms)),
        max_delta_z_norm=float(np.max(delta_z_norms)),
        mean_delta_eta_norm=float(np.mean(delta_eta_norms)),
        max_delta_eta_norm=float(np.max(delta_eta_norms)),
        mean_predicted_residual_norm=float(np.mean(predicted_residual_norms)),
        mean_corrected_residual_norm=float(np.mean(corrected_residual_norms)),
        residual_reduction_fraction=float(residual_reductions / max(len(delta_z_norms), 1)),
        mean_predicted_covariance_trace=float(np.mean(predicted_covariance_traces)),
        mean_corrected_covariance_trace=float(np.mean(corrected_covariance_traces)),
    )
    return prediction, diagnostics.to_dict()


def run_joint_reduced_pcdmd_with_diagnostics(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    prediction_model: DMDModelArtifacts,
    correction_model: CorrectionDMDArtifacts,
    run_config,
    *,
    time_values: np.ndarray | None = None,
    initial_eta: np.ndarray | None = None,
    prepared_artifacts: PreparedReducedPCDMDArtifacts | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    prediction, diagnostics = _run_joint_reduced_pcdmd(
        case,
        snapshots,
        prediction_model,
        correction_model,
        run_config,
        time_values=time_values,
        initial_eta=initial_eta,
        prepared_artifacts=prepared_artifacts,
        collect_diagnostics=True,
    )
    assert diagnostics is not None
    return prediction, diagnostics


def run_joint_reduced_pcdmd(
    case: FullSpaceCase,
    snapshots: np.ndarray,
    prediction_model: DMDModelArtifacts,
    correction_model: CorrectionDMDArtifacts,
    run_config,
    *,
    time_values: np.ndarray | None = None,
    initial_eta: np.ndarray | None = None,
    prepared_artifacts: PreparedReducedPCDMDArtifacts | None = None,
) -> np.ndarray:
    prediction, _ = _run_joint_reduced_pcdmd(
        case,
        snapshots,
        prediction_model,
        correction_model,
        run_config,
        time_values=time_values,
        initial_eta=initial_eta,
        prepared_artifacts=prepared_artifacts,
        collect_diagnostics=False,
    )
    return prediction
