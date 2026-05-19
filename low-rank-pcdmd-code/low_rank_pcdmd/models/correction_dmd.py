from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from low_rank_pcdmd.models.prediction_dmd import DMDModelArtifacts, fit_dmd_model, predict_next_state
from low_rank_pcdmd.models.state_mapping import fit_reduced_coordinates, reconstruct_full_state


@dataclass(frozen=True)
class CorrectionDMDArtifacts:
    model: DMDModelArtifacts
    raw_snapshots: np.ndarray
    processed_snapshots: np.ndarray


def build_disabled_correction_dmd(
    state_dim: int,
    train_stop: int,
) -> CorrectionDMDArtifacts:
    empty_basis = np.zeros((state_dim, 0), dtype=float)
    empty_operator = np.zeros((0, 0), dtype=float)
    empty_snapshots = np.zeros((state_dim, train_stop), dtype=float)
    return CorrectionDMDArtifacts(
        model=DMDModelArtifacts(
            basis=empty_basis,
            operator=empty_operator,
            effective_rank=0,
        ),
        raw_snapshots=empty_snapshots,
        processed_snapshots=empty_snapshots,
    )


def remove_subspace_overlap(
    vectors: np.ndarray,
    reference_basis: np.ndarray,
) -> np.ndarray:
    projection_coordinates = fit_reduced_coordinates(vectors, reference_basis)
    projected_component = reconstruct_full_state(projection_coordinates, reference_basis)
    return np.real_if_close(np.asarray(vectors, dtype=float) - projected_component).astype(float)


def build_correction_snapshots(
    case,
    snapshots: np.ndarray,
    prediction_model: DMDModelArtifacts,
    train_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_corrections = np.zeros((snapshots.shape[0], train_stop), dtype=float)
    for step in range(train_stop):
        predicted_state, _, _ = predict_next_state(case, prediction_model, snapshots[:, step])
        raw_corrections[:, step] = np.real_if_close(snapshots[:, step + 1] - predicted_state)

    processed_corrections = remove_subspace_overlap(raw_corrections, prediction_model.basis)
    return raw_corrections, processed_corrections


def fit_correction_dmd(
    case,
    snapshots: np.ndarray,
    prediction_model: DMDModelArtifacts,
    train_stop: int,
    rank: int,
) -> CorrectionDMDArtifacts:
    raw_corrections, processed_corrections = build_correction_snapshots(
        case,
        snapshots,
        prediction_model,
        train_stop,
    )
    correction_model = fit_dmd_model(processed_corrections, rank)
    return CorrectionDMDArtifacts(
        model=correction_model,
        raw_snapshots=raw_corrections,
        processed_snapshots=processed_corrections,
    )
