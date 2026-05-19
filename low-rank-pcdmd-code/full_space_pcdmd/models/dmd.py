from __future__ import annotations

import numpy as np


def fit_dmd(snapshots: np.ndarray, train_stop: int, rank: int) -> np.ndarray:
    x1 = snapshots[:, :train_stop]
    x2 = snapshots[:, 1 : train_stop + 1]
    u, singular_values, vh = np.linalg.svd(x1, full_matrices=False)
    effective_rank = min(rank, int(np.count_nonzero(singular_values > 1e-12)))
    u_r = u[:, :effective_rank]
    s_r = singular_values[:effective_rank]
    v_r = vh[:effective_rank, :].T
    a_dmd = x2 @ v_r @ np.diag(1.0 / s_r) @ u_r.T
    return np.real_if_close(a_dmd)


def rollout_linear(a_matrix: np.ndarray, x0: np.ndarray, n_steps: int) -> np.ndarray:
    prediction = np.zeros((x0.size, n_steps + 1), dtype=float)
    prediction[:, 0] = x0
    for k in range(n_steps):
        prediction[:, k + 1] = np.real_if_close(a_matrix @ prediction[:, k])
    return prediction
