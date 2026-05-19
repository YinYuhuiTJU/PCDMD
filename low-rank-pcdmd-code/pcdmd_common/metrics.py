from __future__ import annotations

import numpy as np


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(reference, axis=0)
    denominator = np.where(denominator > 1e-12, denominator, 1e-12)
    return np.linalg.norm(estimate - reference, axis=0) / denominator


def summarize_errors(errors: np.ndarray, train_stop: int) -> dict[str, float]:
    return {
        "reconstruction_mean": float(np.mean(errors[: train_stop + 1])),
        "prediction_mean": float(np.mean(errors[train_stop + 1 :])),
        "final_step": float(errors[-1]),
    }
