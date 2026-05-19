from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_metrics(output_path: Path, payload: dict) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_predictions(output_path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(output_path, **arrays)
