from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from full_space_pcdmd.config import DEFAULT_OUTPUT_DIR, DiffusionConfig, RunConfig
from full_space_pcdmd.models.dmd import fit_dmd, rollout_linear
from full_space_pcdmd.models.pcdmd import prepare_filter_artifacts, run_pcdmd
from pcdmd_common.cases.diffusion_1d import Diffusion1DCase
from pcdmd_common.metrics import relative_l2_error, summarize_errors
from pcdmd_common.outputs.export import ensure_output_dir, save_metrics, save_predictions
from pcdmd_common.outputs.plotters import plot_error_curves, plot_snapshots


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full-space PCDMD baseline for the 1D diffusion case."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, plots, and prediction arrays.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=RunConfig.alpha,
        help="Physics-correction weight for full-space PCDMD.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    case_config = DiffusionConfig()
    run_config = RunConfig(output_dir=args.output_dir, alpha=args.alpha)
    case = Diffusion1DCase(case_config)

    ensure_output_dir(run_config.output_dir)
    x, t, truth = case.simulate()

    a_dmd = fit_dmd(truth, run_config.train_stop, run_config.rank)
    dmd_prediction = rollout_linear(a_dmd, truth[:, 0], case_config.n_steps)
    dmd_error = relative_l2_error(truth, dmd_prediction)

    metrics = {
        "case_config": {
            **asdict(case_config),
            "dx": case_config.dx,
            "dt": case_config.dt,
        },
        "run_config": {
            "train_stop": run_config.train_stop,
            "rank": run_config.rank,
            "alpha": run_config.alpha,
            "output_dir": str(run_config.output_dir),
        },
        "dmd": summarize_errors(dmd_error, run_config.train_stop),
    }

    prepared_artifacts = prepare_filter_artifacts(case, truth, a_dmd, run_config, run_config.alpha)
    pcdmd_prediction = run_pcdmd(
        case,
        truth,
        a_dmd,
        run_config,
        run_config.alpha,
        prepared_artifacts=prepared_artifacts,
    )
    pcdmd_error = relative_l2_error(truth, pcdmd_prediction)
    metrics["pcdmd"] = summarize_errors(pcdmd_error, run_config.train_stop)

    plot_error_curves(
        t,
        dmd_error,
        {run_config.alpha: pcdmd_error},
        run_config.train_stop,
        run_config.output_dir / "diffusion_errors.png",
        rank=run_config.rank,
        best_alpha=run_config.alpha,
    )
    plot_snapshots(
        x,
        t,
        truth,
        dmd_prediction,
        pcdmd_prediction,
        case.default_snapshot_indices(),
        run_config.output_dir / "diffusion_snapshots.png",
        alpha=run_config.alpha,
        rank=run_config.rank,
    )
    save_metrics(run_config.output_dir / "metrics.json", metrics)
    save_predictions(
        run_config.output_dir / "predictions.npz",
        x=x,
        t=t,
        truth=truth,
        dmd=dmd_prediction,
        pcdmd=pcdmd_prediction,
    )

    print(f"Saved outputs to: {run_config.output_dir.resolve()}")
    print("DMD metrics:", json.dumps(metrics["dmd"], ensure_ascii=False))
    print(
        "PCDMD metrics:",
        json.dumps(metrics["pcdmd"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
