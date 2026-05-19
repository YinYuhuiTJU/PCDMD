from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from full_space_pcdmd.config import RunConfig as FullSpaceRunConfig
from full_space_pcdmd.models.dmd import fit_dmd as fit_full_space_dmd
from full_space_pcdmd.models.pcdmd import run_pcdmd as run_full_space_pcdmd
from low_rank_pcdmd.config import DEFAULT_OUTPUT_DIR, DiffusionConfig, RunConfig
from low_rank_pcdmd.models.correction_dmd import build_disabled_correction_dmd, fit_correction_dmd
from low_rank_pcdmd.models.prediction_dmd import fit_prediction_dmd, rollout_prediction_states
from low_rank_pcdmd.models.reduced_pcdmd import (
    prepare_reduced_pcdmd_artifacts,
    run_reduced_pcdmd,
    run_reduced_pcdmd_with_diagnostics,
)
from pcdmd_common.benchmarks.timing import benchmark_workflow
from pcdmd_common.cases.diffusion_1d import Diffusion1DCase
from pcdmd_common.metrics import relative_l2_error, summarize_errors
from pcdmd_common.outputs.export import ensure_output_dir, save_metrics, save_predictions
from pcdmd_common.outputs.plotters import (
    plot_error_curves,
    plot_snapshots,
    plot_timing_comparison,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click reduced-space PCDMD runner for the 1D diffusion example."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nx", type=int, default=DiffusionConfig.nx)
    parser.add_argument("--n-steps", type=int, default=DiffusionConfig.n_steps)
    parser.add_argument("--train-stop", type=int, default=RunConfig.train_stop)
    parser.add_argument("--rank", type=int, default=RunConfig.rank)
    parser.add_argument(
        "--correction-rank",
        type=int,
        default=RunConfig.correction_rank,
        help="Optional rank for the correction DMD model. Defaults to --rank.",
    )
    parser.add_argument("--alpha", type=float, default=RunConfig.alpha)
    parser.add_argument(
        "--disable-prior-term",
        action="store_true",
        help="Remove the least-squares prior term that penalizes leaving the predicted reduced DMD state.",
    )
    parser.add_argument(
        "--disable-correction-basis",
        action="store_true",
        help="Disable the correction basis and restrict correction to the prediction basis only.",
    )
    parser.add_argument(
        "--correction-damping",
        type=float,
        default=RunConfig.correction_damping,
        help="Step size applied to each joint reduced-space correction update.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=RunConfig.benchmark_repeats,
        help="Number of repeated timing runs per method.",
    )
    return parser


def _build_reduced_artifacts(case: Diffusion1DCase, truth, run_config: RunConfig):
    prediction_model = fit_prediction_dmd(truth, run_config.train_stop, run_config.rank)
    if run_config.use_correction_basis:
        correction_model = fit_correction_dmd(
            case,
            truth,
            prediction_model,
            run_config.train_stop,
            run_config.active_correction_rank,
        )
    else:
        correction_model = build_disabled_correction_dmd(truth.shape[0], run_config.train_stop)
    prepared_artifacts = prepare_reduced_pcdmd_artifacts(
        case,
        truth,
        prediction_model,
        correction_model,
        run_config,
    )
    return prediction_model, correction_model, prepared_artifacts


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    case_config = DiffusionConfig(nx=args.nx, n_steps=args.n_steps)
    run_config = RunConfig(
        train_stop=args.train_stop,
        rank=args.rank,
        correction_rank=args.correction_rank,
        alpha=args.alpha,
        correction_damping=args.correction_damping,
        use_prior_term=not args.disable_prior_term,
        use_correction_basis=not args.disable_correction_basis,
        benchmark_repeats=args.benchmark_repeats,
        output_dir=args.output_dir,
    )
    if run_config.train_stop >= case_config.n_steps:
        raise ValueError("train_stop must be smaller than n_steps.")

    case = Diffusion1DCase(case_config)
    ensure_output_dir(run_config.output_dir)
    x, t, truth = case.simulate()

    prediction_model, correction_model, prepared_artifacts = _build_reduced_artifacts(
        case,
        truth,
        run_config,
    )

    reduced_dmd_prediction, _ = rollout_prediction_states(case, prediction_model, truth[:, 0], case_config.n_steps)
    reduced_dmd_error = relative_l2_error(truth, reduced_dmd_prediction)

    reduced_pcdmd_prediction, reduced_pcdmd_diagnostics = run_reduced_pcdmd_with_diagnostics(
        case,
        truth,
        prediction_model,
        correction_model,
        run_config,
        prepared_artifacts=prepared_artifacts,
    )
    reduced_pcdmd_error = relative_l2_error(truth, reduced_pcdmd_prediction)

    plot_error_curves(
        t,
        reduced_dmd_error,
        {run_config.alpha: reduced_pcdmd_error},
        run_config.train_stop,
        run_config.output_dir / "diffusion_errors.png",
        rank=run_config.rank,
        best_alpha=run_config.alpha,
        dmd_label="Reduced DMD",
        pcdmd_label="Reduced PCDMD",
    )
    plot_snapshots(
        x,
        t,
        truth,
        reduced_dmd_prediction,
        reduced_pcdmd_prediction,
        case.default_snapshot_indices(),
        run_config.output_dir / "diffusion_snapshots.png",
        alpha=run_config.alpha,
        rank=run_config.rank,
        dmd_label="Reduced DMD",
        pcdmd_label="Reduced PCDMD",
    )

    metrics = {
        "case_config": {
            **asdict(case_config),
            "dx": case_config.dx,
            "dt": case_config.dt,
        },
        "run_config": {
            "train_stop": run_config.train_stop,
            "rank": run_config.rank,
            "correction_rank": run_config.active_correction_rank,
            "alpha": run_config.alpha,
            "correction_damping": run_config.correction_damping,
            "use_prior_term": run_config.use_prior_term,
            "use_correction_basis": run_config.use_correction_basis,
            "benchmark_repeats": run_config.benchmark_repeats,
            "output_dir": str(run_config.output_dir),
        },
        "reduced_dmd": summarize_errors(reduced_dmd_error, run_config.train_stop),
        "reduced_pcdmd": summarize_errors(reduced_pcdmd_error, run_config.train_stop),
        "reduced_pcdmd_diagnostics": reduced_pcdmd_diagnostics,
    }
    save_metrics(run_config.output_dir / "metrics.json", metrics)
    save_predictions(
        run_config.output_dir / "predictions.npz",
        x=x,
        t=t,
        truth=truth,
        reduced_dmd=reduced_dmd_prediction,
        reduced_pcdmd=reduced_pcdmd_prediction,
    )

    full_space_run_config = FullSpaceRunConfig(
        train_stop=run_config.train_stop,
        rank=run_config.rank,
        alpha=run_config.alpha,
        output_dir=run_config.output_dir,
    )
    full_space_timing = benchmark_workflow(
        setup_callable=lambda: fit_full_space_dmd(truth, run_config.train_stop, run_config.rank),
        online_callable=lambda a_dmd: run_full_space_pcdmd(
            case,
            truth,
            a_dmd,
            full_space_run_config,
            run_config.alpha,
        ),
        n_online_steps=case_config.n_steps,
        n_repeats=run_config.benchmark_repeats,
    )
    reduced_space_timing = benchmark_workflow(
        setup_callable=lambda: _build_reduced_artifacts(case, truth, run_config),
        online_callable=lambda artifacts: run_reduced_pcdmd(
            case,
            truth,
            artifacts[0],
            artifacts[1],
            run_config,
            prepared_artifacts=artifacts[2],
        ),
        n_online_steps=case_config.n_steps,
        n_repeats=run_config.benchmark_repeats,
    )

    ratios = {
        "offline_setup_speedup": full_space_timing.offline_setup_s / max(
            reduced_space_timing.offline_setup_s, 1e-12
        ),
        "online_run_speedup": full_space_timing.online_run_s / max(
            reduced_space_timing.online_run_s, 1e-12
        ),
        "total_speedup": full_space_timing.total_s / max(reduced_space_timing.total_s, 1e-12),
        "avg_step_speedup": full_space_timing.avg_step_s / max(
            reduced_space_timing.avg_step_s, 1e-12
        ),
    }
    timing_payload = {
        "rank": run_config.rank,
        "correction_rank": run_config.active_correction_rank,
        "alpha": run_config.alpha,
        "use_prior_term": run_config.use_prior_term,
        "use_correction_basis": run_config.use_correction_basis,
        "benchmark_repeats": run_config.benchmark_repeats,
        "case_config": {
            **asdict(case_config),
            "dx": case_config.dx,
            "dt": case_config.dt,
        },
        "full_space": full_space_timing.to_dict(),
        "reduced_space": reduced_space_timing.to_dict(),
        "ratios": ratios,
    }
    save_metrics(run_config.output_dir / "timing_comparison.json", timing_payload)
    plot_timing_comparison(
        {
            "Full-space": full_space_timing,
            "Reduced-space": reduced_space_timing,
        },
        run_config.output_dir / "timing_comparison.png",
        rank=run_config.rank,
        alpha=run_config.alpha,
    )

    print(f"Saved outputs to: {run_config.output_dir.resolve()}")
    print("Reduced DMD metrics:", json.dumps(metrics["reduced_dmd"], ensure_ascii=False))
    print(
        "Reduced PCDMD metrics:",
        json.dumps(metrics["reduced_pcdmd"], ensure_ascii=False),
    )
    print("Timing comparison:", json.dumps(timing_payload["ratios"], ensure_ascii=False))


if __name__ == "__main__":
    main()
