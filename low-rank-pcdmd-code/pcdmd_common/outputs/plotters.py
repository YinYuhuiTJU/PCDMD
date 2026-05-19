from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

FIGURE_BACKGROUND = "#f7f5ef"
AXES_BACKGROUND = "#ffffff"
GRID_MAJOR = "#d7dde5"
GRID_MINOR = "#ebeff4"
SPINE_COLOR = "#5a6573"
TRUTH_COLOR = "#1f2933"
DMD_COLOR = "#c04b3b"
BEST_PCDMD_COLOR = "#147a73"
OTHER_PCDMD_COLORS = ["#6f8fb3", "#7aa6d8", "#90b9e6", "#adcdf0"]
TRAINING_SHADE = "#eef2f6"
BOUNDARY_COLOR = "#6c757d"


def _rc_params() -> dict[str, object]:
    return {
        "figure.facecolor": FIGURE_BACKGROUND,
        "axes.facecolor": AXES_BACKGROUND,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.8,
        "savefig.facecolor": FIGURE_BACKGROUND,
        "savefig.bbox": "tight",
    }


def _style_axes(ax) -> None:
    ax.grid(True, which="major", color=GRID_MAJOR, alpha=0.8)
    ax.grid(True, which="minor", color=GRID_MINOR, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)


def _title_block(fig, title: str, subtitle: str) -> None:
    fig.suptitle(
        title,
        x=0.06,
        y=0.98,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#18212b",
    )
    fig.text(
        0.06,
        0.935,
        subtitle,
        ha="left",
        va="top",
        fontsize=11,
        color="#4f5d6d",
    )


def plot_error_curves(
    t: np.ndarray,
    dmd_error: np.ndarray,
    pcdmd_errors: dict[float, np.ndarray],
    train_stop: int,
    output_path: Path,
    *,
    rank: int,
    best_alpha: float,
    dmd_label: str = "DMD",
    pcdmd_label: str = "PCDMD",
) -> None:
    with plt.rc_context(_rc_params()):
        fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=180)
        _title_block(
            fig,
            "1D Diffusion Error Curves",
            f"Relative $L^2$ error with DMD rank $r={rank}$; best PCDMD parameter is $\\alpha={best_alpha:g}$",
        )

        t_cutoff = float(t[train_stop])
        ax.axvspan(float(t[0]), t_cutoff, color=TRAINING_SHADE, alpha=0.9, zorder=0)
        ax.axvline(
            t_cutoff,
            color=BOUNDARY_COLOR,
            linestyle="--",
            linewidth=1.6,
            zorder=2,
        )
        ax.text(
            float(t[0]) + 0.02 * (float(t[-1]) - float(t[0])),
            0.97,
            "Training window",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            color="#5f6c7b",
            va="top",
        )
        ax.text(
            t_cutoff + 0.02 * (float(t[-1]) - float(t[0])),
            0.97,
            "Forecast window",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            color="#5f6c7b",
            va="top",
        )

        ax.plot(
            t,
            dmd_error,
            color=DMD_COLOR,
            linewidth=2.4,
            label=dmd_label,
            zorder=3,
        )

        sorted_alphas = sorted(pcdmd_errors)
        fallback_iter = iter(OTHER_PCDMD_COLORS)
        for alpha in sorted_alphas:
            is_best = np.isclose(alpha, best_alpha)
            color = BEST_PCDMD_COLOR if is_best else next(fallback_iter, OTHER_PCDMD_COLORS[-1])
            ax.plot(
                t,
                pcdmd_errors[alpha],
                color=color,
                linewidth=2.8 if is_best else 1.8,
                alpha=1.0 if is_best else 0.9,
                linestyle="-" if is_best else "-.",
                label=f"{pcdmd_label} ($\\alpha={alpha:g}$)",
                zorder=4 if is_best else 3,
            )

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Relative $L^2$ Error")
        ax.set_xlim(float(t[0]), float(t[-1]))
        positive_errors = np.concatenate(
            [dmd_error[dmd_error > 0], *[values[values > 0] for values in pcdmd_errors.values()]]
        )
        y_min = max(float(np.min(positive_errors)) * 0.7, 1e-12)
        y_max = float(np.max(positive_errors)) * 1.35
        ax.set_ylim(y_min, y_max)
        _style_axes(ax)

        legend = ax.legend(
            loc="upper left",
            ncol=2,
            frameon=True,
            fancybox=True,
            framealpha=0.97,
            facecolor="white",
            edgecolor="#d7dde5",
            borderpad=0.8,
            handlelength=2.8,
        )
        for line in legend.get_lines():
            line.set_linewidth(2.6)

        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def plot_snapshots(
    x: np.ndarray,
    t: np.ndarray,
    truth: np.ndarray,
    dmd_prediction: np.ndarray,
    pcdmd_prediction: np.ndarray,
    time_indices: tuple[int, int],
    output_path: Path,
    *,
    alpha: float,
    rank: int,
    dmd_label: str = "DMD",
    pcdmd_label: str = "Best PCDMD",
) -> None:
    with plt.rc_context(_rc_params()):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=180, sharey=True)
        _title_block(
            fig,
            "1D Diffusion Snapshots",
            f"Ground truth vs DMD vs best PCDMD with $r={rank}$ and $\\alpha={alpha:g}$",
        )

        selected_truth = truth[:, list(time_indices)]
        selected_dmd = dmd_prediction[:, list(time_indices)]
        selected_pcdmd = pcdmd_prediction[:, list(time_indices)]
        y_min = min(
            float(np.min(selected_truth)),
            float(np.min(selected_dmd)),
            float(np.min(selected_pcdmd)),
        )
        y_max = max(
            float(np.max(selected_truth)),
            float(np.max(selected_dmd)),
            float(np.max(selected_pcdmd)),
        )
        margin = 0.08 * max(y_max - y_min, 1e-6)

        for ax, idx in zip(axes, time_indices):
            ax.plot(
                x,
                truth[:, idx],
                color=TRUTH_COLOR,
                linewidth=2.6,
                label="Ground Truth",
            )
            ax.plot(
                x,
                dmd_prediction[:, idx],
                color=DMD_COLOR,
                linewidth=2.0,
                linestyle="--",
                label=dmd_label,
            )
            ax.plot(
                x,
                pcdmd_prediction[:, idx],
                color=BEST_PCDMD_COLOR,
                linewidth=2.4,
                label=pcdmd_label,
            )
            ax.set_title(f"$t={t[idx]:.3f}$", loc="left", pad=10, color="#243447")
            ax.set_xlabel("$x$")
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(y_min - margin, y_max + margin)
            _style_axes(ax)

        axes[0].set_ylabel("$u(x,t)$")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.885),
            ncol=3,
            frameon=True,
            fancybox=True,
            framealpha=0.97,
            facecolor="white",
            edgecolor="#d7dde5",
            borderpad=0.8,
        )

        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def plot_timing_comparison(
    timing_by_method: dict[str, object],
    output_path: Path,
    *,
    rank: int,
    alpha: float,
) -> None:
    with plt.rc_context(_rc_params()):
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.6), dpi=180, sharey=False)
        _title_block(
            fig,
            "1D Diffusion Timing Comparison",
            f"Average runtime over repeated runs with $r={rank}$ and $\\alpha={alpha:g}$",
        )

        metric_order = [
            ("offline_setup_s", "Offline setup"),
            ("online_run_s", "Online run"),
            ("total_s", "Total"),
        ]
        method_names = list(timing_by_method.keys())
        colors = [DMD_COLOR, BEST_PCDMD_COLOR, "#587291", "#8c6d46"]

        for axis, (metric_name, title) in zip(axes, metric_order):
            values = [getattr(timing_by_method[name], metric_name) for name in method_names]
            axis.bar(
                method_names,
                values,
                color=colors[: len(method_names)],
                edgecolor=SPINE_COLOR,
                linewidth=1.0,
            )
            axis.set_title(title, loc="left", pad=10, color="#243447")
            axis.set_ylabel("Seconds")
            axis.tick_params(axis="x", rotation=12)
            _style_axes(axis)

        fig.savefig(output_path, dpi=220)
        plt.close(fig)
