# 1D Diffusion Submission Package

This folder is a clean submission package for the `1D diffusion` example only.

It contains:

- `full_space_pcdmd/`: full-space DMD and PCDMD baseline
- `low_rank_pcdmd/`: low-rank DMD and low-rank PCDMD method
- `pcdmd_common/`: shared diffusion case, plotting, metrics, and timing utilities
- `run_full_space.py`: run the full-space baseline
- `run_low_rank.py`: run the low-rank method and timing comparison
- `run_all.py`: run both workflows in sequence

It does not contain:

- other PDE benchmark cases
- advanced reactor or flow cases
- paper writing files
- tuning, ablation, or sensitivity scripts
- historical output folders

## Environment

Recommended Python version: `3.11+`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the full-space baseline:

```bash
python run_full_space.py
```

Run the low-rank method:

```bash
python run_low_rank.py
```

Run both:

```bash
python run_all.py
```

## Output

The scripts create:

- `outputs/full_space_diffusion_1d/`
- `outputs/low_rank_diffusion_1d/`

Typical output files include:

- `metrics.json`
- `predictions.npz`
- `diffusion_errors.png`
- `diffusion_snapshots.png`
- `timing_comparison.json`
- `timing_comparison.png`
