from __future__ import annotations

from low_rank_pcdmd.models.joint_pcdmd import (
    PreparedReducedPCDMDArtifacts,
    ReducedPCDMDDiagnostics,
    estimate_joint_filter_scales,
    prepare_joint_reduced_pcdmd_artifacts,
    run_joint_reduced_pcdmd,
    run_joint_reduced_pcdmd_with_diagnostics,
)

prepare_reduced_pcdmd_artifacts = prepare_joint_reduced_pcdmd_artifacts
run_reduced_pcdmd = run_joint_reduced_pcdmd
run_reduced_pcdmd_with_diagnostics = run_joint_reduced_pcdmd_with_diagnostics

__all__ = [
    "PreparedReducedPCDMDArtifacts",
    "ReducedPCDMDDiagnostics",
    "estimate_joint_filter_scales",
    "prepare_reduced_pcdmd_artifacts",
    "run_reduced_pcdmd",
    "run_reduced_pcdmd_with_diagnostics",
]
