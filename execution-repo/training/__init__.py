"""Shared training helpers for formalizing execution-side pipelines."""

from .supervised import (
    TORCH_AVAILABLE,
    compute_direct_metrics_numpy,
    compute_residual_metrics_numpy,
    fit_torch_direct_supervised,
    fit_torch_residual_supervised,
    set_global_seed,
)

__all__ = [
    "TORCH_AVAILABLE",
    "compute_direct_metrics_numpy",
    "compute_residual_metrics_numpy",
    "fit_torch_direct_supervised",
    "fit_torch_residual_supervised",
    "set_global_seed",
]
