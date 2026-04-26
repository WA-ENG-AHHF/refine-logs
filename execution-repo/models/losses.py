"""Loss helpers for the PCR-NO scaffold."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_loss_breakdown(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Build a readable multi-term loss dictionary.

    The PDE, boundary, and conservation terms are placeholders derived from the
    main prediction error so the training scaffold remains executable before the
    physics-specific residual operators are wired in.
    """

    normalized_weights = {
        "l2": 1.0,
        "pde": 0.0,
        "bc": 0.0,
        "conservation": 0.0,
    }
    normalized_weights.update(weights or {})

    l2 = F.mse_loss(predictions, targets)
    residual = predictions - targets
    pde = residual.abs().mean()
    bc = residual[:, :1].abs().mean()
    conservation = residual.mean(dim=-1).abs().mean()
    total = (
        normalized_weights["l2"] * l2
        + normalized_weights["pde"] * pde
        + normalized_weights["bc"] * bc
        + normalized_weights["conservation"] * conservation
    )
    return {
        "total": total,
        "l2": l2,
        "pde": pde,
        "bc": bc,
        "conservation": conservation,
    }
