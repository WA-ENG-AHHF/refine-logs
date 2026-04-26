"""Projection primitives used by the PCR-NO scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ProjectionConfig:
    """Configuration for the compact PCR projection block."""

    input_dim: int
    hidden_dim: int
    proj_dim: int
    dropout: float = 0.0


class PCRProjection(nn.Module):
    """Project raw features into a compact residual-aware representation.

    The real projective-constrained residual layer can later replace this block
    without changing the training script contract.
    """

    def __init__(self, config: ProjectionConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.proj_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map `[..., input_dim]` features into `[..., proj_dim]` embeddings."""

        return self.net(inputs)
