"""Minimal PCR-NO model scaffold for the S2 training path."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from models.pcr_projection import PCRProjection, ProjectionConfig


@dataclass(slots=True)
class PCRNOConfig:
    """Configuration container for the PCR-NO scaffold."""

    input_dim: int
    hidden_dim: int
    proj_dim: int
    output_dim: int
    num_heads: int
    eq_wise_heads: bool
    dropout: float
    sequence_length: int
    batch_size: int

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a serializable view for summaries and checkpoints."""

        return asdict(self)


class EquationWiseHead(nn.Module):
    """Equation-wise prediction head placeholder.

    When `eq_wise_heads` is enabled we keep independent light heads so later
    equation-specific constraints can be introduced without rewiring the model.
    """

    def __init__(self, proj_dim: int, output_dim: int, num_heads: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(proj_dim, output_dim) for _ in range(max(num_heads, 1))]
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Average per-head predictions into one output tensor."""

        predictions = [head(encoded) for head in self.heads]
        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0)


class PCRNO(nn.Module):
    """Compact PCR-NO skeleton with projection and optional equation-wise heads."""

    def __init__(self, config: PCRNOConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = PCRProjection(
            ProjectionConfig(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                proj_dim=config.proj_dim,
                dropout=config.dropout,
            )
        )
        self.mixing = nn.Sequential(
            nn.LayerNorm(config.proj_dim),
            nn.GELU(),
            nn.Linear(config.proj_dim, config.proj_dim),
        )
        if config.eq_wise_heads:
            self.head = EquationWiseHead(
                proj_dim=config.proj_dim,
                output_dim=config.output_dim,
                num_heads=config.num_heads,
            )
        else:
            self.head = nn.Linear(config.proj_dim, config.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the scaffold model on `[batch, points, channels]` tensors."""

        encoded = self.projection(inputs)
        mixed = self.mixing(encoded)
        return self.head(mixed)
