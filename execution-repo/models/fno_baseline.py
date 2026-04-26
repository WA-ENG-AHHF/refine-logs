from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class FNOBaselineStub(nn.Module):
        """Very small placeholder for the future baseline model."""

        def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 2, depth: int = 2) -> None:
            super().__init__()

            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
            for _ in range(max(depth - 1, 0)):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.network(inputs)

        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

else:

    class FNOBaselineStub:
        """Torch-free fallback so the scaffold can run before dependencies are installed."""

        def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 2, depth: int = 2) -> None:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.depth = depth

        def state_dict(self) -> dict[str, Any]:
            return {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "depth": self.depth,
                "backend": "torch-unavailable",
            }

        def parameter_count(self) -> int:
            hidden_layers = max(self.depth - 1, 0)
            input_block = self.input_dim * self.hidden_dim + self.hidden_dim
            hidden_block = hidden_layers * (self.hidden_dim * self.hidden_dim + self.hidden_dim)
            output_block = self.hidden_dim * self.output_dim + self.output_dim
            return input_block + hidden_block + output_block
