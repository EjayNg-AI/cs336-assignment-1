from __future__ import annotations

import torch

from cs336_basics.nn_linear_embedding_rope_rmsnorm import Linear


def swiglu_d_ff(d_model: int) -> int:
    raw_d_ff = 8.0 * d_model / 3.0
    return max(64, int((raw_d_ff + 32.0) // 64.0) * 64)


def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    z = torch.exp(-torch.abs(x))
    return torch.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * stable_sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else swiglu_d_ff(d_model)
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
