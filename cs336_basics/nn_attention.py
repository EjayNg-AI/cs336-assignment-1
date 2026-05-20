from __future__ import annotations

import math

import torch

from cs336_basics.nn_linear_embedding_rope_rmsnorm import Linear, RotaryPositionalEmbedding


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)
    return attention_weights @ v


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            if max_seq_len is None or theta is None:
                raise ValueError("max_seq_len and theta are required when use_rope=True")
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
        else:
            self.rope = None

    def _project_to_heads(self, x: torch.Tensor, projection: Linear) -> torch.Tensor:
        projected = projection(x)
        projected = projected.reshape(*projected.shape[:-1], self.num_heads, self.head_dim)
        return projected.transpose(-3, -2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        sequence_length = x.shape[-2]

        q = self._project_to_heads(x, self.q_proj)
        k = self._project_to_heads(x, self.k_proj)
        v = self._project_to_heads(x, self.v_proj)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), device=x.device, dtype=torch.bool)
        )
        attention_output = scaled_dot_product_attention(q, k, v, causal_mask)
        attention_output = attention_output.transpose(-3, -2).reshape(*x.shape[:-1], self.d_model)
        return self.output_proj(attention_output)
