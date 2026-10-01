"""mztrain.models.minitf
=======================

A toy decoder-only transformer block; small enough to train on a laptop
but rich enough to demonstrate gradient checkpointing paying off.
"""

from __future__ import annotations

import numpy as np

from ..core.ops import Module, matmul, mean, parameter, relu, softmax, sum, tanh
from ..core.tensor import Tensor


def _softmax(a: Tensor, axis: int = -1) -> Tensor:
    return softmax(a, axis=axis)


class Attention(Module):
    """Single-head scaled dot-product self-attention."""

    def __init__(self, dim: int):
        self.q = parameter(dim, dim)
        self.k = parameter(dim, dim)
        self.v = parameter(dim, dim)
        self.scale = 1.0 / np.sqrt(dim)

    def __call__(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        Q = matmul(x, self.q)
        K = matmul(x, self.k)
        V = matmul(x, self.v)
        scores = matmul(Q, K.transpose((0, 2, 1))) * self.scale
        attn = _softmax(scores, axis=-1)
        return matmul(attn, V)


class FeedForward(Module):
    def __init__(self, dim: int, hidden: int):
        self.w1 = parameter(dim, hidden)
        self.w2 = parameter(hidden, dim)

    def __call__(self, x: Tensor) -> Tensor:
        return matmul(relu(matmul(x, self.w1)), self.w2)


class MiniTransformerBlock(Module):
    def __init__(self, dim: int, ff_hidden: int = 64):
        self.attn = Attention(dim)
        self.ff = FeedForward(dim, ff_hidden)

    def __call__(self, x: Tensor) -> Tensor:
        a = self.attn(x)
        x = x + a
        f = self.ff(x)
        x = x + f
        return x


__all__ = ["Attention", "FeedForward", "MiniTransformerBlock"]