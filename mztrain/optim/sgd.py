"""mztrain.optim.sgd
===================

Stochastic gradient descent with optional momentum and weight decay.
"""

from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        self.params = list(params)
        self.lr = lr
        self.mu = momentum
        self.wd = weight_decay
        self.nesterov = nesterov
        self.state = {id(p): np.zeros_like(p.data) for p in self.params}

    def step(self) -> None:
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            if self.wd:
                g = g + self.wd * p.data
            buf = self.state[id(p)]
            if self.mu != 0.0:
                buf *= self.mu
                buf += g
                if self.nesterov:
                    g = g + self.mu * buf
                else:
                    g = buf
            p.data -= self.lr * g

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)


__all__ = ["SGD"]