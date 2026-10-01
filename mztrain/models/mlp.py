"""mztrain.models.mlp
====================

Standard multi-layer perceptron with ReLU activations.  Designed for both
classification and regression heads.
"""

from __future__ import annotations

from ..core.ops import Module, matmul, mean, mse_loss, parameter, relu, sub
from ..core.tensor import Tensor


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        self.weight = parameter(in_features, out_features)
        self.bias = parameter(out_features)

    def __call__(self, x: Tensor) -> Tensor:
        return matmul(x, self.weight) + self.bias


class MLP(Module):
    """Multi-layer perceptron.

    >>> model = MLP(in_features=8, hidden=[32, 16], out_features=3)
    >>> out = model(Tensor(np.zeros((4, 8), dtype=np.float32), requires_grad=False))
    """

    def __init__(self, in_features: int, hidden: list[int], out_features: int,
                 activation=relu):
        sizes = [in_features] + list(hidden) + [out_features]
        self.layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.activation = activation

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def classify(self, x: Tensor) -> Tensor:
        return self(x).argmax(-1) if hasattr(self(x), "argmax") else None


class MLPRegressor(MLP):
    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        return mse_loss(pred, target)


__all__ = ["Linear", "MLP", "MLPRegressor"]