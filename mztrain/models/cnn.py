"""mztrain.models.cnn
====================

A small VGG-ish CNN suitable for 28x28 or 32x32 images, useful for the
demo benchmarks.
"""

from __future__ import annotations

import numpy as np

from ..core.ops import (
    Module,
    add,
    conv2d,
    cross_entropy,
    matmul,
    max_pool2d,
    mean,
    parameter,
    relu,
    reshape,
)
from ..core.tensor import Tensor


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, stride: int = 1):
        self.weight = parameter(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = parameter(out_channels)
        self.stride = stride
        self.padding = padding

    def __call__(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel: int = 2, stride: int = 2):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x: Tensor) -> Tensor:
        return max_pool2d(x, self.kernel, self.stride)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        self.weight = parameter(in_features, out_features)
        self.bias = parameter(out_features)

    def __call__(self, x: Tensor) -> Tensor:
        return matmul(x, self.weight) + self.bias


class SmallCNN(Module):
    """Conv(1→16) → ReLU → Pool → Conv(16→32) → ReLU → Pool → Linear."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 image_size: int = 28):
        self.conv1 = Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = MaxPool2d(2, 2)
        flat_dim = 32 * (image_size // 4) * (image_size // 4)
        self.fc1 = Linear(flat_dim, 128)
        self.fc2 = Linear(128, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = reshape(x, (x.shape[0], -1))
        x = relu(self.fc1(x))
        return self.fc2(x)


__all__ = ["Conv2d", "MaxPool2d", "Linear", "SmallCNN"]