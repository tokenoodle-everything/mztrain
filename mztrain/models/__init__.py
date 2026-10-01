"""mztrain.models
===============

Ready-to-use neural network building blocks.
"""

from .cnn import Conv2d, Linear, MaxPool2d, SmallCNN
from .minitf import Attention, FeedForward, MiniTransformerBlock
from .mlp import Linear as MLPLinear
from .mlp import MLP, MLPRegressor

__all__ = [
    "MLP",
    "MLPRegressor",
    "SmallCNN",
    "Conv2d",
    "MaxPool2d",
    "Linear",
    "MLPLinear",
    "Attention",
    "FeedForward",
    "MiniTransformerBlock",
]