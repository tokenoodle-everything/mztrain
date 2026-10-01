"""mztrain — Zero-Memory Multi-CPU Local AI Training
====================================================

A tiny, NumPy-only deep-learning framework that trains small models on
ordinary laptops without GPU and without high memory overhead.

Top-level convenience:

>>> import mztrain as mzt
>>> from mztrain.data import moons
>>> from mztrain.models import MLP
>>> from mztrain.train import fit, TrainerConfig
>>>
>>> model = MLP(in_features=2, hidden=[32, 32], out_features=2)
>>> history = fit(
...     model,
...     moons(4000),
...     loss_fn=lambda pred, y: mzt.cross_entropy(pred, y),
...     cfg=TrainerConfig(epochs=2, batch_size=32, lr=1e-3),
... )
"""

from .core import (  # re-export
    CheckpointedBlock,
    Module,
    Tensor,
    add,
    available_memory_bytes,
    calibrate_batch_size,
    checkpoint,
    conv2d,
    cross_entropy,
    matmul,
    max_pool2d,
    mean,
    mse_loss,
    mul,
    neg,
    parameter,
    process_rss_bytes,
    relu,
    reshape,
    sigmoid,
    softmax,
    sub,
    sum,
    tanh,
    transpose,
    true_div,
    zeros_parameter,
)
from .data import (
    BatchIterableDataset,
    CSVDataset,
    InfiniteIterableDataset,
    IterableDataset,
    NpyDataset,
    SyntheticDataset,
    circles,
    moons,
    regression_sine,
)
from .models import (
    Attention,
    Conv2d,
    FeedForward,
    Linear,
    MaxPool2d,
    MiniTransformerBlock,
    MLP,
    MLPRegressor,
    SmallCNN,
)
from .optim import Adam8bit, SGD
from .train import (
    ParallelConfig,
    TrainHistory,
    TrainerConfig,
    evaluate,
    fit,
    parallel_train,
)
from .utils import cpu_count, platform_name

__version__ = "0.1.0"

__all__ = [
    # Core
    "Tensor", "Module", "add", "sub", "mul", "neg", "true_div",
    "matmul", "relu", "sigmoid", "tanh", "softmax", "reshape",
    "transpose", "sum", "mean", "conv2d", "max_pool2d",
    "cross_entropy", "mse_loss", "parameter", "zeros_parameter",
    "CheckpointedBlock", "checkpoint",
    "available_memory_bytes", "calibrate_batch_size", "process_rss_bytes",
    # Data
    "IterableDataset", "CSVDataset", "NpyDataset", "SyntheticDataset",
    "InfiniteIterableDataset", "BatchIterableDataset",
    "moons", "circles", "regression_sine",
    # Models
    "MLP", "MLPRegressor", "SmallCNN", "Conv2d", "MaxPool2d", "Linear",
    "Attention", "FeedForward", "MiniTransformerBlock",
    # Optim
    "SGD", "Adam8bit",
    # Train
    "fit", "evaluate", "TrainerConfig", "TrainHistory",
    "ParallelConfig", "parallel_train",
    # Utils
    "cpu_count", "platform_name",
]