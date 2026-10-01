"""mztrain.core
=============

Tiny autograd engine.  Single source of truth for tensors, ops, and the
gradient-checkpointing decorator.
"""

from .autograd import CheckpointedBlock, checkpoint, checkpoint_sequential
from .memory import (
    available_memory_bytes,
    calibrate_batch_size,
    process_rss_bytes,
)
from .ops import (
    Module,
    add,
    conv2d,
    cross_entropy,
    exp,
    log,
    log_softmax,
    matmul,
    max_pool2d,
    mean,
    mse_loss,
    mul,
    neg,
    nll_loss,
    parameter,
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
from .tensor import Tensor

__all__ = [
    "Tensor",
    "Module",
    "add",
    "sub",
    "mul",
    "neg",
    "true_div",
    "matmul",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "exp",
    "log",
    "reshape",
    "transpose",
    "sum",
    "mean",
    "conv2d",
    "max_pool2d",
    "cross_entropy",
    "mse_loss",
    "nll_loss",
    "parameter",
    "zeros_parameter",
    "CheckpointedBlock",
    "checkpoint",
    "checkpoint_sequential",
    "available_memory_bytes",
    "calibrate_batch_size",
    "process_rss_bytes",
]