"""mztrain.train
===============

High-level training primitives: ``fit()``, ``evaluate()``,
``parallel_train()``.
"""

from .parallel import ParallelConfig, parallel_train
from .trainer import TrainHistory, TrainerConfig, evaluate, fit

__all__ = [
    "fit",
    "evaluate",
    "TrainerConfig",
    "TrainHistory",
    "ParallelConfig",
    "parallel_train",
]