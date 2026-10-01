"""mztrain.data
=============

Streaming datasets and toy problem generators.
"""

from .stream import (
    BatchIterableDataset,
    CSVDataset,
    InfiniteIterableDataset,
    IterableDataset,
    NpyDataset,
    SyntheticDataset,
)
from .synthetic import circles, moons, regression_sine

__all__ = [
    "IterableDataset",
    "CSVDataset",
    "NpyDataset",
    "SyntheticDataset",
    "InfiniteIterableDataset",
    "BatchIterableDataset",
    "moons",
    "circles",
    "regression_sine",
]