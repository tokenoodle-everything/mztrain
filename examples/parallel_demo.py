"""Multi-process CPU parallelism demo.

Spawns ``world_size`` worker processes (one per CPU core by default) that
each compute a micro-batch gradient; the main process sums them, takes an
optimiser step, and broadcasts updated parameters back.

Run with:
    python examples/parallel_demo.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

import mztrain as mzt
from mztrain.data import moons
from mztrain.data.stream import BatchIterableDataset
from mztrain.models import MLP
from mztrain.train import ParallelConfig, parallel_train


def model_factory() -> mzt.MLP:
    return MLP(in_features=2, hidden=[32, 32], out_features=2)


def loss_fn(pred, target):
    return mzt.cross_entropy(pred, target)


def main():
    np.random.seed(0)
    bds = BatchIterableDataset(moons(5000), batch_size=32, drop_last=True)
    losses = parallel_train(
        model_factory=model_factory,
        loss_fn=loss_fn,
        batch_iter=iter(bds),
        steps=30,
        cfg=ParallelConfig(world_size=2, verbose=True),
    )
    print(f"avg loss over last 5 global steps: {np.mean(losses[-5:]):.4f}")


if __name__ == "__main__":
    main()