"""Quickstart: classify the classic two-moons dataset with memzero-train.

Run with:
    python examples/quickstart_classification.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

import mztrain as mzt
from mztrain.data import moons
from mztrain.models import MLP
from mztrain.train import TrainerConfig, evaluate, fit


def main():
    np.random.seed(0)
    model = MLP(in_features=2, hidden=[32, 32], out_features=2)

    history = fit(
        model,
        moons(8000),
        loss_fn=lambda pred, target: mzt.cross_entropy(pred, target),
        cfg=TrainerConfig(
            epochs=2,
            batch_size=64,
            optimizer="adam8bit",
            lr=1e-3,
            auto_batch_size=False,
            verbose=True,
        ),
    )

    print("\n--- evaluation on a held-out moons stream ---")
    metrics = evaluate(
        model,
        moons(2000),
        loss_fn=lambda pred, target: mzt.cross_entropy(pred, target),
        batch_size=128,
        max_batches=15,
    )
    print("metrics:", metrics)


if __name__ == "__main__":
    main()