"""Quickstart: fit a tiny MLP to a noisy sine wave.

Run with:
    python examples/quickstart_regression.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

import mztrain as mzt
from mztrain.data import regression_sine
from mztrain.models import MLP
from mztrain.train import TrainerConfig, fit


def main():
    np.random.seed(0)
    model = MLP(in_features=1, hidden=[32, 32], out_features=1)

    history = fit(
        model,
        regression_sine(5000),
        loss_fn=lambda pred, target: mzt.mse_loss(pred, mzt.Tensor(target)),
        cfg=TrainerConfig(
            epochs=2,
            batch_size=32,
            optimizer="adam8bit",
            lr=1e-3,
            verbose=True,
            max_steps=80,
        ),
    )

    # Spot-check predictions.
    xs = np.linspace(-1, 1, 10, dtype=np.float32).reshape(-1, 1)
    ys = model(mzt.Tensor(xs)).data
    print("\nxs:", xs.flatten())
    print("ys:", ys.flatten())


if __name__ == "__main__":
    main()