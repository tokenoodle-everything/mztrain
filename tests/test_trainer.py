"""Integration test: train an MLP on moons for a few steps."""

import numpy as np

import mztrain as mzt
from mztrain.data import moons
from mztrain.models import MLP
from mztrain.train import TrainerConfig, evaluate, fit


def test_fit_decreases_loss():
    np.random.seed(0)
    model = MLP(in_features=2, hidden=[32, 32], out_features=2)
    from mztrain.data import InfiniteIterableDataset
    history = fit(
        model,
        InfiniteIterableDataset(moons(4000), max_repeat=10),
        loss_fn=lambda p, y: mzt.cross_entropy(p, y),
        cfg=TrainerConfig(epochs=1, batch_size=64, optimizer="sgd",
                          lr=1e-2, max_steps=80, verbose=False),
    )
    assert history.steps == 80
    # Loss should trend down.
    assert history.losses[-1] < history.losses[0], (
        f"loss did not decrease: {history.losses[0]:.3f} -> {history.losses[-1]:.3f}"
    )


def test_evaluate_runs():
    model = MLP(in_features=2, hidden=[16], out_features=2)
    metrics = evaluate(
        model,
        moons(200),
        loss_fn=lambda p, y: mzt.cross_entropy(p, y),
        batch_size=32,
        max_batches=5,
    )
    assert "loss" in metrics and "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0