"""mztrain.train.trainer
======================

The user-facing training loop.  Composes the streaming dataset, the
model, the optimiser, and the optional dynamic-batch-size calibrator into a
``fit()`` call.

Why one big function?
---------------------
Because there are *so few* knobs.  The whole project promises that you
shouldn't have to think about them.  Pass a model + dataset, optionally
pass an optimiser; ``fit`` does the rest.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.memory import calibrate_batch_size
from ..core.tensor import Tensor
from ..data.stream import BatchIterableDataset, IterableDataset
from ..optim.adam8bit import Adam8bit
from ..optim.sgd import SGD


@dataclass
class TrainerConfig:
    epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: int = 32
    lr: float = 1e-3
    auto_batch_size: bool = False
    optimizer: str = "adam8bit"
    log_every: int = 50
    grad_clip: Optional[float] = 1.0
    seed: int = 0
    verbose: bool = True


@dataclass
class TrainHistory:
    losses: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    steps: int = 0
    epoch_time_s: float = 0.0
    final_batch_size: int = 32


def fit(
    model,
    dataset: IterableDataset,
    *,
    loss_fn: Callable,
    cfg: TrainerConfig = TrainerConfig(),
) -> TrainHistory:
    """Train ``model`` on a streaming ``dataset``.

    >>> import mztrain as mzt
    >>> from mztrain.data import moons
    >>> from mztrain.models import MLP
    >>> model = MLP(in_features=2, hidden=[16, 16], out_features=2)
    >>> history = fit(model, moons(2000), loss_fn=lambda p, y: mzt.cross_entropy(p, y))
    """
    np.random.seed(cfg.seed)
    params = model.parameters()
    if cfg.optimizer == "adam8bit":
        opt = Adam8bit(params, lr=cfg.lr)
    elif cfg.optimizer == "sgd":
        opt = SGD(params, lr=cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")

    bds = BatchIterableDataset(dataset, batch_size=cfg.batch_size, drop_last=False,
                               shuffle=False)
    history = TrainHistory(final_batch_size=cfg.batch_size)

    def _probe(bsz: int) -> None:
        bds_probe = BatchIterableDataset(dataset, batch_size=bsz, drop_last=False,
                                         shuffle=False)
        xb, yb = next(iter(bds_probe))
        xb_t = Tensor(xb, requires_grad=False)
        out = model(xb_t)
        loss = loss_fn(out, yb)
        model.zero_grad()
        loss.backward()

    if cfg.auto_batch_size:
        cfg.batch_size = calibrate_batch_size(
            _probe,
            initial=cfg.batch_size,
            safety_factor=0.6,
        )
        history.final_batch_size = cfg.batch_size
        if cfg.verbose:
            print(f"[mztrain] auto-tuned batch_size = {cfg.batch_size}")
        bds = BatchIterableDataset(dataset, batch_size=cfg.batch_size, drop_last=False,
                                   shuffle=False)

    iter_bds = iter(bds)
    steps = 0
    t0 = _time.perf_counter()
    try:
        for epoch in range(cfg.epochs):
            iter_bds = iter(bds)
            stop = False
            while not stop:
                if cfg.max_steps is not None and steps >= cfg.max_steps:
                    stop = True
                    break
                try:
                    xb, yb = next(iter_bds)
                except StopIteration:
                    break
                if xb.shape[0] < 2:
                    continue
                xb_t = Tensor(xb, requires_grad=False)
                out = model(xb_t)
                loss = loss_fn(out, yb)
                model.zero_grad()
                loss.backward()
                if cfg.grad_clip is not None:
                    _grad_clip(params, cfg.grad_clip)
                opt.step()
                opt.zero_grad()
                history.losses.append(float(loss.data))
                steps += 1
                if cfg.verbose and steps % cfg.log_every == 0:
                    print(
                        f"[mztrain] step {steps:5d} epoch {epoch+1} "
                        f"loss={float(loss.data):.4f}"
                    )
    finally:
        history.steps = steps
        history.epoch_time_s = _time.perf_counter() - t0
    if cfg.verbose:
        print(f"[mztrain] done in {history.epoch_time_s:.2f}s ({steps} steps)")
    return history


def _grad_clip(params, max_norm: float) -> None:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float((p.grad ** 2).sum())
    norm = np.sqrt(sq) + 1e-12
    if norm > max_norm:
        scale = max_norm / norm
        for p in params:
            if p.grad is not None:
                p.grad *= scale


def evaluate(model, dataset: IterableDataset, loss_fn: Callable,
             batch_size: int = 64, max_batches: Optional[int] = None) -> dict:
    """Evaluate ``model`` over a streaming dataset, returning avg loss + acc."""
    if hasattr(model, "eval"):
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    bds = BatchIterableDataset(dataset, batch_size=batch_size, drop_last=False)
    for i, (xb, yb) in enumerate(bds):
        if max_batches is not None and i >= max_batches:
            break
        out = model(Tensor(xb, requires_grad=False))
        loss = loss_fn(out, yb)
        total_loss += float(loss.data) * xb.shape[0]
        pred = np.asarray(out.data).argmax(axis=-1)
        if yb.ndim == 1:
            total_correct += int((pred == yb).sum())
        total_n += xb.shape[0]
    return {"loss": total_loss / max(1, total_n), "accuracy": total_correct / max(1, total_n)}


__all__ = ["fit", "evaluate", "TrainerConfig", "TrainHistory"]