"""mztrain.data.synthetic
========================

Ready-to-use toy datasets, generated lazily so they never blow memory.
Each function returns an :class:`mztrain.data.stream.IterableDataset`.
"""

from __future__ import annotations

import numpy as np

from .stream import IterableDataset, SyntheticDataset


def moons(n_samples: int = 10000, noise: float = 0.1, seed: int = 0) -> IterableDataset:
    """Two interleaving half-circles (classic benchmark)."""

    class _Moons(IterableDataset):
        def __iter__(self):
            rng = np.random.default_rng(seed)
            n_half = n_samples // 2
            outer = (rng.uniform(0, np.pi, n_half)[:, None])
            inner = (rng.uniform(0, np.pi, n_half)[:, None])
            outer_xy = np.hstack([np.cos(outer), np.sin(outer)])
            inner_xy = np.hstack([1 - np.cos(inner), 1 - np.sin(inner) - 0.5])
            X = np.vstack([outer_xy, inner_xy]).astype(np.float32)
            X += noise * rng.standard_normal(X.shape).astype(np.float32)
            y = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(np.int64)
            idx = rng.permutation(len(X))
            for i in idx:
                yield X[i], y[i]

    return _Moons()


def circles(n_samples: int = 10000, noise: float = 0.1, seed: int = 0) -> IterableDataset:
    """Concentric circles (non-linearly separable)."""

    class _Circles(IterableDataset):
        def __iter__(self):
            rng = np.random.default_rng(seed)
            n_half = n_samples // 2
            r1 = rng.uniform(0, 1, n_half)
            r2 = rng.uniform(0.7, 1.3, n_samples - n_half)
            a1 = rng.uniform(0, 2 * np.pi, n_half)
            a2 = rng.uniform(0, 2 * np.pi, n_samples - n_half)
            X1 = np.hstack([r1 * np.cos(a1), r1 * np.sin(a1)])
            X2 = np.hstack([r2 * np.cos(a2), r2 * np.sin(a2)])
            X = np.vstack([X1, X2]).astype(np.float32)
            X += noise * rng.standard_normal(X.shape).astype(np.float32)
            y = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(np.int64)
            idx = rng.permutation(len(X))
            for i in idx:
                yield X[i], y[i]

    return _Circles()


def regression_sine(n_samples: int = 10000, noise: float = 0.05, seed: int = 0) -> IterableDataset:
    """y = sin(2πx) + ε for x ∈ [-1, 1]."""

    class _Sine(IterableDataset):
        def __iter__(self):
            rng = np.random.default_rng(seed)
            x = rng.uniform(-1, 1, n_samples).astype(np.float32)
            y = (np.sin(2 * np.pi * x) + noise * rng.standard_normal(n_samples)).astype(np.float32)
            for i in range(n_samples):
                yield x[i:i + 1], y[i:i + 1]

    return _Sine()


__all__ = ["moons", "circles", "regression_sine", "SyntheticDataset"]