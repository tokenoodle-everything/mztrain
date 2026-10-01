"""mztrain.data.synthetic
========================

Ready-to-use toy datasets, generated lazily **row-by-row** so they never
hold more than one sample in memory at any time.

These are the canary datasets for the "near-zero memory" promise of the
project: ``moons(1_000_000)`` does **not** materialise a million-row
array — each ``(x, y)`` pair is produced on demand from a local RNG and
discarded as soon as the caller is done with it.
"""

from __future__ import annotations

import numpy as np

from .stream import IterableDataset, SyntheticDataset  # noqa: F401


def moons(n_samples: int = 10000, noise: float = 0.1, seed: int = 0) -> IterableDataset:
    """Two interleaving half-circles.

    Generated row-by-row using a single seeded :class:`numpy.random.Generator`.
    The dataset contains exactly ``n_samples`` rows then stops; wrap in
    :class:`InfiniteIterableDataset` to loop forever.

    Memory footprint at any instant: one ``(2,)`` float32 + one ``int``,
    independent of ``n_samples``.
    """

    class _Moons(IterableDataset):
        def __iter__(self):
            rng = np.random.default_rng(seed)
            n_half = n_samples // 2
            # We sample angles in tiny chunks instead of preallocating,
            # so memory stays bounded by the chunk size (256 here).
            chunk = 256
            for start in range(0, n_half, chunk):
                end = min(start + chunk, n_half)
                m = end - start
                # Outer half (class 0): standard half-circle.
                outer = rng.uniform(0.0, np.pi, m)
                outer_x = np.cos(outer)
                outer_y = np.sin(outer)
                outer_x = (outer_x + noise * rng.standard_normal(m)).astype(np.float32)
                outer_y = (outer_y + noise * rng.standard_normal(m)).astype(np.float32)
                for i in range(m):
                    yield np.array([outer_x[i], outer_y[i]], dtype=np.float32), np.int64(0)
            for start in range(0, n_samples - n_half, chunk):
                end = min(start + chunk, n_samples - n_half)
                m = end - start
                # Inner half (class 1): inverted, offset half-circle.
                inner = rng.uniform(0.0, np.pi, m)
                inner_x = 1.0 - np.cos(inner)
                inner_y = 1.0 - np.sin(inner) - 0.5
                inner_x = (inner_x + noise * rng.standard_normal(m)).astype(np.float32)
                inner_y = (inner_y + noise * rng.standard_normal(m)).astype(np.float32)
                for i in range(m):
                    yield np.array([inner_x[i], inner_y[i]], dtype=np.float32), np.int64(1)

    return _Moons()


def circles(n_samples: int = 10000, noise: float = 0.1, seed: int = 0) -> IterableDataset:
    """Concentric circles (non-linearly separable).

    Generated row-by-row from a local seeded :class:`numpy.random.Generator`.
    """

    class _Circles(IterableDataset):
        def __iter__(self):
            rng = np.random.default_rng(seed)
            n_half = n_samples // 2
            chunk = 256
            for start in range(0, n_half, chunk):
                end = min(start + chunk, n_half)
                m = end - start
                r1 = rng.uniform(0.0, 1.0, m)
                a1 = rng.uniform(0.0, 2 * np.pi, m)
                x1 = (r1 * np.cos(a1) + noise * rng.standard_normal(m)).astype(np.float32)
                y1 = (r1 * np.sin(a1) + noise * rng.standard_normal(m)).astype(np.float32)
                for i in range(m):
                    yield np.array([x1[i], y1[i]], dtype=np.float32), np.int64(0)
            for start in range(0, n_samples - n_half, chunk):
                end = min(start + chunk, n_samples - n_half)
                m = end - start
                r2 = rng.uniform(0.7, 1.3, m)
                a2 = rng.uniform(0.0, 2 * np.pi, m)
                x2 = (r2 * np.cos(a2) + noise * rng.standard_normal(m)).astype(np.float32)
                y2 = (r2 * np.sin(a2) + noise * rng.standard_normal(m)).astype(np.float32)
                for i in range(m):
                    yield np.array([x2[i], y2[i]], dtype=np.float32), np.int64(1)

    return _Circles()


def regression_sine(n_samples: int = 10000, noise: float = 0.05, seed: int = 0) -> IterableDataset:
    """``y = sin(2πx) + ε`` for ``x ∈ [-1, 1]``.

    Generated row-by-row.
    """

    class _Sine(IterableDataset):
        def __iter__(self):
            rng = np.random.default_rng(seed)
            chunk = 256
            remaining = n_samples
            while remaining > 0:
                m = min(chunk, remaining)
                xs = rng.uniform(-1.0, 1.0, m).astype(np.float32)
                ys = (np.sin(2 * np.pi * xs) + noise * rng.standard_normal(m)).astype(np.float32)
                for i in range(m):
                    yield xs[i:i + 1].copy(), ys[i:i + 1].copy()
                remaining -= m

    return _Sine()


__all__ = ["moons", "circles", "regression_sine", "SyntheticDataset"]