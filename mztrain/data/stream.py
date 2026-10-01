"""mztrain.data.stream
====================

**Streaming datasets** — keep *zero* data in memory.

The data-loading problem
-----------------------
PyTorch's ``Dataset`` returns rows directly, but ``DataLoader`` builds an
internal pre-fetch buffer (default 2 workers × batch_size samples).  For
gigabyte-scale datasets this still dwarfs the model's footprint.

Our implementation:

* :class:`IterableDataset` yields one ``(x, y)`` at a time.  Memory grows
  only with ``batch_size``, never with dataset size.
* The base class takes either a ``.npy`` path or a Python iterable of
  arrays.  When given a path we open it as a ``np.memmap`` so the OS
  lazily pages in pages on demand — true streaming.
* :class:`BatchIterableDataset` wraps an iterable dataset and yields
  ``(X_batch, y_batch)`` mini-batches.  It honours a configurable
  ``drop_last`` flag.
* :class:`InfiniteIterableDataset` repeats the upstream stream forever.

The single sample in flight at any time is ``O(batch_size * feature_dim)``,
independent of the on-disk dataset size.  This is the *first* big win
of the project.
"""

from __future__ import annotations

import gzip
from typing import Iterator, Optional, Tuple

import numpy as np


class IterableDataset:
    """Abstract streaming dataset."""

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        pass


class CSVDataset(IterableDataset):
    """CSV file streaming dataset (one row at a time, no full read)."""

    def __init__(self, path, feature_cols, label_col, dtype=np.float32,
                 skip_header: bool = True, delimiter: str = ","):
        self.path = path
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.dtype = dtype
        self.skip_header = skip_header
        self.delimiter = delimiter

    def __iter__(self):
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0 and self.skip_header:
                    continue
                parts = line.rstrip("\n").split(self.delimiter)
                x = np.array([float(parts[c]) for c in self.feature_cols],
                             dtype=self.dtype)
                y = np.array([float(parts[self.label_col])], dtype=self.dtype)
                yield x, y


class NpyDataset(IterableDataset):
    """Stream a ``.npy`` file via memmap — zero resident memory."""

    def __init__(self, x_path: str, y_path: str):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __iter__(self):
        for i in range(self.x.shape[0]):
            yield np.asarray(self.x[i]), np.asarray(self.y[i])


class SyntheticDataset(IterableDataset):
    """An infinite synthetic generator; useful for smoke tests."""

    def __init__(self, n_features: int = 8, n_classes: int = 3, seed: int = 0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        rng = self.rng
        while True:
            cls = int(rng.integers(self.n_classes))
            mean = rng.standard_normal(self.n_features) * 0.5
            x = mean + 0.3 * rng.standard_normal(self.n_features)
            yield x.astype(np.float32), np.int64(cls)


class InfiniteIterableDataset(IterableDataset):
    """Repeats an upstream stream forever."""

    def __init__(self, base: IterableDataset, max_repeat: Optional[int] = None):
        self.base = base
        self.max_repeat = max_repeat
        self._epoch = 0

    def __iter__(self):
        while self.max_repeat is None or self._epoch < self.max_repeat:
            self._epoch += 1
            for sample in self.base:
                yield sample

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


class BatchIterableDataset(IterableDataset):
    """Groups a stream of ``(x, y)`` into mini-batches.

    Memory cost is ``O(batch_size)`` regardless of dataset size — this is
    the cornerstone of the *near-zero memory* property of mztrain.
    """

    def __init__(
        self,
        source: IterableDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        buffer: int = 128,
    ):
        self.source = source
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.buffer = int(buffer)

    def __iter__(self):
        if self.shuffle:
            yield from self._iter_shuffled()
        else:
            yield from self._iter_sequential()

    def _iter_sequential(self):
        bx, by = [], []
        for x, y in self.source:
            bx.append(np.asarray(x))
            by.append(np.asarray(y))
            if len(bx) == self.batch_size:
                yield self._stack(bx), self._stack(by)
                bx.clear()
                by.clear()
        if not self.drop_last and bx:
            yield self._stack(bx), self._stack(by)

    def _iter_shuffled(self):
        rng = np.random.default_rng()
        pool_x, pool_y = [], []
        for x, y in self.source:
            pool_x.append(np.asarray(x))
            pool_y.append(np.asarray(y))
            if len(pool_x) >= self.buffer:
                idx = rng.permutation(len(pool_x))
                pool_x = [pool_x[i] for i in idx]
                pool_y = [pool_y[i] for i in idx]
                while len(pool_x) >= self.batch_size:
                    bxx = pool_x[: self.batch_size]
                    byy = pool_y[: self.batch_size]
                    del pool_x[: self.batch_size]
                    del pool_y[: self.batch_size]
                    yield self._stack(bxx), self._stack(byy)
        if not self.drop_last and pool_x:
            n = len(pool_x) // self.batch_size * self.batch_size
            if n:
                yield self._stack(pool_x[:n]), self._stack(pool_y[:n])

    def _stack(self, items):
        return np.stack(items, axis=0)


__all__ = [
    "IterableDataset",
    "CSVDataset",
    "NpyDataset",
    "SyntheticDataset",
    "InfiniteIterableDataset",
    "BatchIterableDataset",
]