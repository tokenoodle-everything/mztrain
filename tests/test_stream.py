"""Tests for streaming datasets."""

import gc

import numpy as np

import mztrain as mzt
from mztrain.data import SyntheticDataset, moons
from mztrain.data.stream import BatchIterableDataset


def test_moons_count():
    ds = moons(50)
    items = list(ds)
    assert len(items) == 50
    x, y = items[0]
    assert x.shape == (2,)
    assert y.dtype in (np.int64, np.int32)


def test_batched_stream_yields_batches():
    ds = moons(100)
    bds = BatchIterableDataset(ds, batch_size=10, drop_last=True)
    batches = list(bds)
    assert len(batches) == 10
    xb, yb = batches[0]
    assert xb.shape == (10, 2)
    assert yb.shape == (10,)


def test_synthetic_infinite():
    ds = SyntheticDataset(n_features=4, n_classes=2)
    it = iter(ds)
    for _ in range(8):
        x, y = next(it)
        assert x.shape == (4,)
        assert y in (0, 1)


def test_construction_is_lazy():
    """Constructing a 1M-row moons dataset should not allocate O(N) memory.

    We measure RSS **before** any iteration; if the dataset were eager
    (preallocating all rows) this would balloon to ~10 MB+.
    """
    gc.collect()
    rss_before = mzt.process_rss_bytes()
    ds = moons(1_000_000)  # 1 million rows requested
    gc.collect()
    rss_after_construct = mzt.process_rss_bytes()
    # Allow some overhead for Python object, RNG state, etc. — but it
    # MUST NOT scale with n_samples. 256 KB is generous.
    growth = rss_after_construct - rss_before
    assert growth < 256 * 1024, (
        f"dataset construction allocated {growth} bytes — "
        f"streaming datasets must not preallocate O(n_samples) memory"
    )


def test_memory_constant_with_stream():
    """A streaming dataset should not retain samples in memory after batch yield."""
    gc.collect()
    rss_before = mzt.process_rss_bytes()
    ds = moons(1_000_000)
    bds = BatchIterableDataset(ds, batch_size=32, drop_last=True)
    it = iter(bds)
    for _ in range(50):  # 50 batches -> 1600 rows but dataset has 1M rows
        xb, yb = next(it)
    # Drop references so the yielded batches can be collected.
    del xb, yb, it, bds, ds
    gc.collect()
    rss_after = mzt.process_rss_bytes()
    # Allow OS allocator jitter on Windows / macOS (Python keeps freed
    # pages in its heap). 10 MB is a *very* loose cap; for a 1M-row
    # dataset preallocating in memory this would be ~80 MB+.
    growth = rss_after - rss_before
    assert growth < 10 * 1024 * 1024, (
        f"RSS grew too much after streaming: {rss_before} -> {rss_after} "
        f"(+{growth} bytes). Dataset is likely preallocating in memory."
    )