"""Tests for streaming datasets."""

import numpy as np

import mztrain as mzt
from mztrain.data import moons, circles, SyntheticDataset
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


def test_memory_constant_with_stream():
    """A streaming dataset should not retain samples in memory after batch yield."""
    import gc

    gc.collect()
    rss_before = mzt.process_rss_bytes()
    ds = moons(1_000_000)
    bds = BatchIterableDataset(ds, batch_size=32, drop_last=True)
    it = iter(bds)
    for _ in range(50):  # 50 batches -> 1600 rows but dataset has 1M rows
        xb, yb = next(it)
    gc.collect()
    rss_after = mzt.process_rss_bytes()
    # Allow some growth but not linear in dataset size.
    assert rss_after < rss_before + 50 * 1024 * 1024, f"RSS grew too much: {rss_before} -> {rss_after}"