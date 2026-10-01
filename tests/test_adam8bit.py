"""Tests for the 8-bit Adam optimiser."""

import numpy as np

import mztrain as mzt
from mztrain.core import Tensor


def test_step_runs():
    a = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    opt = mzt.Adam8bit([a], lr=1e-2)
    a.grad = np.array([0.1, -0.1])
    before = a.data.copy()
    opt.step()
    assert not np.allclose(a.data, before)
    assert np.all(np.isfinite(a.data))


def test_memory_reduction():
    """Adam8bit should use roughly half the per-parameter memory of fp32 Adam."""
    p = Tensor(np.zeros(1024, dtype=np.float32), requires_grad=True)
    opt = mzt.Adam8bit([p], lr=1e-2)
    # 1024 * 2 bytes (m) + 1024 * 2 bytes (v) = 4096 bytes for state
    expected = 1024 * 2 * 2
    actual = sum(arr.nbytes for arr in [opt.state[id(p)]["m"], opt.state[id(p)]["v"]])
    assert actual <= expected + 32, f"expected ~{expected} bytes, got {actual}"