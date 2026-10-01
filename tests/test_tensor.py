"""Tests for mztrain.core.tensor."""

import numpy as np

from mztrain.core import Tensor


def test_creation_and_shape():
    t = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    assert t.shape == (2, 3)
    assert t.requires_grad is True
    assert t.grad.shape == (2, 3)


def test_no_grad():
    t = Tensor(np.zeros((4, 4)))
    assert t.requires_grad is False
    assert t.grad is None


def test_backward_runs():
    a = Tensor([[2.0]], requires_grad=True)
    loss = a * a * 3
    loss.backward()
    assert np.allclose(a.grad, [[12.0]])