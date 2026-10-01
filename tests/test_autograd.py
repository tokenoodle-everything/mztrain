"""Tests for the autograd ops."""

import numpy as np

import mztrain as mzt


def test_add_backward():
    a = mzt.Tensor([[1.0, 2.0]], requires_grad=True)
    b = mzt.Tensor([[3.0, 4.0]], requires_grad=True)
    out = (a + b).sum()
    out.backward()
    assert np.allclose(a.grad, [[1.0, 1.0]])
    assert np.allclose(b.grad, [[1.0, 1.0]])


def test_matmul_backward():
    np.random.seed(0)
    a = mzt.Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    b = mzt.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    out = (a @ b).sum()
    out.backward()
    # d_a = ones @ b.T
    expected_a_grad = np.ones((2, 4)) @ b.data.T
    assert np.allclose(a.grad, expected_a_grad, atol=1e-5)


def test_relu_backward():
    a = mzt.Tensor([[-1.0, 2.0, -3.0, 4.0]], requires_grad=True)
    out = a.relu().sum()
    out.backward()
    assert np.allclose(a.grad, [[0.0, 1.0, 0.0, 1.0]])


def test_cross_entropy():
    logits = mzt.Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
    y = np.array([0, 1])
    loss = mzt.cross_entropy(logits, y)
    assert loss.data.shape == ()
    loss.backward()
    assert logits.grad.shape == (2, 3)


def test_checkpointed_block():
    from mztrain.core import CheckpointedBlock
    from mztrain.models import MLP

    block = CheckpointedBlock(MLP(in_features=4, hidden=[8], out_features=2))
    xb = np.random.randn(3, 4).astype(np.float32)
    yb = np.array([0, 1, 0])
    out = block(mzt.Tensor(xb))
    loss = mzt.cross_entropy(out, yb)
    loss.backward()
    for p in block.parameters():
        assert p.grad is not None
        assert np.all(np.isfinite(p.grad))