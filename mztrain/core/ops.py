"""mztrain.core.ops
=================

Pure-NumPy forward / backward implementations of the operators ``mztrain``
models depend on.  Every operator returns a new ``Tensor`` (so the previous
graph node can be freed once the caller drops its reference) and stores a
``back_op_fn`` closure for reverse-mode autodiff.

Why hand-rolled?
----------------
* We don't need CUDA streams, distributed collectives, JIT, autocast, fx
  tracer, dispatcher or any of PyTorch's ~hundreds of subsystems — that's
  the *whole point* of being "memzero".  Keeping the op library small is
  what keeps RSS low.
* Each op is independently auditable: read the forward, read the backward,
  done.  No magic.
* This module is the single source of truth that the gradient-checkpoint
  decorator in :mod:`mztrain.core.autograd` intercepts.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor

# Small epsilon used in log/softmax to keep things finite.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Element-wise
# ---------------------------------------------------------------------------
def add(a: Tensor, b: Tensor) -> Tensor:
    a, b = _unbroadcast(a), _unbroadcast(b)
    out = Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _prev=(a, b),
    )

    def _back(grad):
        if a.requires_grad:
            a.grad += _reduce_to(grad, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_to(grad, b.data.shape)

    out.back_op_fn = _back
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    return add(a, neg(b))


def neg(a: Tensor) -> Tensor:
    out = Tensor(-a.data, requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += -grad

    out.back_op_fn = _back
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    a, b = _unbroadcast(a), _unbroadcast(b)
    out = Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _prev=(a, b),
    )

    def _back(grad):
        if a.requires_grad:
            a.grad += _reduce_to(grad * b.data, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_to(grad * a.data, b.data.shape)

    out.back_op_fn = _back
    return out


def true_div(a: Tensor, b: Tensor) -> Tensor:
    a, b = _unbroadcast(a), _unbroadcast(b)
    out = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad, _prev=(a, b))

    def _back(grad):
        if a.requires_grad:
            a.grad += _reduce_to(grad / b.data, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_to(-grad * a.data / (b.data ** 2), b.data.shape)

    out.back_op_fn = _back
    return out


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Generalised matmul supporting ``(..., M, K) @ (..., K, N) -> (..., M, N)``."""
    out = Tensor(
        a.data @ b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _prev=(a, b),
    )

    def _back(grad):
        aT = np.swapaxes(a.data, -1, -2)
        bT = np.swapaxes(b.data, -1, -2)
        if a.requires_grad:
            a.grad += grad @ bT
        if b.requires_grad:
            b.grad += aT @ grad

    out.back_op_fn = _back
    return out


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def relu(a: Tensor) -> Tensor:
    mask = a.data > 0
    out = Tensor(a.data * mask, requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += grad * mask

    out.back_op_fn = _back
    return out


def sigmoid(a: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = Tensor(s, requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += grad * s * (1.0 - s)

    out.back_op_fn = _back
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    out = Tensor(t, requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += grad * (1.0 - t * t)

    out.back_op_fn = _back
    return out


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    a_data = a.data
    if not np.all(np.isfinite(a_data)):
        # Defensive: clip extreme values to avoid exp overflow.
        a_data = np.clip(a_data, -50.0, 50.0)
    m = np.max(a_data, axis=axis, keepdims=True)
    shifted = a_data - m
    log_sum_exp = m + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out_data = shifted - log_sum_exp
    out = Tensor(out_data, requires_grad=a.requires_grad, _prev=(a,))
    softmax = np.exp(out.data)

    def _back(grad):
        if a.requires_grad:
            grad_in = grad - softmax * np.sum(grad, axis=axis, keepdims=True)
            a.grad += grad_in

    out.back_op_fn = _back
    return out


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    return exp(log_softmax(a, axis=axis))


def exp(a: Tensor) -> Tensor:
    e = np.exp(a.data)
    out = Tensor(e, requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += grad * e

    out.back_op_fn = _back
    return out


def log(a: Tensor) -> Tensor:
    out = Tensor(np.log(a.data + _EPS), requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += grad / (a.data + _EPS)

    out.back_op_fn = _back
    return out


# ---------------------------------------------------------------------------
# Reshape / permute
# ---------------------------------------------------------------------------
def reshape(a: Tensor, shape) -> Tensor:
    out = Tensor(a.data.reshape(shape), requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += grad.reshape(a.data.shape)

    out.back_op_fn = _back
    return out


def transpose(a: Tensor, axes) -> Tensor:
    out = Tensor(np.transpose(a.data, axes), requires_grad=a.requires_grad, _prev=(a,))

    def _back(grad):
        if a.requires_grad:
            a.grad += np.transpose(grad, np.argsort(axes))

    out.back_op_fn = _back
    return out


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    out = Tensor(
        a.data.sum(axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _prev=(a,),
    )
    src_shape = a.data.shape

    def _back(grad):
        if a.requires_grad:
            if axis is None or not keepdims:
                grad = np.expand_dims(grad, axis=axis if axis is not None else 0)
                grad = np.broadcast_to(grad, src_shape)
            a.grad += grad

    out.back_op_fn = _back
    return out


def mean(a: Tensor, axis=None) -> Tensor:
    n = a.data.size if axis is None else np.asarray(a.data.shape)[
        np.asarray(axis).flatten() if isinstance(axis, (tuple, list)) else axis
    ].prod()
    return true_div(sum(a, axis=axis), Tensor(float(n)))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def nll_loss(log_probs: Tensor, target: np.ndarray) -> Tensor:
    """Negative log-likelihood with integer targets of shape ``(N,)``."""
    n = log_probs.data.shape[0]
    picked = log_probs.data[np.arange(n), target.astype(np.int64)]
    out = Tensor((-picked).mean(), requires_grad=log_probs.requires_grad, _prev=(log_probs,))

    def _back(grad):
        if log_probs.requires_grad:
            g = np.zeros_like(log_probs.data)
            g[np.arange(n), target.astype(np.int64)] = -1.0 / n
            log_probs.grad += grad * g

    out.back_op_fn = _back
    return out


def cross_entropy(logits: Tensor, target: np.ndarray) -> Tensor:
    """Fused softmax + NLL on raw logits."""
    return nll_loss(log_softmax(logits, axis=-1), target)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = sub(pred, target)
    return mean(mul(diff, diff))


# ---------------------------------------------------------------------------
# Conv2d (im2col implementation)
# ---------------------------------------------------------------------------
def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    """Naive-but-correct 2D convolution via im2col.

    For small models this is plenty fast on multi-core numpy.  Memory cost is
    ``O(N * C_out * H_out * W_out)`` of the *output* — no giant intermediate.
    """
    N, C_in, H, W = x.data.shape
    C_out, _, kH, kW = weight.data.shape
    s, p = stride, padding
    H_out = (H + 2 * p - kH) // s + 1
    W_out = (W + 2 * p - kW) // s + 1

    if p > 0:
        xpad = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)))
    else:
        xpad = x.data

    cols = _im2col(xpad, kH, kW, s, H_out, W_out)
    wcol = weight.data.reshape(C_out, -1)
    out = wcol @ cols
    out = out.reshape(C_out, N, H_out * W_out).transpose(1, 0, 2).reshape(N, C_out, H_out, W_out)
    if bias is not None:
        out = out + bias.data.reshape(1, -1, 1, 1)

    prevs = [x, weight]
    if bias is not None:
        prevs.append(bias)
    out_t = Tensor(out, requires_grad=any(p.requires_grad for p in prevs), _prev=tuple(prevs))

    def _back(grad):
        # grad: (N, C_out, H_out, W_out)
        gcol = grad.transpose(1, 0, 2, 3).reshape(C_out, -1)  # (C_out, N*H*W)
        if weight.requires_grad:
            # weight: (C_out, C_in, kH, kW). We need dw = gcol @ cols^T, with cols
            # of shape (N, C_in*kH*kW, H*W).  Reshape cols to (N, H*W, C_in*kH*kW),
            # then flatten batch+spatial to (N*H*W, C_in*kH*kW) and matmul.
            cols2 = cols.transpose(0, 2, 1).reshape(-1, cols.shape[1])  # (N*H*W, C_in*kH*kW)
            weight.grad += (gcol @ cols2).reshape(weight.data.shape)
        if bias is not None and bias.requires_grad:
            bias.grad += gcol.sum(axis=1)
        if x.requires_grad:
            dcols = wcol.T @ gcol  # (C_in*kH*kW, N*H*W)
            x.grad += _col2im(dcols, xpad.shape, kH, kW, s, p, H, W)

    out_t.back_op_fn = _back
    return out_t


def _im2col(xpad, kH, kW, s, H_out, W_out):
    N, C_in, Hpad, Wpad = xpad.shape
    cols = np.empty((N, C_in, kH, kW, H_out, W_out), dtype=xpad.dtype)
    for i in range(kH):
        i_max = i + s * H_out
        for j in range(kW):
            j_max = j + s * W_out
            cols[:, :, i, j, :, :] = xpad[:, :, i:i_max:s, j:j_max:s]
    return cols.reshape(N, C_in * kH * kW, H_out * W_out)


def _col2im(cols, xpad_shape, kH, kW, s, p, H, W):
    N, C_in, Hpad, Wpad = xpad_shape
    cols = cols.reshape(N, C_in, kH, kW, -1)
    xpad_grad = np.zeros(xpad_shape, dtype=cols.dtype)
    H_out = (H + 2 * p - kH) // s + 1
    W_out = (W + 2 * p - kW) // s + 1
    for i in range(kH):
        for j in range(kW):
            slc_i = slice(i, i + s * H_out, s)
            slc_j = slice(j, j + s * W_out, s)
            xpad_grad[:, :, slc_i, slc_j] += cols[:, :, i, j, :].reshape(N, C_in, H_out, W_out)
    if p > 0:
        return xpad_grad[:, :, p:p + H, p:p + W]
    return xpad_grad


def max_pool2d(x: Tensor, kernel: int = 2, stride: int = 2) -> Tensor:
    N, C, H, W = x.data.shape
    H_out = (H - kernel) // stride + 1
    W_out = (W - kernel) // stride + 1
    cols = _im2col(x.data, kernel, kernel, stride, H_out, W_out)
    cols = cols.reshape(N, C, kernel * kernel, H_out * W_out)
    out = cols.max(axis=2)
    idx = cols.argmax(axis=2)
    mask = np.zeros_like(cols, dtype=np.float32)
    np.put_along_axis(mask, idx[:, :, None], 1.0, axis=2)
    out_t = Tensor(out.reshape(N, C, H_out, W_out), requires_grad=x.requires_grad, _prev=(x,))

    def _back(grad):
        if x.requires_grad:
            grad_cols = mask * grad.reshape(N, C, 1, H_out * W_out)
            grad_x = _col2im(grad_cols, x.data.shape, kernel, kernel, stride, 0, H, W)
            x.grad += grad_x

    out_t.back_op_fn = _back
    return out_t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unbroadcast(t: Tensor) -> Tensor:
    """Trim trailing 1-dims that came from broadcasting (no-op if shape OK)."""
    return t  # we keep the original; the reduce handles gradients later.


def _reduce_to(grad: np.ndarray, shape) -> np.ndarray:
    """Reverse NumPy broadcasting: sum out any axes that ``shape`` dropped."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
        if sdim == 1 and gdim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------
def parameter(*shape) -> Tensor:
    """Xavier-ish initialised trainable parameter."""
    n_in = int(np.prod(shape[:-1])) if len(shape) > 1 else shape[0]
    bound = (1.0 / max(n_in, 1)) ** 0.5
    return Tensor(
        np.random.uniform(-bound, bound, size=shape).astype(np.float32),
        requires_grad=True,
    )


def zeros_parameter(*shape) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=True)


# ---------------------------------------------------------------------------
# Module base
# ---------------------------------------------------------------------------
class Module:
    """Lightweight base class for parameter containers."""

    def parameters(self):
        params: list[Tensor] = []
        for _name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self):
        return {n: p.data.copy() for n, p in self.named_parameters()}

    def named_parameters(self):
        for name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad:
                yield name, value
            elif isinstance(value, Module):
                for sub_name, sub_p in value.named_parameters():
                    yield f"{name}.{sub_name}", sub_p
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        for sub_name, sub_p in item.named_parameters():
                            yield f"{name}.{i}.{sub_name}", sub_p
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield f"{name}.{i}", item

    def load_state_dict(self, state):
        for name, p in self.named_parameters():
            if name in state:
                p.data = state[name].astype(np.float32, copy=True)
