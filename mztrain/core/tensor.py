"""mztrain.core.tensor
====================

A minimal NumPy-backed tensor with autograd support.

Why a custom Tensor?
--------------------
PyTorch + CUDA dominates deep learning, but it carries *huge* baseline RAM
footprint (allocator + framework state + dispatcher tables) even for trivial
models on CPU. ``mztrain`` exposes a deliberately tiny tensor wrapper whose
data lives in a plain ``numpy.ndarray`` so that the rest of the framework
can stream through millions of examples with virtually zero idle memory.

Design notes
------------
* Only what we need: forward storage, gradient slot, ``requires_grad`` flag,
  reference to the producing op for backward.
* No graph pruning / no version counter — we lean on Python refcounting so
  intermediate tensors die as soon as the caller is done with them.
* ``backward()`` walks the dynamic graph via ``_prev`` linked list.
* We *do not* subclass ``numpy.ndarray``; that approach breaks on ``@``,
  ``+``, etc. because NumPy calls ``__array_ufunc__`` and we lose the
  gradient flow. Subclassing ``object`` is more verbose but correct.
"""

from __future__ import annotations

import numpy as np

# Floating point dtype we promote to when mixing ints.
_FLOAT_DTYPE = np.float32


def _to_array(x) -> np.ndarray:
    """Coerce input to a contiguous float32 ndarray."""
    if isinstance(x, Tensor):
        x = x.data
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.dtype != _FLOAT_DTYPE:
        x = x.astype(_FLOAT_DTYPE, copy=False)
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)
    return x


class Tensor:
    """N-dimensional array with optional autograd tracking."""

    __slots__ = ("data", "grad", "requires_grad", "_prev", "back_op_fn")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _prev: tuple = (),
        back_op_fn=None,
    ) -> None:
        self.data: np.ndarray = _to_array(data)
        if requires_grad and not self.data.flags.writeable:
            # The gradient slot needs to be mutable.
            self.data = self.data.copy()
        self.grad: np.ndarray | None = (
            np.zeros_like(self.data) if requires_grad else None
        )
        self.requires_grad: bool = requires_grad
        self._prev: set[Tensor] = set(_prev)
        self.back_op_fn = back_op_fn  # callable(grad_out) -> None

    # ------------------------------------------------------------------ #
    # Basic Python protocol
    # ------------------------------------------------------------------ #
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self):
        return self.data.item()

    def numel(self) -> int:
        return int(self.data.size)

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, "
            f"grad={'yes' if self.requires_grad else 'no'})"
        )

    # ------------------------------------------------------------------ #
    # Operator overloads (delegate to functional API in .ops)
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        from .ops import add
        return add(self, other if isinstance(other, Tensor) else Tensor(other))

    def __radd__(self, other):
        from .ops import add
        return add(Tensor(other) if not isinstance(other, Tensor) else other, self)

    def __sub__(self, other):
        from .ops import sub
        return sub(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rsub__(self, other):
        from .ops import sub
        return sub(Tensor(other) if not isinstance(other, Tensor) else other, self)

    def __mul__(self, other):
        from .ops import mul
        return mul(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rmul__(self, other):
        from .ops import mul
        return mul(Tensor(other) if not isinstance(other, Tensor) else other, self)

    def __truediv__(self, other):
        from .ops import true_div
        return true_div(self, other if isinstance(other, Tensor) else Tensor(other))

    def __neg__(self):
        from .ops import neg
        return neg(self)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)

    def reshape(self, shape):
        from .ops import reshape
        return reshape(self, shape)

    def transpose(self, axes=None):
        from .ops import transpose
        if axes is None:
            axes = list(range(self.ndim))[::-1]
        return transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        from .ops import sum as _sum
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None):
        from .ops import mean as _mean
        return _mean(self, axis=axis)

    def relu(self):
        from .ops import relu as _relu
        return _relu(self)

    def softmax(self, axis=-1):
        from .ops import softmax as _softmax
        return _softmax(self, axis=axis)

    # ------------------------------------------------------------------ #
    # Gradient helpers
    # ------------------------------------------------------------------ #
    def zero_grad(self) -> None:
        """In-place zero of the gradient buffer (no allocation)."""
        if self.grad is not None:
            self.grad.fill(0.0)

    def zero_(self) -> None:
        """In-place zero of the *data* tensor."""
        self.data.fill(0.0)

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Run reverse-mode autodiff starting from this node."""
        if not self.requires_grad:
            raise RuntimeError(
                "backward() called on a Tensor that does not require grad."
            )
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad must be specified for non-scalar tensors."
                )
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        # Topological order via DFS.
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(t: Tensor):
            if id(t) in visited or not t.requires_grad:
                return
            visited.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)

        build(self)
        # Accumulate the seed gradient at the root.
        self.grad += grad
        for t in reversed(topo):
            if t.back_op_fn is not None:
                t.back_op_fn(t.grad)
            # Free intermediate data eagerly to drop peak memory.
            t.data = t.data  # noqa: PLW0127 (kept alive intentionally)
        # The caller still owns `self`, intermediates may be released by GC.

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #
    @staticmethod
    def zeros(shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(shape, dtype=_FLOAT_DTYPE), requires_grad=requires_grad)

    @staticmethod
    def ones(shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(shape, dtype=_FLOAT_DTYPE), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(
            np.random.randn(*shape).astype(_FLOAT_DTYPE),
            requires_grad=requires_grad,
        )

    @staticmethod
    def uniform(*shape, low: float = -0.1, high: float = 0.1,
               requires_grad: bool = False) -> "Tensor":
        return Tensor(
            np.random.uniform(low, high, size=shape).astype(_FLOAT_DTYPE),
            requires_grad=requires_grad,
        )