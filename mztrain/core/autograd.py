"""mztrain.core.autograd
======================

**Gradient checkpointing** — the central memory-saving trick of memzero-train.

Background
----------
PyTorch (and every other production framework) keeps *every* forward
activation tensor alive until :func:`backward` finishes, because each
intermediate value is needed to compute the gradient of parameters that
depend on it.  For an L-layer network this costs ``O(L)`` extra activation
memory.

The classical *gradient checkpointing* / *re-materialisation* technique
(Chen et al., 2016 — "Training Deep Nets with Sublinear Memory Cost")
splits the network into *segments*.  For each segment we run forward and
**discard** the activations immediately; during backward we re-run forward
on that segment to recompute the activations, then compute the local
gradient, then discard again.

This trades a ~33% slowdown for ``sqrt(L)`` memory — going from
``O(L)`` activations to ``O(sqrt(L))``.

API
---
``CheckpointedBlock`` wraps any forward callable.  ``checkpoint_sequential``
chains several of them.
"""

from __future__ import annotations

from typing import Callable, Iterable

from .tensor import Tensor


class CheckpointedBlock:
    """Wraps a forward callable so activations are discarded until backward.

    >>> block = CheckpointedBlock(layer)
    >>> out = block(x)
    >>> loss.backward()
    """

    def __init__(self, forward_fn: Callable[..., Tensor]):
        self.forward_fn = forward_fn
        self._last_inputs: tuple[Tensor, ...] = ()
        self._last_output: Tensor | None = None

    def __call__(self, *inputs: Tensor) -> Tensor:
        self._last_inputs = inputs
        self._last_output = self.forward_fn(*inputs)
        out = self._last_output
        if not out.requires_grad:
            return out
        original_back = out.back_op_fn

        def recompute(grad_out):
            new_out = self.forward_fn(*self._last_inputs)
            if original_back is not None:
                original_back(grad_out)
            self._last_output = None
            self._last_inputs = ()

        out.back_op_fn = recompute
        return out

    def parameters(self):
        if hasattr(self.forward_fn, "parameters"):
            return self.forward_fn.parameters()
        return []


def checkpoint_sequential(blocks: Iterable[CheckpointedBlock], x: Tensor) -> Tensor:
    """Apply several :class:`CheckpointedBlock` in sequence."""
    for b in blocks:
        x = b(x)
    return x


def checkpoint(forward_fn: Callable[..., Tensor], *inputs: Tensor) -> Tensor:
    """Convenience: a single-shot checkpointed block."""
    return CheckpointedBlock(forward_fn)(*inputs)


__all__ = ["checkpoint", "CheckpointedBlock", "checkpoint_sequential"]