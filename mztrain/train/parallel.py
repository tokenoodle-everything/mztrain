"""mztrain.train.parallel
========================

Multi-process CPU parallelism — *real* multi-core usage.

Why
---
CPython's GIL means every thread in a single process is restricted to one
CPU at a time.  PyTorch's DataLoader workers are therefore restricted to
one core (each) — a 16-core machine typically sees 4-6 cores busy at best.

We side-step the GIL entirely by spawning independent worker processes,
each computing its own micro-batch forward+backward.  Workers exchange
model parameters and gradients through a single ``multiprocessing.RawArray``
shared-memory region (no GIL contention, no pickling, no sockets).

Workflow
--------
1. Main process broadcasts current parameters into the shared buffer.
2. Each worker process pulls a batch from a shared queue, runs forward +
   backward, and writes its gradient into the shared buffer.
3. Main process sums, divides by world size, applies the optimiser step.
4. Repeat.

This is essentially *synchronous data-parallel* training — the simplest
correct scheme, scaled down for laptops.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..core.tensor import Tensor
from ..utils.system import cpu_count


@dataclass
class ParallelConfig:
    world_size: int = 0  # 0 = auto -> cpu_count
    micro_batch_size: int = 32
    broadcast_every: int = 1
    verbose: bool = False


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------
def _worker_loop(
    worker_id: int,
    param_specs,
    shared_param_buf,
    shared_grad_buf,
    batch_queue,
    result_queue,
    model_factory,
    loss_fn,
):
    """One worker process.

    Each worker holds its own copy of the model.  It pulls parameters from
    ``shared_param_buf`` every ``broadcast_every`` batches and pushes
    computed gradients back into ``shared_grad_buf``.
    """
    model = model_factory()
    params = model.parameters()
    total = sum(p.data.size for p in params)
    assert sum(int(np.prod(s)) for s in param_specs) == total
    param_view = np.frombuffer(shared_param_buf, dtype=np.float32, count=total).reshape(-1)
    grad_view = np.frombuffer(shared_grad_buf, dtype=np.float32, count=total).reshape(-1)

    def sync_in():
        offset = 0
        for p in params:
            sz = p.data.size
            p.data = param_view[offset:offset + sz].reshape(p.data.shape).copy()
            if p.grad is None:
                p.grad = np.zeros_like(p.data)
            offset += sz

    def sync_out():
        offset = 0
        for p in params:
            sz = p.data.size
            grad_view[offset:offset + sz] = p.grad.reshape(-1)
            offset += sz

    sync_in()

    while True:
        try:
            xb, yb = batch_queue.get(timeout=0.5)
        except queue.Empty:
            try:
                _ = batch_queue.get_nowait()
                break
            except queue.Empty:
                continue
            except Exception:
                break
        if xb is None:
            break
        try:
            xb_t = Tensor(xb, requires_grad=False)
            out = model(xb_t)
            loss = loss_fn(out, yb)
            model.zero_grad()
            loss.backward()
            sync_out()
            result_queue.put(("ok", worker_id, float(loss.data)))
        except Exception as e:  # pragma: no cover
            result_queue.put(("err", worker_id, repr(e)))
        sync_in()


def parallel_train(
    model_factory: Callable,
    loss_fn: Callable,
    batch_iter,
    steps: int,
    cfg: ParallelConfig,
    optimizer_step_fn: Optional[Callable] = None,
):
    """Run ``steps`` synchronous data-parallel steps.

    Parameters
    ----------
    model_factory
        Zero-arg callable returning a *fresh* model each call.
    loss_fn
        ``callable(pred, target) -> Tensor`` with autograd.
    batch_iter
        Iterable yielding ``(np.ndarray xb, np.ndarray yb)``.
    steps
        Number of global steps to run.
    cfg
        :class:`ParallelConfig`.
    optimizer_step_fn
        Called after each global step with the sum-of-grads shared buffer.
    """
    world = cfg.world_size or cpu_count()
    world = max(1, world)
    proto = model_factory()
    params = proto.parameters()
    total = sum(int(p.data.size) for p in params)
    if total == 0:
        raise ValueError("Model has no parameters.")
    if cfg.verbose:
        print(f"[parallel] world_size={world} total params={total}")

    shared_p = mp.RawArray("f", total)
    shared_g = mp.RawArray("f", total)
    param_specs = [p.data.shape for p in params]
    bq: "mp.Queue" = mp.Queue(maxsize=world * 2)
    rq: "mp.Queue" = mp.Queue()

    view = np.frombuffer(shared_p, dtype=np.float32, count=total)
    offset = 0
    for p in params:
        sz = p.data.size
        view[offset:offset + sz] = p.data.reshape(-1)
        offset += sz

    procs = []
    for wid in range(world):
        proc = mp.Process(
            target=_worker_loop,
            args=(wid, param_specs, shared_p, shared_g, bq, rq,
                  model_factory, loss_fn),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    losses = []
    try:
        for step in range(steps):
            for _ in range(world):
                xb, yb = next(batch_iter)
                bq.put((xb, yb))
            for _ in range(world):
                tag, wid, info = rq.get(timeout=120)
                if tag == "err":
                    print(f"[parallel worker {wid}] {info}")
                else:
                    losses.append(info)
            gview = np.frombuffer(shared_g, dtype=np.float32, count=total)
            avg = gview / world
            offset = 0
            for p in params:
                sz = p.data.size
                p.grad = avg[offset:offset + sz].reshape(p.data.shape).astype(np.float32)
                offset += sz
            if optimizer_step_fn is None:
                p.data -= 1e-3 * p.grad
            else:
                optimizer_step_fn()
            offset = 0
            for p in params:
                sz = p.data.size
                view[offset:offset + sz] = p.data.reshape(-1)
                offset += sz
            if cfg.verbose and step % 10 == 0:
                print(f"[parallel] step {step:4d} loss={losses[-1]:.4f}")
    finally:
        for _ in range(world):
            try:
                bq.put_nowait((None, None))
            except Exception:
                pass
        for proc in procs:
            proc.join(timeout=2)
    return losses


__all__ = ["ParallelConfig", "parallel_train"]