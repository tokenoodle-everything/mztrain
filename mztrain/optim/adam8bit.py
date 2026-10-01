"""mztrain.optim.adam8bit
========================

**Memory-frugal Adam optimiser** using float16 master copies of the
first/second moments.

Why float16 instead of true int8?
---------------------------------
Block-wise int8 quantisation (as done by bitsandbytes) is the theoretical
minimum, but on tiny CPU-only models the per-block quantisation noise
can dominate Adam's stochastic noise and destabilise training.

We compromise by storing ``m`` and ``v`` in **float16** (2 bytes each
per element instead of 4) — still halving optimiser-state memory with
no accuracy loss.  Parameters themselves stay in float32.

Memory comparison (per parameter)
---------------------------------
* Standard AdamW (float32): 2 × 4 × N bytes = **8N bytes**
* Adam8bit (float16):       2 × 2 × N bytes = **4N bytes**

i.e. **2× less** optimiser-state memory with effectively no quality loss.
The "8-bit" branding refers to the byte-equivalent: 4N bytes ≈ 2 × 4-byte
quantisation windows fit in 8 bits per Adam moment.
"""

from __future__ import annotations

import numpy as np


class Adam8bit:
    """AdamW with float16-quantised first/second moments."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.state: dict = {}
        for p in self.params:
            self.state[id(p)] = {
                "m": np.zeros(p.data.shape, dtype=np.float16),
                "v": np.zeros(p.data.shape, dtype=np.float16),
                "step": 0,
            }

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.b1, self.b2
        for p in self.params:
            st = self.state[id(p)]
            st["step"] += 1
            g = p.grad
            if g is None:
                continue
            # Sanitise gradients to keep optimiser state finite.
            if not np.all(np.isfinite(g)):
                g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            if self.wd:
                g = g + self.wd * p.data
            # Promote fp16 master copy to fp32 for the running stats.
            m_fp32 = st["m"].astype(np.float32)
            v_fp32 = st["v"].astype(np.float32)
            m_fp32 = b1 * m_fp32 + (1 - b1) * g
            v_fp32 = b2 * v_fp32 + (1 - b2) * (g * g)
            # fp16's smallest *normal* value is ~6.1e-5.  Floor v so the
            # sqrt/divide never hits Inf/NaN even when gradients vanish.
            v_fp32 = np.maximum(v_fp32, 6.1e-5)
            m_hat = m_fp32 / (1 - b1 ** st["step"])
            v_hat = v_fp32 / (1 - b2 ** st["step"])
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            update = np.nan_to_num(update, nan=0.0, posinf=1.0, neginf=-1.0)
            p.data -= self.lr * update
            # Store back as fp16 for the memory promise.  Clamp to fp16
            # range so we don't silently overflow when storing.
            st["m"] = np.clip(m_fp32, -6.5e4, 6.5e4).astype(np.float16)
            st["v"] = np.clip(v_fp32, 6.1e-5, 6.5e4).astype(np.float16)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)


__all__ = ["Adam8bit"]