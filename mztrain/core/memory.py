"""mztrain.core.memory
====================

*Dynamic batch sizing* — automatically find the largest ``batch_size``
that fits in current free RAM.  This is the user-facing "no OOM" promise
of the project.

Two public pieces:

* :func:`available_memory_bytes`  — best-effort current free RAM (bytes).
* :func:`calibrate_batch_size`    — binary-search up to the OOM threshold
  for a given ``probe_fn(batch_size) -> None`` callback.

We deliberately keep zero hard requirements (``psutil`` is optional): if
it isn't installed we fall back to a conservative static estimate derived
from platform heuristics.  This is part of the "memzero" philosophy: every
extra dep bloats the framework's baseline RSS.
"""

from __future__ import annotations

import gc
import os
import platform
import shutil
import sys
from typing import Callable, Optional


def available_memory_bytes() -> int:
    """Return approximate free memory available to this process.

    Uses ``psutil`` if available; otherwise tries ``/proc/meminfo`` on
    Linux, ``vm_stat`` on macOS, or ``GlobalMemoryStatusEx`` on Windows.
    Returns 0 when nothing works (we will then fall back to a small batch).
    """
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        elif system == "Darwin":
            import subprocess

            out = subprocess.check_output(["vm_stat"]).decode()
            page = 4096
            free = 0
            for line in out.splitlines():
                if "Pages free" in line or "Pages inactive" in line:
                    free += int(line.split()[-1].rstrip("."))
            return free * page
        elif system == "Windows":
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys)
    except Exception:
        pass
    return 0


def process_rss_bytes() -> int:
    """Resident set size of the current Python process."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage if sys.platform != "darwin" else usage  # already bytes on linux, KB on darwin
    except Exception:
        try:
            import psutil  # type: ignore

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return 0


def calibrate_batch_size(
    probe_fn: Callable[[int], None],
    *,
    initial: int = 64,
    minimum: int = 1,
    maximum: Optional[int] = None,
    safety_factor: float = 0.6,
    max_iters: int = 6,
) -> int:
    """Binary-search for the largest ``batch_size`` that doesn't OOM.

    Parameters
    ----------
    probe_fn
        Callable that takes an int batch_size and runs a probe forward+back
        pass; raises ``MemoryError`` (or returns ``"oom"``) if it can't fit.
    initial
        Starting guess for the batch size.
    minimum, maximum
        Search bounds.  ``maximum`` defaults to ``initial * 2**max_iters``.
    safety_factor
        Multiply the discovered maximum by this number (e.g. 0.6) so we
        leave headroom for parameter gradients and optimiser state.
    """
    if maximum is None:
        maximum = initial * (2 ** max_iters)
    lo, hi = minimum, max(minimum, initial)
    chosen = hi
    while hi <= maximum:
        try:
            probe_fn(hi)
            chosen = hi
            hi <<= 1
        except (MemoryError, RuntimeError) as exc:
            if "OOM" in str(exc).upper() or isinstance(exc, MemoryError):
                break
            raise
        finally:
            gc.collect()
    # Refine: the highest passing size is `chosen`; we don't keep halving
    # to avoid training too long in the calibration step.
    return max(minimum, int(chosen * safety_factor))


__all__ = ["available_memory_bytes", "process_rss_bytes", "calibrate_batch_size"]