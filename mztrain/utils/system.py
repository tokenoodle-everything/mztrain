"""mztrain.utils.system
=====================

Lightweight OS / hardware introspection.  No third-party deps.
"""

from __future__ import annotations

import os
import platform
from typing import Optional


def cpu_count(fallback: int = 4) -> int:
    """Best-effort physical CPU count.

    Returns ``os.cpu_count()`` when available, otherwise ``fallback``.
    We avoid pulling in ``psutil`` for this single piece of information.
    """
    try:
        n = os.cpu_count()
        if n and n > 0:
            return n
    except Exception:
        pass
    system = platform.system()
    if system == "Windows":
        try:
            import ctypes

            class SYSTEM_INFO(ctypes.Structure):
                _fields_ = [
                    ("dwOemId", ctypes.c_ulong),
                    ("dwPageSize", ctypes.c_ulong),
                    ("lpMinimumApplicationAddress", ctypes.c_void_p),
                    ("lpMaximumApplicationAddress", ctypes.c_void_p),
                    ("dwActiveProcessorMask", ctypes.c_void_p),
                    ("dwNumberOfProcessors", ctypes.c_ulong),
                    ("dwProcessorType", ctypes.c_ulong),
                    ("dwAllocationGranularity", ctypes.c_ulong),
                    ("wProcessorLevel", ctypes.c_ushort),
                    ("wProcessorRevision", ctypes.c_ushort),
                ]

            info = SYSTEM_INFO()
            ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(info))
            return int(info.dwNumberOfProcessors) or fallback
        except Exception:
            return fallback
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                count = sum(1 for line in f if line.startswith("processor"))
            if count > 0:
                return count
        except Exception:
            pass
    return fallback


def platform_name() -> str:
    return platform.platform()


__all__ = ["cpu_count", "platform_name"]