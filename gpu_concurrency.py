"""
GPU Concurrency Manager - Controls concurrent access to GPU-bound video generation.

Without concurrency limits, multiple simultaneous WebSocket sessions will all
compete for the same GPU, causing severe slowdowns or hangs for everyone.

This module provides a central asyncio semaphore that gates GPU access so only
a bounded number of pipelines run at a time.  Excess requests queue up and
receive position updates, or are rejected after a configurable timeout.

Configuration via environment variables:
    GPU_MAX_CONCURRENT  – max simultaneous pipelines (default: 1)
    GPU_QUEUE_TIMEOUT   – seconds before a queued request is rejected (default: 60)
"""

import asyncio
import os
import time
from typing import Optional


class GPUConcurrencyManager:
    """
    Singleton that limits concurrent GPU-bound video generation.

    Usage in an async handler::

        mgr = GPUConcurrencyManager()

        # Optionally tell the client their queue position while waiting
        acquired = await mgr.acquire(timeout=60.0)
        if not acquired:
            # send SERVER_BUSY to client
            return

        try:
            # ... run GPU pipeline ...
        finally:
            mgr.release()
    """

    _instance: Optional["GPUConcurrencyManager"] = None

    def __new__(cls) -> "GPUConcurrencyManager":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._initialized = False
            cls._instance = inst
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._max_concurrent = int(os.getenv("GPU_MAX_CONCURRENT", "1"))
        self._queue_timeout = float(os.getenv("GPU_QUEUE_TIMEOUT", "60"))

        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._active: int = 0
        self._waiting: int = 0
        self._lock = asyncio.Lock()
        self._initialized = True

        print(
            f"[GPUConcurrency] Initialized: max_concurrent={self._max_concurrent}, "
            f"queue_timeout={self._queue_timeout}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a GPU slot, blocking until one is available or *timeout* expires.

        Args:
            timeout: Seconds to wait.  ``None`` uses the default from
                     ``GPU_QUEUE_TIMEOUT``.  Set to ``0`` for a non-blocking
                     try-acquire.

        Returns:
            ``True`` if the slot was acquired, ``False`` if timed out.
        """
        if timeout is None:
            timeout = self._queue_timeout

        async with self._lock:
            self._waiting += 1

        position = self._waiting
        print(
            f"[GPUConcurrency] Request queued (position={position}, "
            f"active={self._active}/{self._max_concurrent})"
        )

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiting -= 1
            print(
                f"[GPUConcurrency] Request timed out after {timeout}s "
                f"(active={self._active}/{self._max_concurrent})"
            )
            return False

        async with self._lock:
            self._waiting -= 1
            self._active += 1

        print(
            f"[GPUConcurrency] Slot acquired "
            f"(active={self._active}/{self._max_concurrent}, "
            f"waiting={self._waiting})"
        )
        return True

    def release(self):
        """Release a previously acquired GPU slot."""
        # Use a synchronous decrement; fine because we only touch an int.
        self._active = max(0, self._active - 1)
        self._semaphore.release()
        print(
            f"[GPUConcurrency] Slot released "
            f"(active={self._active}/{self._max_concurrent}, "
            f"waiting={self._waiting})"
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of GPU usage for the ``/gpu/status`` endpoint."""
        return {
            "active": self._active,
            "waiting": self._waiting,
            "max_concurrent": self._max_concurrent,
            "queue_timeout": self._queue_timeout,
        }

    @property
    def queue_position(self) -> int:
        """Current number of waiters (approximate)."""
        return self._waiting

    @property
    def is_at_capacity(self) -> bool:
        """``True`` when all slots are occupied."""
        return self._active >= self._max_concurrent
