"""Finite queues with explicit policies for freshness and dropped work."""

from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedChannel(Generic[T]):
    def __init__(self, capacity: int, *, replace_oldest: bool) -> None:
        self._queue: queue.Queue[T] = queue.Queue(maxsize=capacity)
        self._write_lock = threading.Lock()
        self.replace_oldest = replace_oldest
        self.dropped = 0

    def put(self, item: T) -> bool:
        with self._write_lock:
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self.dropped += 1
                if not self.replace_oldest:
                    return False
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(item)
                return True

    def get(self, timeout: float = 0.05) -> T:
        return self._queue.get(timeout=timeout)

    @property
    def size(self) -> int:
        return self._queue.qsize()
