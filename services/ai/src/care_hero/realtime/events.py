"""Thread-safe event journal. No raw media, landmarks or exception text is logged."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


class EventJournal:
    """Serialize events in sequence; retain only each stage's most recent status."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._epoch = time.monotonic()
        self._sequence = 0
        self._latest: dict[str, dict[str, Any]] = {}
        self._closed = False

    def emit(self, stage: str, status: str, **data: Any) -> None:
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "elapsed_seconds": round(time.monotonic() - self._epoch, 4),
                "stage": stage,
                "status": status,
                "data": data,
            }
            self._file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            self._file.flush()
            self._latest[stage] = event

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._latest)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._file.close()
