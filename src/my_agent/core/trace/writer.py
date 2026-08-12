from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TextIO

from my_agent.core.trace.record import TraceRecord

logger = logging.getLogger(__name__)
_MAX_QUEUE_SIZE = 2048
_STOP_TIMEOUT_SECONDS = 5.0


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._queue: asyncio.Queue[TraceRecord] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._task: asyncio.Task[None] | None = None
        self._file: TextIO | None = None
        self._disabled = False
        self.dropped_records = 0

    async def start(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")
        except OSError as exc:
            self._disabled = True
            logger.warning("trace disabled: %s", exc)
            return
        self._task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=_STOP_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("trace shutdown timed out; dropping pending records")
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._file is not None:
            self._file.close()
            self._file = None

    def emit(self, record: TraceRecord) -> None:
        if self._disabled:
            self.dropped_records += 1
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped_records += 1

    async def _drain(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                if self._file is not None:
                    self._file.write(record.model_dump_json() + "\n")
                    self._file.flush()
            except OSError as exc:
                self._disabled = True
                logger.warning("trace disabled after write failure: %s", exc)
            finally:
                self._queue.task_done()
