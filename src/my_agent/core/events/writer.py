from __future__ import annotations

import logging
from pathlib import Path
from typing import IO

from pydantic import BaseModel

from my_agent.core.events.bus import EventBus

logger = logging.getLogger(__name__)


class EventWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: IO[str] | None = None

    async def __aenter__(self) -> EventWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe(self._handle)

    async def _handle(self, event: BaseModel) -> None:
        if self._file is None:
            return
        try:
            self._file.write(event.model_dump_json() + "\n")
            self._file.flush()
        except (OSError, ValueError) as e:
            logger.error("EventWriter: failed to write event: %s", e)
