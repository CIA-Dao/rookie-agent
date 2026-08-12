from __future__ import annotations

from typing import Any

from textual.message import Message


class CoreConnected(Message):
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id


class CoreDisconnected(Message):
    def __init__(self, reason: str, *, transport_lost: bool = False) -> None:
        super().__init__()
        self.reason = reason
        self.transport_lost = transport_lost


class CoreEvent(Message):
    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event
