from __future__ import annotations

import asyncio
from typing import cast

from my_agent.core.bus.events import RunStartedEvent
from my_agent.core.transport.ipc_broadcaster import IpcEventBroadcaster


class _BrokenWriter:
    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        raise BrokenPipeError

    def get_extra_info(self, _name: str, default: object = None) -> object:
        return default


async def test_broadcaster_calls_on_disconnect_when_push_fails() -> None:
    disconnected = asyncio.Event()
    reasons: list[str] = []
    writer = cast(asyncio.StreamWriter, _BrokenWriter())

    async def on_disconnect(seen_writer: asyncio.StreamWriter, reason: str) -> None:
        assert seen_writer is writer
        reasons.append(reason)
        disconnected.set()

    broadcaster = IpcEventBroadcaster(on_disconnect=on_disconnect)
    broadcaster.subscribe(writer, ["run.*"], "global")

    await broadcaster.handle(
        RunStartedEvent(
            run_id="run-1",
            goal="hello",
            ts="2026-01-01T00:00:00+00:00",
        )
    )

    await asyncio.wait_for(disconnected.wait(), timeout=1.0)

    assert reasons == ["push_failed"]
    assert broadcaster._subscriptions == []
