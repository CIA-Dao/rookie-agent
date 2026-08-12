from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from my_agent.core.bus.envelope import JsonRpcRequest

type EventHandler = Callable[[dict[str, Any]], Awaitable[None]]

_MAX_LINE_BYTES = 1 * 1024 * 1024


class IpcError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.data = data


class SocketClient:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._event_handlers: list[EventHandler] = []

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(
            self._host,
            self._port,
            limit=_MAX_LINE_BYTES,
        )

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
            except TimeoutError:
                pass
            except (ConnectionError, OSError):
                # Windows may report ERROR_NETNAME_DELETED/connection reset
                # while closing a peer that has already gone away.
                pass

    def on_event(self, handler: EventHandler) -> None:
        self._event_handlers.append(handler)

    async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._writer is None:
            raise RuntimeError("not connected; call connect() first")
        req_id = str(uuid.uuid4())
        request = JsonRpcRequest(id=req_id, method=method, params=params)
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut

        self._writer.write(request.model_dump_json().encode() + b"\n")
        await self._writer.drain()

        return await fut

    async def run_event_loop(self) -> None:
        if self._reader is None:
            raise RuntimeError("not connected; call connect() first")

        try:
            while True:
                try:
                    line = await self._reader.readline()
                except (ConnectionResetError, OSError):
                    break

                if not line:
                    break

                await self._dispatch(line)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()

    async def _dispatch(self, line: bytes) -> None:
        try:
            msg: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return

        if "jsonrpc" in msg:
            req_id: str | None = msg.get("id")
            if req_id is None:
                return

            if req_id and req_id in self._pending:
                fut = self._pending.pop(req_id)
                if not fut.done():
                    if "error" in msg:
                        err = msg["error"]
                        fut.set_exception(
                            IpcError(
                                err.get("code", -1),
                                err.get("message", "unknown"),
                                err.get("data"),
                            )
                        )
                    else:
                        fut.set_result(msg.get("result") or {})
            return

        if msg.get("kind") == "event":
            event_data: dict[str, Any] = msg.get("event", {})
            for handler in self._event_handlers:
                await handler(event_data)
