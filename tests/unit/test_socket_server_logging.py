from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pytest import LogCaptureFixture

from my_agent.core.bus.envelope import HandlerError
from my_agent.core.transport.socket_server import SocketServer


async def test_socket_server_logs_handler_exceptions(
    free_port: int,
    caplog: LogCaptureFixture,
) -> None:
    async def failing_handler(params: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    server = SocketServer("127.0.0.1", free_port)
    server.register("test.fail", failing_handler)
    await server.start()

    caplog.set_level(logging.ERROR, logger="my_agent.core.transport.socket_server")
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
        request = {"jsonrpc": "2.0", "id": "req-1", "method": "test.fail", "params": {}}
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()

        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()

    response = json.loads(line)
    assert response["error"]["code"] == -32603
    assert "handler failed for method test.fail" in caplog.text


async def test_socket_server_converts_handler_error_to_json_rpc_error(
    free_port: int,
) -> None:
    async def handler_error(params: dict[str, Any]) -> None:
        raise HandlerError(-32099, "custom failure", {"field": "goal"})

    server = SocketServer("127.0.0.1", free_port)
    server.register("test.handler_error", handler_error)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
        request = {
            "jsonrpc": "2.0",
            "id": "req-2",
            "method": "test.handler_error",
            "params": {},
        }
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()

        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()

    response = json.loads(line)
    assert response["error"]["code"] == -32099
    assert response["error"]["message"] == "custom failure"
    assert response["error"]["data"] == {"field": "goal"}


async def test_socket_server_calls_on_disconnect(free_port: int) -> None:
    disconnected = asyncio.Event()
    reasons: list[str] = []

    async def on_disconnect(_writer: asyncio.StreamWriter, reason: str) -> None:
        reasons.append(reason)
        disconnected.set()

    server = SocketServer("127.0.0.1", free_port, on_disconnect=on_disconnect)
    await server.start()

    try:
        _reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
        writer.close()
        await writer.wait_closed()

        await asyncio.wait_for(disconnected.wait(), timeout=5.0)
    finally:
        await server.stop()

    assert reasons == ["client_disconnected"]
