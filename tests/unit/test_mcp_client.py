from __future__ import annotations

import json
from typing import Any

import pytest

from my_agent.core.mcp.client import McpClient, McpServerUnavailableError, McpToolError


class _FakeMcpClient(McpClient):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self.lines = lines
        self.writes: list[dict[str, Any]] = []

    async def _write_line(self, line: str) -> None:
        self.writes.append(json.loads(line))

    async def _read_line(self) -> str:
        if not self.lines:
            raise McpServerUnavailableError("no more fake lines")
        return self.lines.pop(0)


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        self.waited = True


async def test_call_ignores_notifications_and_returns_matching_response() -> None:
    client = _FakeMcpClient(
        [
            "not json",
            json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"}),
            json.dumps({"jsonrpc": "2.0", "id": "2", "result": {"wrong": True}}),
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}),
        ]
    )

    result = await client._call("tools/list", {})

    assert result == {"ok": True}
    assert client.writes == [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    ]


async def test_call_raises_tool_error_for_json_rpc_error() -> None:
    client = _FakeMcpClient(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32602, "message": "bad arguments"},
                }
            )
        ]
    )

    with pytest.raises(McpToolError, match="bad arguments"):
        await client._call("tools/call", {"name": "search"})


async def test_notify_writes_json_rpc_notification_without_id() -> None:
    client = _FakeMcpClient([])

    await client._notify("notifications/initialized", {})

    assert client.writes == [
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    ]


async def test_list_tools_parses_valid_tool_definitions() -> None:
    client = _FakeMcpClient(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {
                                "name": "search",
                                "description": "Search docs",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                },
                            },
                            {"name": "", "description": "skip empty names"},
                            {"name": 123, "description": "skip invalid names"},
                        ]
                    },
                }
            )
        ]
    )

    tools = await client.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "search"
    assert tools[0].description == "Search docs"
    assert tools[0].input_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }


async def test_call_tool_joins_text_content() -> None:
    client = _FakeMcpClient(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image", "data": "ignored"},
                            {"type": "text", "text": "world"},
                        ]
                    },
                }
            )
        ]
    )

    result = await client.call_tool("search", {"query": "agent"})

    assert result == "hello\nworld"
    assert client.writes == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "agent"}},
        }
    ]


async def test_connect_tcp_runs_initialize_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeMcpClient(
        [json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}})]
    )
    fake_writer = _FakeWriter()

    async def fake_open_connection(host: str, port: int) -> tuple[object, _FakeWriter]:
        assert host == "127.0.0.1"
        assert port == 9999
        return object(), fake_writer

    monkeypatch.setattr("asyncio.open_connection", fake_open_connection)
    monkeypatch.setattr(client, "_write_line", client._write_line)

    await client.connect_tcp("127.0.0.1", 9999)

    assert client.writes == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "my-agent", "version": "0.1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    ]


async def test_close_closes_tcp_writer() -> None:
    client = McpClient()
    writer = _FakeWriter()
    client._writer = writer  # type: ignore[attr-defined]

    await client.close()

    assert writer.closed
    assert client._reader is None  # type: ignore[attr-defined]
    assert client._writer is None  # type: ignore[attr-defined]


async def test_close_terminates_stdio_process_and_clears_state() -> None:
    client = McpClient()
    process = _FakeProcess()
    client._transport = "stdio"  # type: ignore[attr-defined]
    client._proc = process  # type: ignore[assignment]
    client._writer = _FakeWriter()  # type: ignore[assignment]

    await client.close()

    assert process.terminated
    assert process.waited
    assert not process.killed
    assert client._reader is None  # type: ignore[attr-defined]
    assert client._writer is None  # type: ignore[attr-defined]
    assert client._proc is None  # type: ignore[attr-defined]
    assert client._transport == ""  # type: ignore[attr-defined]
