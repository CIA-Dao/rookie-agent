from __future__ import annotations

from typing import Any

import pytest

from my_agent.core.config import McpServerConfig
from my_agent.core.mcp.client import McpToolDef
from my_agent.core.mcp.server import McpServerManager


class _FakeClient:
    def __init__(
        self,
        *,
        tools: list[McpToolDef] | None = None,
        fail_connect: bool = False,
    ) -> None:
        self.tools = tools or []
        self.fail_connect = fail_connect
        self.closed = False
        self.connects: list[tuple[str, tuple[Any, ...]]] = []

    async def connect_stdio(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        self.connects.append(("stdio", (command, args, env)))
        if self.fail_connect:
            raise RuntimeError("connect failed")

    async def connect_tcp(self, host: str, port: int) -> None:
        self.connects.append(("tcp", (host, port)))
        if self.fail_connect:
            raise RuntimeError("connect failed")

    async def list_tools(self) -> list[McpToolDef]:
        return self.tools

    async def close(self) -> None:
        self.closed = True


async def test_start_all_connects_tcp_and_wraps_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(tools=[McpToolDef(name="search", description="Search docs")])
    monkeypatch.setattr("my_agent.core.mcp.server.McpClient", lambda: client)

    manager = McpServerManager()
    await manager.start_all(
        [McpServerConfig(name="docs", transport="tcp", host="127.0.0.1", port=9999)]
    )

    tools = manager.get_tools()

    assert client.connects == [("tcp", ("127.0.0.1", 9999))]
    assert len(tools) == 1
    assert tools[0].name == "docs__search"


async def test_start_all_connects_stdio_and_passes_args_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr("my_agent.core.mcp.server.McpClient", lambda: client)

    manager = McpServerManager()
    await manager.start_all(
        [
            McpServerConfig(
                name="fs",
                transport="stdio",
                command="mcp-filesystem",
                args=["D:/project"],
                env={"TOKEN": "x"},
            )
        ]
    )

    assert client.connects == [("stdio", ("mcp-filesystem", ["D:/project"], {"TOKEN": "x"}))]


async def test_start_all_skips_failed_server(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = [
        _FakeClient(fail_connect=True),
        _FakeClient(tools=[McpToolDef(name="echo", description="Echo")]),
    ]
    monkeypatch.setattr("my_agent.core.mcp.server.McpClient", lambda: clients.pop(0))

    manager = McpServerManager()
    await manager.start_all(
        [
            McpServerConfig(name="bad", transport="tcp"),
            McpServerConfig(name="ok", transport="tcp"),
        ]
    )

    tools = manager.get_tools()

    assert len(tools) == 1
    assert tools[0].name == "ok__echo"


async def test_stop_all_closes_clients_and_clears_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(tools=[McpToolDef(name="echo", description="Echo")])
    monkeypatch.setattr("my_agent.core.mcp.server.McpClient", lambda: client)

    manager = McpServerManager()
    await manager.start_all([McpServerConfig(name="ok", transport="tcp")])

    await manager.stop_all()

    assert client.closed
    assert manager.get_tools() == []


async def test_stdio_config_requires_command() -> None:
    manager = McpServerManager()

    with pytest.raises(ValueError, match="stdio transport requires 'command'"):
        await manager._connect(McpServerConfig(name="fs", transport="stdio"))
