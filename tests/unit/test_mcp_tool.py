from __future__ import annotations

from typing import Any

from my_agent.core.mcp.client import (
    McpServerUnavailableError,
    McpToolDef,
    McpToolError,
)
from my_agent.core.mcp.tool import McpTool


class _FakeMcpClient:
    def __init__(self, result: str = "ok", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


def test_mcp_tool_uses_server_prefixed_name_and_schema() -> None:
    client = _FakeMcpClient()
    tool_def = McpToolDef(
        name="search",
        description="Search docs",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    tool = McpTool(client, "docs", tool_def)

    assert tool.name == "docs__search"
    assert tool.description == "Search docs"
    assert tool.input_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }


def test_mcp_tool_uses_default_description_and_schema() -> None:
    client = _FakeMcpClient()
    tool_def = McpToolDef(name="lookup", description="")

    tool = McpTool(client, "kb", tool_def)

    assert tool.description == "MCP tool from kb"
    assert tool.input_schema == {"type": "object", "properties": {}}


async def test_mcp_tool_invokes_underlying_client() -> None:
    client = _FakeMcpClient(result="remote result")
    tool = McpTool(client, "docs", McpToolDef(name="search", description="Search"))

    result = await tool.invoke({"query": "agent"})

    assert not result.is_error
    assert result.content == "remote result"
    assert client.calls == [("search", {"query": "agent"})]


async def test_mcp_tool_returns_error_when_server_unavailable() -> None:
    client = _FakeMcpClient(error=McpServerUnavailableError("offline"))
    tool = McpTool(client, "docs", McpToolDef(name="search", description="Search"))

    result = await tool.invoke({"query": "agent"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert result.content == "mcp server 'docs' unavailable: offline"


async def test_mcp_tool_returns_error_when_remote_tool_fails() -> None:
    client = _FakeMcpClient(error=McpToolError("bad args"))
    tool = McpTool(client, "docs", McpToolDef(name="search", description="Search"))

    result = await tool.invoke({"query": "agent"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert result.content == "mcp tool 'docs__search' error: bad args"


async def test_mcp_tool_returns_error_when_unexpected_error_occurs() -> None:
    client = _FakeMcpClient(error=ValueError("boom"))
    tool = McpTool(client, "docs", McpToolDef(name="search", description="Search"))

    result = await tool.invoke({"query": "agent"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert result.content == "mcp tool 'docs__search' unexpected error: boom"
