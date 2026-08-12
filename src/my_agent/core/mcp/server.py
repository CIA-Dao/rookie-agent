from __future__ import annotations

import logging

from my_agent.core.config import McpServerConfig
from my_agent.core.mcp.client import McpClient
from my_agent.core.mcp.tool import McpTool

log = logging.getLogger(__name__)


class McpServerManager:
    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}
        self._tools: list[McpTool] = []

    def get_tools(self) -> list[McpTool]:
        return list(self._tools)

    async def start_all(self, servers: list[McpServerConfig]) -> None:
        for cfg in servers:
            try:
                client = await self._connect(cfg)
                tool_defs = await client.list_tools()
                for tool_def in tool_defs:
                    self._tools.append(McpTool(client, cfg.name, tool_def))
                self._clients[cfg.name] = client
                log.info(
                    "mcp: server '%s' connected, %d tool(s) discovered",
                    cfg.name,
                    len(tool_defs),
                )
            except Exception:
                log.exception("mcp: server '%s' failed to start, skipping", cfg.name)

    async def stop_all(self) -> None:
        for name, client in list(self._clients.items()):
            try:
                await client.close()
                log.info("mcp: server '%s' closed", name)
            except Exception:
                log.warning("mcp: error closing server '%s'", name)
        self._clients.clear()
        self._tools.clear()

    async def _connect(self, cfg: McpServerConfig) -> McpClient:
        client = McpClient()

        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"mcp server '{cfg.name}': stdio transport requires 'command'")
            await client.connect_stdio(cfg.command, cfg.args, cfg.env or None)
        elif cfg.transport == "tcp":
            await client.connect_tcp(cfg.host, cfg.port)
        else:
            raise ValueError(f"mcp server '{cfg.name}': unknown transport '{cfg.transport}'")

        return client
