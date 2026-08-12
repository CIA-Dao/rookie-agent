from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any


class McpServerUnavailableError(Exception):
    pass


class McpToolError(Exception):
    pass


@dataclass
class McpToolDef:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


log = logging.getLogger(__name__)


class McpClient:
    def __init__(self) -> None:
        self._id = 0
        self._lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._transport = ""
        self._stderr_task: asyncio.Task[None] | None = None

    async def _initialize(self) -> None:
        await self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "my-agent",
                    "version": "0.1.0",
                },
            },
        )
        await self._notify("notifications/initialized", {})

    async def connect_tcp(self, host: str, port: int) -> None:
        self._reader, self._writer = await asyncio.open_connection(host, port)
        self._transport = "tcp"
        await self._initialize()

    async def connect_stdio(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        merged_env = {**os.environ, **(env or {})}

        self._proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )

        if self._proc.stdout is None or self._proc.stdin is None:
            raise McpServerUnavailableError("stdio pipes unavailable")

        self._reader = self._proc.stdout
        self._writer = self._proc.stdin
        self._transport = "stdio"
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._initialize()

    async def close(self) -> None:
        if self._transport == "stdio" and self._proc is not None:
            if self._writer is not None:
                self._writer.close()
                try:
                    await self._writer.wait_closed()
                except Exception:
                    pass

            try:
                if getattr(self._proc, "returncode", None) is None:
                    self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
                try:
                    await self._proc.wait()
                except Exception:
                    pass
            except Exception:
                pass

            if self._stderr_task is not None:
                try:
                    await asyncio.wait_for(self._stderr_task, timeout=1.0)
                except TimeoutError:
                    self._stderr_task.cancel()
                    try:
                        await self._stderr_task
                    except asyncio.CancelledError:
                        pass
                except asyncio.CancelledError:
                    pass
                self._stderr_task = None
        elif self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass

        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        self._reader = None
        self._writer = None
        self._proc = None
        self._transport = ""

    async def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            while True:
                data = await self._proc.stderr.readline()
                if data == b"":
                    break
                line = data.decode("utf-8", errors="replace").rstrip()
                if line:
                    log.debug("mcp stderr: %s", line)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.debug("mcp stderr drain stopped", exc_info=True)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._call(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        parts: list[str] = []
        for item in result.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text", "")
                parts.append(text if isinstance(text, str) else str(text))
        return "\n".join(parts)

    async def list_tools(self) -> list[McpToolDef]:
        result = await self._call("tools/list", {})
        tools: list[McpToolDef] = []
        for item in result.get("tools", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = item.get("description", "")
            input_schema = item.get("inputSchema", {})

            tools.append(
                McpToolDef(
                    name=name,
                    description=description if isinstance(description, str) else "",
                    input_schema=input_schema if isinstance(input_schema, dict) else {},
                )
            )
        return tools

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        req_id = self._id

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        async with self._lock:
            await self._write_line(json.dumps(request))

            while True:
                line = await self._read_line()

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("ignore non-json mcp line: %r", line[:200])
                    continue

                message_id = message.get("id")

                if message_id is None:
                    log.debug("ignore mcp notification: %s", message.get("method"))
                    continue

                if str(message_id) != str(req_id):
                    continue

                if "error" in message:
                    error = message["error"]
                    message_text = error.get("message", str(error))
                    code = error.get("code")
                    raise McpToolError(f"{message_text} (code={code})")

                result = message.get("result", {})
                if isinstance(result, dict):
                    return result
                return {}

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._write_line(json.dumps(request))

    async def _write_line(self, line: str) -> None:
        if self._writer is None:
            raise McpServerUnavailableError("MCP client is not connected")

        self._writer.write((line + "\n").encode("utf-8"))
        await self._writer.drain()

    async def _read_line(self) -> str:
        if self._reader is None:
            raise McpServerUnavailableError("MCP client is not connected")

        while True:
            try:
                data = await asyncio.wait_for(self._reader.readline(), timeout=30.0)
            except TimeoutError as exc:
                raise McpServerUnavailableError("MCP server read timeout") from exc

            if data == b"":
                raise McpServerUnavailableError("MCP server closed connection")

            line = data.decode("utf-8", errors="replace").strip()
            if line:
                return line
