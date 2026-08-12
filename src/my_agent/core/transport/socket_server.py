from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from my_agent.core.bus.envelope import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    HandlerError,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcSuccess,
    make_error,
)
from my_agent.core.trace.record import TraceRecord
from my_agent.core.trace.writer import TraceWriter
from my_agent.core.transport.ipc_broadcaster import IpcEventBroadcaster

type CommandHandler = Callable[[dict[str, Any]], Awaitable[Any]]

_MAX_LINE_BYTES = 1 * 1024 * 1024  # 1 MB per frame
logger = logging.getLogger(__name__)

_writer_var: ContextVar[asyncio.StreamWriter] = ContextVar("_writer_var")
_connection_id_var: ContextVar[str] = ContextVar("_connection_id_var")


def get_connection_writer() -> asyncio.StreamWriter:
    return _writer_var.get()


def get_connection_id() -> str:
    return _connection_id_var.get()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        broadcaster: IpcEventBroadcaster | None = None,
        trace: TraceWriter | None = None,
        on_disconnect: Callable[[asyncio.StreamWriter, str], Awaitable[None]] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._trace = trace
        self._handlers: dict[str, CommandHandler] = {}
        self._server: asyncio.AbstractServer | None = None
        self._broadcaster = broadcaster
        self._on_disconnect = on_disconnect

    # 注册一个方法名对应的命令处理函数
    def register(self, method: str, handler: CommandHandler) -> None:
        self._handlers[method] = handler

    # 启动 TCP 服务器；若端口已被占用则退出进程
    async def start(self) -> str:
        try:
            _r, w = await asyncio.open_connection(self._host, self._port)
            w.close()
            await w.wait_closed()
            raise SystemExit(f"core already running at {self._host}:{self._port}")
        except (ConnectionRefusedError, OSError):
            pass

        self._server = await asyncio.start_server(
            self._handle_connection, host=self._host, port=self._port, limit=_MAX_LINE_BYTES
        )
        return f"{self._host}:{self._port}"

    # 处理单个客户端连接，完成后关闭写流
    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        connection_id = f"conn-{uuid.uuid4().hex[:12]}"
        try:
            while True:
                line = await reader.readline()
                # 1. 读一行
                if not line:
                    break  # 2. 客户端断开

                # 3. 转成 JSON
                try:
                    raw = json.loads(line)
                except Exception:
                    await self._send(writer, make_error(None, PARSE_ERROR, "Parse error"))
                    continue

                # 4. 校验成 JsonRpcRequest
                try:
                    req = JsonRpcRequest.model_validate(raw)
                except Exception:
                    await self._send(writer, make_error(None, INVALID_REQUEST, "Invalid Request"))
                    continue

                # 5. 记录 trace
                if self._trace is not None:
                    client_id = str(writer.get_extra_info("peername", "<unknown>"))
                    self._trace.emit(
                        TraceRecord(
                            ts=_now(),
                            direction="CLIENT->CORE",
                            layer="ipc",
                            kind="command",
                            client_id=client_id,
                            data={"method": req.method, "id": req.id, "params": req.params},
                        )
                    )

                # 6. 查找 handler
                handler = self._handlers.get(req.method)
                if not handler:
                    await self._send(
                        writer,
                        make_error(req.id, METHOD_NOT_FOUND, f"Method {req.method} not found"),
                    )
                    continue

                # 7. 调用 handler
                try:
                    token = _writer_var.set(writer)
                    connection_token = _connection_id_var.set(connection_id)
                    try:
                        result = await handler(req.params)
                    finally:
                        _connection_id_var.reset(connection_token)
                        _writer_var.reset(token)
                except HandlerError as e:
                    await self._send(writer, make_error(req.id, e.code, str(e), e.data))
                    continue
                except ValidationError:
                    await self._send(writer, make_error(req.id, INVALID_PARAMS, "Invalid params"))
                    continue
                except Exception:
                    logger.exception("handler failed for method %s", req.method)
                    await self._send(writer, make_error(req.id, INTERNAL_ERROR, "Internal error"))
                    continue

                # 8. 返回结果
                await self._send(writer, JsonRpcSuccess(id=req.id, result=result))

        finally:
            if self._on_disconnect is not None:
                await self._on_disconnect(writer, "client_disconnected")

            if self._broadcaster is not None:
                self._broadcaster.unsubscribe(writer)

            writer.close()
            await writer.wait_closed()

    async def _send(self, writer: asyncio.StreamWriter, msg: JsonRpcSuccess | JsonRpcError) -> None:
        writer.write(msg.model_dump_json().encode() + b"\n")
        await writer.drain()

        kind = "error" if isinstance(msg, JsonRpcError) else "response"
        client_id = str(writer.get_extra_info("peername", "<unknown>"))
        if self._trace is not None:
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CORE->CLIENT",
                    layer="ipc",
                    kind=kind,
                    client_id=client_id,
                    data=msg.model_dump(),
                )
            )

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
