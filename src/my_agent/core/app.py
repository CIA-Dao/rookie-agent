from __future__ import annotations

import asyncio
import datetime
import fnmatch
import json
import logging
import uuid
from pathlib import Path
from time import monotonic
from typing import Any

from pydantic import BaseModel

from my_agent.core.bus.commands import (
    AgentRunCommand,
    AgentRunResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    PermissionRespondCommand,
    PermissionRespondResult,
    PongResult,
    SessionCloseCommand,
    SessionCloseResult,
    SessionCompactCommand,
    SessionCompactResult,
    SessionCreateCommand,
    SessionCreateResult,
    SessionGetHistoryCommand,
    SessionGetHistoryResult,
    SessionHeartbeatCommand,
    SessionHeartbeatResult,
    SessionSendMessageCommand,
    SessionSendMessageResult,
)
from my_agent.core.bus.envelope import EventPushEnvelope, HandlerError
from my_agent.core.config import (
    DEEPSEEK_MODEL_OPTIONS,
    Config,
    get_config,
    normalize_deepseek_model,
)
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.provider import DeepSeekProvider
from my_agent.core.logging_setup import setup_logging
from my_agent.core.mcp.server import McpServerManager
from my_agent.core.permissions.manager import PermissionManager
from my_agent.core.runner import AgentRunner
from my_agent.core.runs import events_file, new_run_id
from my_agent.core.session.manager import SessionManager
from my_agent.core.session.store import SessionStore
from my_agent.core.trace.record import TraceRecord
from my_agent.core.trace.writer import TraceWriter
from my_agent.core.transport.ipc_broadcaster import IpcEventBroadcaster
from my_agent.core.transport.socket_server import (
    SocketServer,
    get_connection_id,
    get_connection_writer,
)
from my_agent.core.user_config import save_deepseek_model
from my_agent.core.workspace import (
    WorkspaceCanonicalizationError,
    WorkspaceInUseError,
    WorkspaceLeaseRegistry,
    WorkspaceLeaseStaleError,
)

logger = logging.getLogger(__name__)

WORKSPACE_IN_USE = -32030
WORKSPACE_INVALID = -32031
MODEL_INVALID = -32033


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _maybe_connection_writer() -> asyncio.StreamWriter | None:
    try:
        return get_connection_writer()
    except LookupError:
        return None


def _maybe_connection_id() -> str | None:
    try:
        return get_connection_id()
    except LookupError:
        return None


class CoreApp:
    def __init__(self) -> None:
        self._started_at = monotonic()
        self._running_runs: set[asyncio.Task[None]] = set()
        self._config: Config | None = None
        self._trace: TraceWriter | None = None
        self._bus = EventBus()
        self._broadcaster: IpcEventBroadcaster | None = None
        self._session_manager: SessionManager | None = None
        self._permission_manager: PermissionManager | None = None
        self._connection_sessions: dict[asyncio.StreamWriter, set[str]] = {}
        self._connection_ids: dict[asyncio.StreamWriter, str] = {}
        self._workspace_leases = WorkspaceLeaseRegistry()
        self._workspace_lease_sweeper: asyncio.Task[None] | None = None
        self._mcp_manager: McpServerManager | None = None

    async def _trace_event_handler(self, event: BaseModel) -> None:
        assert self._trace is not None
        event_dict = event.model_dump()
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE",
                layer="event",
                kind="event",
                run_id=event_dict.get("run_id"),
                data=event_dict,
            )
        )

    async def _event_subscribe_handler(self, params: dict[str, Any]) -> EventSubscribeResult:
        command = EventSubscribeCommand.model_validate(params)
        writer = get_connection_writer()

        replayed_count = 0
        if command.replay_from_run is not None:
            replayed_count = await self._replay_events(
                command.replay_from_run,
                writer,
                command.topics,
            )
        assert self._broadcaster is not None
        sub_id = self._broadcaster.subscribe(
            writer,
            command.topics,
            command.scope,
        )

        return EventSubscribeResult(subscription_id=sub_id, replayed_count=replayed_count)

    async def _replay_events(
        self,
        run_id: str,
        writer: asyncio.StreamWriter,
        topics: list[str],
    ) -> int:
        assert self._config is not None
        runs_root = Path(self._config.runs_dir).expanduser()
        path = events_file(run_id, runs_root)
        if not path.exists():
            for candidate in (runs_root / "sessions").glob(f"*/runs/{run_id}/events.jsonl"):
                path = candidate
                break
        if not path.exists():
            for candidate in (runs_root / "workspaces").glob(
                f"*/sessions/*/runs/{run_id}/events.jsonl"
            ):
                path = candidate
                break
        if not path.exists():
            return 0

        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type: str = event.get("type", "")
            if not any(fnmatch.fnmatch(event_type, pattern) for pattern in topics):
                continue

            envelope = EventPushEnvelope(event=event)
            writer.write(envelope.model_dump_json().encode() + b"\n")
            count += 1

        if count:
            await writer.drain()

        return count

    async def _ping_handler(self, params: dict[str, Any]) -> PongResult:
        uptime_ms = max(0, int((monotonic() - self._started_at) * 1000))
        return PongResult(
            server_version="0.0.1",
            uptime_ms=uptime_ms,
            received_at=_now(),
        )

    async def _agent_run_handler(self, params: dict[str, Any]) -> AgentRunResult:
        assert self._session_manager is not None
        session_manager = self._session_manager
        writer = _maybe_connection_writer()
        command = AgentRunCommand.model_validate(params)
        goal = command.goal
        session = await session_manager.create(
            "one_shot",
            title=goal[:40],
            workspace_root=command.workspace_root,
        )
        if writer is not None:
            self._remember_connection_session(writer, session.id)
        run_id = new_run_id()

        async def run_one_shot() -> None:
            await session_manager.send_message(session.id, goal, run_id=run_id)

        run_task = asyncio.create_task(run_one_shot())
        self._running_runs.add(run_task)
        run_task.add_done_callback(self._running_runs.discard)
        return AgentRunResult(run_id=run_id)

    async def _session_create_handler(self, params: dict[str, Any]) -> SessionCreateResult:
        assert self._session_manager is not None
        writer = _maybe_connection_writer()
        command = SessionCreateCommand.model_validate(params)
        lease_id: str | None = None
        workspace_root = command.workspace_root
        if command.client_type == "tui":
            connection_id = _maybe_connection_id() or "direct-handler"
            try:
                lease = await self._workspace_leases.acquire(
                    command.workspace_root,
                    owner_session_id=f"pending-{uuid.uuid4().hex[:12]}",
                    owner_connection_id=connection_id,
                )
            except WorkspaceInUseError as exc:
                active = exc.active
                raise HandlerError(
                    WORKSPACE_IN_USE,
                    (
                        "workspace_in_use: the requested workspace overlaps "
                        f"active workspace {active.canonical_workspace.path}"
                    ),
                    {
                        "code": "workspace_in_use",
                        "reason": "overlapping_workspace",
                        "requested_workspace_root": str(exc.requested.path),
                        "active_workspace_root": str(active.canonical_workspace.path),
                        "owner_session_id": active.owner_session_id,
                        "owner_client_type": active.owner_client_type,
                    },
                ) from exc
            except WorkspaceCanonicalizationError as exc:
                raise HandlerError(
                    WORKSPACE_INVALID,
                    f"workspace_invalid: {exc}",
                    {
                        "code": "workspace_invalid",
                        "requested_workspace_root": command.workspace_root,
                    },
                ) from exc
            lease_id = lease.lease_id
            workspace_root = str(lease.canonical_workspace.path)

        try:
            session = await self._session_manager.create(
                command.mode,
                command.title,
                workspace_root=workspace_root,
            )
        except Exception:
            if lease_id is not None:
                await self._workspace_leases.release_lease(lease_id)
            raise

        if lease_id is not None:
            bound = await self._workspace_leases.bind_session(lease_id, session.id)
            if not bound:
                await self._session_manager.close(session.id)
                raise HandlerError(
                    WORKSPACE_IN_USE,
                    "workspace_in_use: workspace lease disappeared during session creation",
                    {"code": "workspace_in_use", "reason": "lease_lost"},
                )
        if writer is not None:
            self._remember_connection_session(writer, session.id)
        return SessionCreateResult(
            session_id=session.id,
            status=session.status,
            title=session.title,
        )

    async def _session_send_message_handler(
        self, params: dict[str, Any]
    ) -> SessionSendMessageResult:
        assert self._session_manager is not None
        writer = _maybe_connection_writer()
        command = SessionSendMessageCommand.model_validate(params)
        if writer is not None:
            self._remember_connection_session(writer, command.session_id)
        run_id = await self._session_manager.send_message(command.session_id, command.content)
        return SessionSendMessageResult(
            session_id=command.session_id,
            run_id=run_id,
        )

    async def _session_get_history_handler(self, params: dict[str, Any]) -> SessionGetHistoryResult:
        assert self._session_manager is not None
        command = SessionGetHistoryCommand.model_validate(params)
        messages = await self._session_manager.get_history(command.session_id)
        return SessionGetHistoryResult(
            session_id=command.session_id,
            messages=messages,
        )

    async def _session_close_handler(self, params: dict[str, Any]) -> SessionCloseResult:
        assert self._session_manager is not None
        command = SessionCloseCommand.model_validate(params)
        if self._permission_manager is not None:
            self._permission_manager.cancel_session(command.session_id, reason="session_closed")
        await self._session_manager.close(command.session_id)
        await self._workspace_leases.release_session(command.session_id)
        return SessionCloseResult(
            session_id=command.session_id,
            status="closed",
        )

    async def _session_heartbeat_handler(
        self, params: dict[str, Any]
    ) -> SessionHeartbeatResult:
        command = SessionHeartbeatCommand.model_validate(params)
        try:
            await self._workspace_leases.heartbeat(command.session_id)
        except WorkspaceLeaseStaleError as exc:
            raise HandlerError(
                -32032,
                str(exc),
                {"code": "session_stale", "session_id": command.session_id},
            ) from exc
        return SessionHeartbeatResult(session_id=command.session_id)

    async def _config_model_get_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        assert self._config is not None
        return {"model": self._config.llm.default_model, "models": DEEPSEEK_MODEL_OPTIONS}

    async def _config_model_set_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        assert self._config is not None
        raw_model = str(params.get("model", ""))
        model = normalize_deepseek_model(raw_model)
        if model is None:
            raise HandlerError(
                MODEL_INVALID,
                f"unsupported DeepSeek model: {raw_model}",
                {"models": DEEPSEEK_MODEL_OPTIONS},
            )
        self._config.llm.default_model = model
        save_deepseek_model(model)
        return {"model": model, "description": DEEPSEEK_MODEL_OPTIONS[model]}

    async def _permission_respond_handler(self, params: dict[str, Any]) -> PermissionRespondResult:
        assert self._permission_manager is not None
        command = PermissionRespondCommand.model_validate(params)
        self._permission_manager.respond(command.tool_use_id, command.decision)
        return PermissionRespondResult(ok=True)

    async def _session_compact_handler(self, params: dict[str, Any]) -> SessionCompactResult:
        assert self._session_manager is not None
        command = SessionCompactCommand.model_validate(params)
        return await self._session_manager.compact(command.session_id, focus=command.focus)

    def _remember_connection_session(
        self,
        writer: asyncio.StreamWriter,
        session_id: str,
    ) -> None:
        self._connection_sessions.setdefault(writer, set()).add(session_id)
        connection_id = _maybe_connection_id()
        if connection_id is not None:
            self._connection_ids[writer] = connection_id

    async def _connection_closed(
        self,
        writer: asyncio.StreamWriter,
        reason: str,
    ) -> None:
        session_ids = self._connection_sessions.pop(writer, set())
        for session_id in session_ids:
            if self._permission_manager is not None:
                self._permission_manager.cancel_session(session_id, reason=reason)
        connection_id = self._connection_ids.pop(writer, None)
        if connection_id is not None:
            await self._workspace_leases.release_connection(connection_id)

    async def _run_workspace_lease_sweeper(self) -> None:
        while True:
            await asyncio.sleep(10.0)
            await self._workspace_leases.reap_stale()

    async def run(self) -> None:
        self._config = get_config()
        config = self._config
        setup_logging(config)

        self._permission_manager = PermissionManager(
            policy_file=Path("~/.my-agent/policy.toml").expanduser()
        )
        self._mcp_manager = McpServerManager()

        if config.trace.enabled:
            trace_path = Path(config.trace.file).expanduser()
            self._trace = TraceWriter(trace_path)
            await self._trace.start()
            self._bus.subscribe(self._trace_event_handler)

        if self._config.mcp.servers:
            logger.info("mcp: starting %d server(s)", len(self._config.mcp.servers))
            await self._mcp_manager.start_all(self._config.mcp.servers)

        self._broadcaster = IpcEventBroadcaster(
            trace=self._trace,
            on_disconnect=self._connection_closed,
        )
        self._bus.subscribe(self._broadcaster.handle)
        compact_provider = DeepSeekProvider(config.llm.default_model)
        self._session_manager = SessionManager(
            SessionStore(Path(config.runs_dir).expanduser(), isolate_by_workspace=True),
            lambda: AgentRunner(
                config,
                bus=self._bus,
                trace=self._trace,
                permission_manager=self._permission_manager,
                mcp_manager=self._mcp_manager,
            ),
            self._bus,
            provider=compact_provider,
        )

        server = SocketServer(
            config.host,
            config.port,
            self._broadcaster,
            trace=self._trace,
            on_disconnect=self._connection_closed,
        )
        server.register("core.ping", self._ping_handler)
        server.register("agent.run", self._agent_run_handler)
        server.register("event.subscribe", self._event_subscribe_handler)
        server.register("session.create", self._session_create_handler)
        server.register("session.send_message", self._session_send_message_handler)
        server.register("session.get_history", self._session_get_history_handler)
        server.register("session.close", self._session_close_handler)
        server.register("session.heartbeat", self._session_heartbeat_handler)
        server.register("session.compact", self._session_compact_handler)
        server.register("config.model.get", self._config_model_get_handler)
        server.register("config.model.set", self._config_model_set_handler)
        server.register("permission.respond", self._permission_respond_handler)
        await server.start()
        self._workspace_lease_sweeper = asyncio.create_task(
            self._run_workspace_lease_sweeper()
        )
        logger.info("Server started at %s:%s", config.host, config.port)

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            if self._workspace_lease_sweeper is not None:
                self._workspace_lease_sweeper.cancel()
                await asyncio.gather(self._workspace_lease_sweeper, return_exceptions=True)
                self._workspace_lease_sweeper = None
            for run_task in list(self._running_runs):
                run_task.cancel()
            if self._running_runs:
                await asyncio.gather(*self._running_runs, return_exceptions=True)
            await server.stop()
            if self._mcp_manager is not None:
                await self._mcp_manager.stop_all()
            if self._trace is not None:
                await self._trace.stop()


def run() -> None:
    try:
        asyncio.run(CoreApp().run())
    except KeyboardInterrupt:
        pass
