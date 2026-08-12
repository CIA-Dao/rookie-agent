from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from my_agent.core.app import CoreApp
from my_agent.core.bus.envelope import HandlerError
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse, UsageStats
from my_agent.core.runner import RunOutcome
from my_agent.core.session.manager import SessionManager
from my_agent.core.session.model import Session
from my_agent.core.session.store import SessionStore


class _FakeRunner:
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str,
        session: Session,
        store: SessionStore,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        store.append_message(session.id, "assistant", f"answer: {goal}", run_id=run_id)
        return RunOutcome(status="success", result="", reason=None)


class _FakeCompactProvider:
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        return LlmResponse(
            stop_reason="end_turn",
            text="compact summary",
            usage=UsageStats(input_tokens=100, output_tokens=10),
        )


async def test_session_create_handler_returns_session_result(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path)
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())

    result = await app._session_create_handler(
        {"type": "session.create", "mode": "chat", "title": "learning"}
    )

    assert result.session_id.startswith("sess-")
    assert result.status == "active"
    assert result.title == "learning"
    assert store.read_meta(result.session_id).title == "learning"


async def test_tui_session_create_rejects_overlapping_workspace(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path / "runs")
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())
    parent = tmp_path / "project"

    first = await app._session_create_handler(
        {
            "type": "session.create",
            "mode": "chat",
            "workspace_root": str(parent),
            "client_type": "tui",
        }
    )

    with pytest.raises(HandlerError) as exc_info:
        await app._session_create_handler(
            {
                "type": "session.create",
                "mode": "chat",
                "workspace_root": str(parent / "nested"),
                "client_type": "tui",
            }
        )

    assert exc_info.value.code == -32030
    assert exc_info.value.data["code"] == "workspace_in_use"
    assert exc_info.value.data["owner_session_id"] == first.session_id


async def test_tui_session_close_releases_workspace_lease(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path / "runs")
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())
    workspace = tmp_path / "project"

    first = await app._session_create_handler(
        {
            "type": "session.create",
            "mode": "chat",
            "workspace_root": str(workspace),
            "client_type": "tui",
        }
    )
    await app._session_close_handler({"type": "session.close", "session_id": first.session_id})

    replacement = await app._session_create_handler(
        {
            "type": "session.create",
            "mode": "chat",
            "workspace_root": str(workspace),
            "client_type": "tui",
        }
    )
    assert replacement.session_id != first.session_id


async def test_connection_disconnect_releases_tui_workspace_lease(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path / "runs")
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())
    writer = cast(asyncio.StreamWriter, object())
    app._connection_ids[writer] = "conn-test"
    await app._workspace_leases.acquire(
        str(tmp_path / "project"),
        owner_session_id="sess-1",
        owner_connection_id="conn-test",
    )

    await app._connection_closed(writer, "client_disconnected")

    assert await app._workspace_leases.active_leases() == []


async def test_session_heartbeat_refreshes_tui_lease(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path / "runs")
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())
    result = await app._session_create_handler(
        {
            "type": "session.create",
            "mode": "chat",
            "workspace_root": str(tmp_path / "project"),
            "client_type": "tui",
        }
    )

    heartbeat = await app._session_heartbeat_handler(
        {"type": "session.heartbeat", "session_id": result.session_id}
    )

    assert heartbeat.session_id == result.session_id
    assert heartbeat.status == "active"


async def test_session_send_message_handler_returns_run_id(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path)
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())
    session = await app._session_manager.create("chat")

    result = await app._session_send_message_handler(
        {"type": "session.send_message", "session_id": session.id, "content": "hello"}
    )

    assert result.session_id == session.id
    assert result.run_id
    assert store.read_messages(session.id) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer: hello"},
    ]


async def test_session_get_history_handler_returns_messages(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path)
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())
    session = await app._session_manager.create("chat")
    store.append_message(session.id, "user", "hello")

    result = await app._session_get_history_handler(
        {"type": "session.get_history", "session_id": session.id}
    )

    assert result.session_id == session.id
    assert result.messages == [{"role": "user", "content": "hello"}]


async def test_session_close_handler_returns_closed_status(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path)
    app._session_manager = SessionManager(store, lambda: _FakeRunner(), EventBus())
    session = await app._session_manager.create("chat")

    result = await app._session_close_handler({"type": "session.close", "session_id": session.id})

    assert result.session_id == session.id
    assert result.status == "closed"
    assert store.read_meta(session.id).status == "closed"


async def test_session_compact_handler_returns_compact_result(tmp_path: Path) -> None:
    app = CoreApp()
    store = SessionStore(tmp_path)
    app._session_manager = SessionManager(
        store,
        lambda: _FakeRunner(),
        EventBus(),
        provider=_FakeCompactProvider(),
    )
    session = await app._session_manager.create("chat")
    store.append_message(session.id, "user", "old message " * 100)

    result = await app._session_compact_handler(
        {
            "type": "session.compact",
            "session_id": session.id,
            "focus": "keep facts",
        }
    )

    assert result.summary_tokens == 10
    assert result.saved_tokens > 0
    assert store.read_messages(session.id)[0]["content"] == (
        "Compacted conversation summary:\n\ncompact summary"
    )
