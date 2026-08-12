from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from my_agent.core.bus.commands import SessionCompactResult
from my_agent.core.bus.envelope import HandlerError
from my_agent.core.bus.events import (
    ContextCompactionFailedEvent,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
    SkillToolCompatibilityEvent,
)
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.base import LLMProvider
from my_agent.core.runner import RunOutcome
from my_agent.core.runs import new_run_id
from my_agent.core.session.model import Session, SessionMode
from my_agent.core.session.store import SessionStore
from my_agent.core.skills.loader import SkillLoader
from my_agent.core.skills.tool_compat import available_tool_names_for_runner, resolve_allowed_tools

SESSION_NOT_FOUND = -32010
SESSION_CLOSED = -32011
SESSION_BUSY = -32012
COMPACTION_PROVIDER_UNAVAILABLE = -32020
COMPACTION_FAILED = -32021


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionRunner(Protocol):
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str,
        session: Session,
        store: SessionStore,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome: ...


class SessionManager:
    def __init__(
        self,
        store: SessionStore,
        runner_factory: Callable[[], SessionRunner],
        bus: EventBus,
        provider: LLMProvider | None = None,
    ) -> None:
        self._store = store
        self._runner_factory = runner_factory
        self._bus = bus
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._provider = provider

    async def create(
        self,
        mode: SessionMode,
        title: str = "",
        *,
        workspace_root: str = "",
    ) -> Session:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        now = _now()
        session = Session(
            id=sid,
            mode=mode,
            status="active",
            title=title,
            created_at=now,
            updated_at=now,
            run_ids=[],
            workspace_root=workspace_root,
        )
        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()
        self._store.write_meta(session)
        await self._bus.publish(SessionCreatedEvent(session_id=sid, mode=mode, ts=now))
        return session

    async def send_message(self, sid: str, content: str, *, run_id: str | None = None) -> str:
        session = self._get_session(sid)
        lock = self._locks[sid]

        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")

        async with lock:
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session closed")

            if session.status == "waiting_for_input":
                await self._bus.publish(SessionResumedEvent(session_id=sid, ts=_now()))

            self._store.append_message(sid, "user", content)
            await self._bus.publish(
                SessionMessageReceivedEvent(session_id=sid, content=content, ts=_now())
            )
            if not session.title:
                session.title = content[:40]
            run_id = run_id or new_run_id()
            session.run_ids.append(run_id)
            session.updated_at = _now()
            self._store.write_meta(session)

            runner = self._runner_factory()

            goal = content
            system_prompt_override: str | None = None
            tool_whitelist: list[str] | None = None

            if content.startswith("//"):
                goal = content[1:]
            elif content.startswith("/"):
                parts = content[1:].split(None, 1)
                skill_name = parts[0]
                arguments = parts[1] if len(parts) > 1 else ""
                loader = SkillLoader(session.workspace_root)
                skill = loader.resolve(skill_name)
                if skill is not None:
                    goal = loader.render_prompt(skill, arguments)
                    system_prompt_override = skill.system_prompt_template
                    # P6: resolve declared allowed_tools against the built-in tool
                    # set. None means the skill did not declare a whitelist and the
                    # run stays unrestricted; [] means the skill declared tools but
                    # none resolved, so the run is locked to zero tools.
                    declared = skill.allowed_tools or None
                    available_tool_names = available_tool_names_for_runner(runner)
                    compat = resolve_allowed_tools(declared, available_tool_names)
                    if compat.unrestricted_by_skill:
                        tool_whitelist = None
                    else:
                        tool_whitelist = list(compat.resolved_tools)
                    await self._bus.publish(
                        SkillInvokedEvent(
                            skill_name=skill.name,
                            arguments=arguments,
                            run_id=run_id,
                            session_id=session.id,
                            ts=_now(),
                        )
                    )
                    if compat.has_diagnostics:
                        await self._bus.publish(
                            SkillToolCompatibilityEvent(
                                skill_name=skill.name,
                                run_id=run_id,
                                session_id=session.id,
                                resolved_tools=list(compat.resolved_tools),
                                aliases=[dict(pair) for pair in compat.aliases],
                                unresolved_tools=list(compat.unresolved_tools),
                                ts=_now(),
                            )
                        )

            await runner.run_and_capture(
                goal,
                run_id=run_id,
                session=session,
                store=self._store,
                system_prompt_override=system_prompt_override,
                tool_whitelist=tool_whitelist,
            )
            session.updated_at = _now()
            if session.mode == "one_shot":
                session.status = "closed"
                await self._bus.publish(
                    SessionClosedEvent(session_id=sid, ts=session.updated_at)
                )
            else:
                session.status = "waiting_for_input"
                await self._bus.publish(
                    SessionWaitingForInputEvent(
                        session_id=sid,
                        last_run_id=run_id,
                        ts=session.updated_at,
                    )
                )

            self._store.write_meta(session)
            return run_id

    async def close(self, sid: str) -> None:
        session = self._get_session(sid)
        lock = self._locks[sid]

        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")

        async with lock:
            session.status = "closed"
            session.updated_at = _now()
            self._store.write_meta(session)
            await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))

    async def compact(self, sid: str, focus: str = "") -> SessionCompactResult:
        self._get_session(sid)
        lock = self._locks[sid]

        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        if self._provider is None:
            raise HandlerError(COMPACTION_PROVIDER_UNAVAILABLE, "compaction provider unavailable")

        async with lock:
            from my_agent.core.compact.compactor import Compactor, _recent_plain_messages
            from my_agent.core.compact.policy import policy_for_task
            from my_agent.core.task.classifier import TaskType

            compact_id = f"compact-{uuid.uuid4().hex[:12]}"
            messages = self._store.read_messages(sid)
            task_type = TaskType.CHAT
            policy = policy_for_task(task_type)
            recent_messages = _recent_plain_messages(messages, policy)
            session_dir = self._store.session_dir(sid)
            compactor = Compactor(
                self._bus,
                session_dir,
                sid,
                store=self._store,
                task_type=task_type,
            )
            result = await compactor.compact_messages(
                messages,
                self._provider,
                focus=f"{policy.summary_focus}\n{focus}".strip(),
            )
            if result is None:
                await self._bus.publish(
                    ContextCompactionFailedEvent(
                        session_id=sid,
                        run_id=compact_id,
                        reason="summary_unavailable",
                        ts=_now(),
                    )
                )
                raise HandlerError(COMPACTION_FAILED, "compaction failed or not beneficial")

            self._store.compact_active_thread(
                sid,
                result.summary_text,
                compact_id,
                recent_messages=recent_messages,
            )
            self._store.append_compaction_record(
                sid,
                {
                    "compact_id": compact_id,
                    "run_id": compact_id,
                    "task_type": task_type.value,
                    "policy": policy.name,
                    "original_tokens": result.original_token_estimate,
                    "summary_tokens": result.summary_tokens,
                    "recent_message_keep": policy.recent_message_keep,
                    "preserved_recent_count": len(recent_messages),
                    "created_at": _now(),
                },
            )

            return SessionCompactResult(
                summary_tokens=result.summary_tokens,
                saved_tokens=max(0, result.original_token_estimate - result.summary_tokens),
            )

    async def get_history(self, sid: str) -> list[dict[str, Any]]:
        self._get_session(sid)
        return self._store.read_messages(sid)

    def _get_session(self, sid: str) -> Session:
        session = self._sessions.get(sid)
        if session is None:
            raise HandlerError(SESSION_NOT_FOUND, "session not found")
        return session
