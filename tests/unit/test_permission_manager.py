from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from my_agent.core.permissions.manager import PermissionManager
from my_agent.core.permissions.policy import PermissionDecision


def test_evaluate_delegates_to_policy() -> None:
    manager = PermissionManager()

    assert manager.evaluate("read_file", {"path": "README.md"}) == PermissionDecision.ALLOW
    assert manager.evaluate("bash", {"command": "echo hi"}) == PermissionDecision.ASK
    assert manager.evaluate("write_file", {"path": "out.txt"}) == PermissionDecision.ASK


async def test_check_and_wait_allow_returns_without_event() -> None:
    manager = PermissionManager()
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)

    allowed, decision = await manager.check_and_wait(
        tool_use_id="tc-1",
        tool_name="read_file",
        params={"path": "README.md"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert allowed is True
    assert decision == "auto_allow"
    assert emitted == []


async def test_check_and_wait_ask_emits_event_and_waits_for_response() -> None:
    manager = PermissionManager()
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        manager.respond("tc-2", "allow_once")

    allowed, decision = await manager.check_and_wait(
        tool_use_id="tc-2",
        tool_name="bash",
        params={"command": "echo hi"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert allowed is True
    assert decision == "allow_once"
    assert emitted == [
        {
            "type": "permission.requested",
            "tool_use_id": "tc-2",
            "tool_name": "bash",
            "params": {"command": "echo hi"},
            "param_preview": "command='echo hi'",
            "session_id": "sess-1",
        }
    ]


async def test_check_and_wait_deny_once_returns_false() -> None:
    manager = PermissionManager()

    async def emit(event: dict[str, Any]) -> None:
        manager.respond(str(event["tool_use_id"]), "deny_once")

    allowed, decision = await manager.check_and_wait(
        tool_use_id="tc-3",
        tool_name="bash",
        params={"command": "echo hi"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert allowed is False
    assert decision == "deny_once"


async def test_permission_timeout_returns_false() -> None:
    manager = PermissionManager(timeout_s=0.01)
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)

    allowed, decision = await manager.check_and_wait(
        tool_use_id="tc-timeout",
        tool_name="bash",
        params={"command": "echo hi"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert allowed is False
    assert decision == "timeout"
    assert len(emitted) == 1
    assert "tc-timeout" not in manager._pending


async def test_always_allow_skips_future_ask() -> None:
    manager = PermissionManager()
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        manager.respond(str(event["tool_use_id"]), "always_allow")

    first_allowed, first_decision = await manager.check_and_wait(
        tool_use_id="tc-allow-1",
        tool_name="bash",
        params={"command": "echo first"},
        session_id="sess-1",
        event_emitter=emit,
    )
    second_allowed, second_decision = await manager.check_and_wait(
        tool_use_id="tc-allow-2",
        tool_name="bash",
        params={"command": "echo second"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert first_allowed is True
    assert first_decision == "always_allow"
    assert second_allowed is True
    assert second_decision == "auto_allow"
    assert len(emitted) == 1


async def test_always_allow_does_not_bypass_outside_cwd_check() -> None:
    manager = PermissionManager()
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        if event["tool_use_id"] == "tc-allow-1":
            manager.respond(str(event["tool_use_id"]), "always_allow")
        else:
            manager.respond(str(event["tool_use_id"]), "deny_once")

    first_allowed, first_decision = await manager.check_and_wait(
        tool_use_id="tc-allow-1",
        tool_name="bash",
        params={"command": "echo first"},
        session_id="sess-1",
        event_emitter=emit,
    )
    second_allowed, second_decision = await manager.check_and_wait(
        tool_use_id="tc-allow-2",
        tool_name="bash",
        params={"command": "cat /etc/hosts"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert first_allowed is True
    assert first_decision == "always_allow"
    assert second_allowed is False
    assert second_decision == "deny_once"
    assert len(emitted) == 2
    assert emitted[1]["params"] == {"command": "cat /etc/hosts"}
    assert emitted[1]["param_preview"] == "command='cat /etc/hosts'"


async def test_always_deny_skips_future_ask() -> None:
    manager = PermissionManager()
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        manager.respond(str(event["tool_use_id"]), "always_deny")

    first_allowed, first_decision = await manager.check_and_wait(
        tool_use_id="tc-deny-1",
        tool_name="bash",
        params={"command": "echo first"},
        session_id="sess-1",
        event_emitter=emit,
    )
    second_allowed, second_decision = await manager.check_and_wait(
        tool_use_id="tc-deny-2",
        tool_name="bash",
        params={"command": "echo second"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert first_allowed is False
    assert first_decision == "always_deny"
    assert second_allowed is False
    assert second_decision == "auto_deny"
    assert len(emitted) == 1


async def test_always_allow_persists_to_policy_file(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.toml"
    manager = PermissionManager(policy_file=policy_file)
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        manager.respond(str(event["tool_use_id"]), "always_allow")

    allowed, decision = await manager.check_and_wait(
        tool_use_id="tc-persist-1",
        tool_name="bash",
        params={"command": "echo persisted"},
        session_id="sess-1",
        event_emitter=emit,
    )

    assert allowed is True
    assert decision == "always_allow"
    assert policy_file.exists()

    restarted = PermissionManager(policy_file=policy_file)
    emitted_after_restart: list[dict[str, Any]] = []

    async def emit_after_restart(event: dict[str, Any]) -> None:
        emitted_after_restart.append(event)

    allowed_after_restart, decision_after_restart = await restarted.check_and_wait(
        tool_use_id="tc-persist-2",
        tool_name="bash",
        params={"command": "echo loaded"},
        session_id="sess-2",
        event_emitter=emit_after_restart,
    )

    assert allowed_after_restart is True
    assert decision_after_restart == "auto_allow"
    assert emitted_after_restart == []


async def test_persistent_allow_does_not_bypass_outside_cwd_check(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.toml"
    manager = PermissionManager(policy_file=policy_file)

    async def allow_always(event: dict[str, Any]) -> None:
        manager.respond(str(event["tool_use_id"]), "always_allow")

    await manager.check_and_wait(
        tool_use_id="tc-persist-safe",
        tool_name="bash",
        params={"command": "echo safe"},
        session_id="sess-1",
        event_emitter=allow_always,
    )

    restarted = PermissionManager(policy_file=policy_file)
    emitted: list[dict[str, Any]] = []

    async def deny_once(event: dict[str, Any]) -> None:
        emitted.append(event)
        restarted.respond(str(event["tool_use_id"]), "deny_once")

    allowed, decision = await restarted.check_and_wait(
        tool_use_id="tc-persist-outside",
        tool_name="bash",
        params={"command": "cat /etc/hosts"},
        session_id="sess-2",
        event_emitter=deny_once,
    )

    assert allowed is False
    assert decision == "deny_once"
    assert len(emitted) == 1


async def test_cancel_session_resolves_pending_permission() -> None:
    manager = PermissionManager(timeout_s=0)
    emitted = asyncio.Event()

    async def emit(_event: dict[str, Any]) -> None:
        emitted.set()

    task = asyncio.create_task(
        manager.check_and_wait(
            tool_use_id="tc-cancel-1",
            tool_name="bash",
            params={"command": "echo pending"},
            session_id="sess-1",
            event_emitter=emit,
        )
    )

    await emitted.wait()
    manager.cancel_session("sess-1", reason="client_disconnected")

    allowed, decision = await task

    assert allowed is False
    assert decision == "deny_once"
    assert "tc-cancel-1" not in manager._pending


async def test_cancel_session_only_resolves_matching_session() -> None:
    manager = PermissionManager(timeout_s=0)
    emitted_count = 0
    both_emitted = asyncio.Event()

    async def emit(_event: dict[str, Any]) -> None:
        nonlocal emitted_count
        emitted_count += 1
        if emitted_count == 2:
            both_emitted.set()

    task_1 = asyncio.create_task(
        manager.check_and_wait(
            tool_use_id="tc-cancel-s1",
            tool_name="bash",
            params={"command": "echo one"},
            session_id="sess-1",
            event_emitter=emit,
        )
    )
    task_2 = asyncio.create_task(
        manager.check_and_wait(
            tool_use_id="tc-cancel-s2",
            tool_name="bash",
            params={"command": "echo two"},
            session_id="sess-2",
            event_emitter=emit,
        )
    )

    await both_emitted.wait()
    manager.cancel_session("sess-2", reason="client_disconnected")

    allowed_2, decision_2 = await task_2
    assert allowed_2 is False
    assert decision_2 == "deny_once"

    manager.respond("tc-cancel-s1", "allow_once")
    allowed_1, decision_1 = await task_1

    assert allowed_1 is True
    assert decision_1 == "allow_once"
