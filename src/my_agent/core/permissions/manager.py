from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_agent.core.permissions.policy import (
    DEFAULT_POLICIES,
    PermissionDecision,
    ToolPolicy,
    evaluate,
    matches_outside_cwd,
    param_preview,
)
from my_agent.core.permissions.storage import load_policy_file, save_policy_file

logger = logging.getLogger(__name__)


@dataclass
class _PendingRequest:
    future: asyncio.Future[str]
    session_id: str
    tool_name: str


class PermissionManager:
    def __init__(
        self,
        policies: dict[str, ToolPolicy] | None = None,
        *,
        policy_file: Path | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._policies = policies or dict(DEFAULT_POLICIES)
        self._timeout_s = timeout_s
        self._pending: dict[str, _PendingRequest] = {}
        self._session_always: dict[tuple[str, str], str] = {}
        self._policy_file = policy_file
        self._persistent_always = load_policy_file(policy_file) if policy_file is not None else {}

    def evaluate(self, tool_name: str, params: dict[str, Any]) -> PermissionDecision:
        policy = self._policies.get(tool_name)
        return evaluate(tool_name, params, policy)

    async def check_and_wait(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> tuple[bool, str]:
        decision = self.evaluate(tool_name, params)

        command = str(params.get("command", "")) if tool_name == "bash" else ""
        outside_cwd = bool(command and matches_outside_cwd(command))

        if decision == PermissionDecision.DENY:
            return False, "auto_deny"

        if decision == PermissionDecision.ALLOW:
            return True, "auto_allow"

        if not outside_cwd:
            session_key = (session_id, tool_name)
            if session_key in self._session_always:
                cached = self._session_always[session_key]
                return cached == "allow", f"auto_{cached}"
            if tool_name in self._persistent_always:
                cached = self._persistent_always[tool_name]
                return cached == "allow", f"auto_{cached}"

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[tool_use_id] = _PendingRequest(future, session_id, tool_name)
        await event_emitter(
            {
                "type": "permission.requested",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "params": params,
                "param_preview": param_preview(tool_name, params),
                "session_id": session_id,
            }
        )

        try:
            if self._timeout_s > 0:
                raw = await asyncio.wait_for(future, timeout=self._timeout_s)
            else:
                raw = await future
        except TimeoutError:
            self._pending.pop(tool_use_id, None)
            return False, "timeout"
        if raw == "always_allow":
            self._session_always[(session_id, tool_name)] = "allow"
            self._persistent_always[tool_name] = "allow"
            if self._policy_file is not None:
                save_policy_file(self._persistent_always, self._policy_file)
        elif raw == "always_deny":
            self._session_always[(session_id, tool_name)] = "deny"
            self._persistent_always[tool_name] = "deny"
            if self._policy_file is not None:
                save_policy_file(self._persistent_always, self._policy_file)
        return raw in ("allow_once", "always_allow"), raw

    def respond(self, tool_use_id: str, decision: str) -> None:
        request = self._pending.pop(tool_use_id, None)
        if request is None:
            return

        if not request.future.done():
            request.future.set_result(decision)

    def cancel_session(self, session_id: str, reason: str = "session_closed") -> None:
        to_cancel = [
            tool_use_id
            for tool_use_id, request in self._pending.items()
            if request.session_id == session_id
        ]

        for tool_use_id in to_cancel:
            request = self._pending.pop(tool_use_id, None)
            if request is not None and not request.future.done():
                logger.debug(
                    "permission: cancel pending tool_use_id=%s session_id=%s reason=%s",
                    tool_use_id,
                    session_id,
                    reason,
                )
                request.future.set_result("deny_once")
