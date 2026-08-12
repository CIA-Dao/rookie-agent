from __future__ import annotations

import json
from typing import Any

from my_agent.core.delegation import DelegationPolicy
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult


class DelegationPolicyTool(BaseTool):
    name = "delegation_policy"
    description = (
        "Return a read-only delegation policy recommendation for direct execution, "
        "task graph creation, explicit orchestration, or manual review. Automatic "
        "dispatch is disabled by default."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "Optional current user goal text to classify when no task graph exists."
                ),
            },
            "allow_auto_dispatch": {
                "type": "boolean",
                "description": (
                    "Opt-in flag for checking whether auto-dispatch would be safe. "
                    "Defaults to false and does not dispatch by itself."
                ),
            },
        },
    }

    def __init__(
        self,
        task_manager: TaskManager,
        registry: BackgroundTaskRegistry,
        *,
        limits: SubagentLimits | None = None,
        parent_run_id: str = "",
        root_run_id: str = "",
        session_id: str = "",
        depth: int = 0,
    ) -> None:
        self._policy = DelegationPolicy(
            task_manager,
            registry,
            limits=limits,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id,
            session_id=session_id,
            depth=depth,
        )

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        payload = self._policy.evaluate(
            goal=str(params.get("goal", "")),
            allow_auto_dispatch=bool(params.get("allow_auto_dispatch", False)),
        )
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))
