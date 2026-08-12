from __future__ import annotations

import json
import re
from typing import Any

from my_agent.core.events.bus import EventBus
from my_agent.core.scheduler import SchedulerPlanner
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.builtin.task_update import TaskUpdateTool

_RUN_ID_RE = re.compile(r"run_id=([^\s]+)")


class DispatchPlanTool(BaseTool):
    name = "dispatch_plan"
    description = (
        "Explicitly dispatch current SchedulerPlan dispatch envelopes to background "
        "sub-agents. This starts sub-agents and assigns task run ownership, but does "
        "not collect results or mark tasks completed."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional task IDs to dispatch. Defaults to all dispatchable tasks.",
            },
            "max_tasks": {
                "type": "integer",
                "description": "Optional maximum number of dispatchable tasks to start.",
            },
            "run_in_background": {
                "type": "boolean",
                "description": (
                    "Reserved for consistency with schedule_plan; dispatch_plan "
                    "uses background sub-agents."
                ),
            },
        },
    }

    def __init__(
        self,
        task_manager: TaskManager,
        registry: BackgroundTaskRegistry,
        spawn_tool: BaseTool,
        *,
        limits: SubagentLimits | None = None,
        parent_run_id: str = "",
        root_run_id: str = "",
        session_id: str = "",
        depth: int = 0,
        bus: EventBus | None = None,
    ) -> None:
        self._task_manager = task_manager
        self._registry = registry
        self._spawn_tool = spawn_tool
        self._limits = limits
        self._parent_run_id = parent_run_id
        self._root_run_id = root_run_id
        self._session_id = session_id
        self._depth = depth
        self._bus = bus

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        selected_ids = self._selected_task_ids(params)
        max_tasks = self._max_tasks(params)
        planner = SchedulerPlanner(
            self._task_manager,
            self._registry,
            limits=self._limits,
            parent_run_id=self._parent_run_id,
            root_run_id=self._root_run_id,
            session_id=self._session_id,
            depth=self._depth,
        )
        plan = planner.plan(run_in_background=True)

        dispatched: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []
        envelopes = plan.dispatch_envelopes
        if selected_ids is not None:
            envelopes = [item for item in envelopes if item.task_id in selected_ids]
        if max_tasks is not None:
            envelopes = envelopes[:max_tasks]

        update_tool = TaskUpdateTool(
            self._task_manager,
            bus=self._bus,
            run_id=self._parent_run_id,
            session_id=self._session_id,
        )

        for envelope in envelopes:
            spawn_result = await self._spawn_tool.invoke(
                {
                    "description": f"Task #{envelope.task_id}: {envelope.subject}",
                    "prompt": envelope.prompt,
                    "run_in_background": True,
                    "subagent_type": "",
                }
            )
            if spawn_result.is_error:
                errors.append(
                    {
                        "task_id": envelope.task_id,
                        "error_type": spawn_result.error_type,
                        "message": spawn_result.content,
                    }
                )
                continue

            child_run_id = self._extract_run_id(spawn_result.content)
            if not child_run_id:
                errors.append(
                    {
                        "task_id": envelope.task_id,
                        "error_type": "runtime_error",
                        "message": f"Could not parse child run_id from: {spawn_result.content}",
                    }
                )
                continue

            update_result = await update_tool.invoke(
                {
                    "task_id": envelope.task_id,
                    "status": "in_progress",
                    "assigned_run_id": child_run_id,
                    "completed_by_run_id": "",
                    "failed_by_run_id": "",
                    "failure_reason": "",
                }
            )
            if update_result.is_error:
                errors.append(
                    {
                        "task_id": envelope.task_id,
                        "run_id": child_run_id,
                        "error_type": update_result.error_type,
                        "message": update_result.content,
                    }
                )
                continue

            dispatched.append(
                {
                    "task_id": envelope.task_id,
                    "run_id": child_run_id,
                    "recommended_agent_level": envelope.recommended_agent_level,
                }
            )

        skipped_task_ids = [
            task_id
            for task_id in plan.ready_task_ids
            if task_id not in {item["task_id"] for item in dispatched}
        ]
        payload = {
            "dispatched": dispatched,
            "errors": errors,
            "skipped_task_ids": skipped_task_ids,
            "plan": plan.to_dict(),
        }
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False),
            is_error=bool(errors) and not dispatched,
            error_type="runtime_error" if errors and not dispatched else None,
        )

    def _selected_task_ids(self, params: dict[str, object]) -> set[int] | None:
        if "task_ids" not in params:
            return None
        raw_values: list[object] = list(params.get("task_ids") or [])  # type: ignore[call-overload]
        return {int(str(value)) for value in raw_values}

    def _max_tasks(self, params: dict[str, object]) -> int | None:
        if "max_tasks" not in params:
            return None
        return max(0, int(str(params["max_tasks"])))

    def _extract_run_id(self, content: str) -> str:
        match = _RUN_ID_RE.search(content)
        return match.group(1) if match else ""
