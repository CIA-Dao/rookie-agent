from __future__ import annotations

import json
from datetime import UTC, datetime

from my_agent.core.bus.events import TaskCreatedEvent
from my_agent.core.events.bus import EventBus
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskCreateTool(BaseTool):
    name = "task_create"
    description = (
        "Create a new task to track a unit of work. "
        "Use this to break down a complex goal into smaller, trackable steps. "
        "Returns the created task as JSON."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Short title for the task.",
            },
            "description": {
                "type": "string",
                "description": "Optional longer description of what needs to be done.",
            },
            "blocked_by": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "IDs of tasks that must be completed before this one.",
            },
            "task_type": {
                "type": "string",
                "description": "Kind of work, such as planning, implementation, test, or review.",
            },
            "priority": {
                "type": "integer",
                "description": "Scheduling priority. Higher values should be considered first.",
            },
            "risk": {
                "type": "string",
                "description": "Risk level for this task.",
            },
            "suggested_agent_level": {
                "type": "string",
                "description": "Planner suggestion for root, child, or grandchild handling.",
            },
            "required_capabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Capabilities or tool families needed for this task.",
            },
            "expected_outputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete outputs expected from this task.",
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Criteria used to accept this task as complete.",
            },
            "can_parallelize": {
                "type": "boolean",
                "description": "Whether this task can be safely run in parallel.",
            },
            "requires_human_review": {
                "type": "boolean",
                "description": "Whether human review is required before accepting this task.",
            },
            "estimated_complexity": {
                "type": "string",
                "description": "Rough complexity estimate for scheduler planning.",
            },
        },
        "required": ["subject"],
    }

    def __init__(
        self,
        task_manager: TaskManager,
        *,
        bus: EventBus | None = None,
        run_id: str = "",
        session_id: str = "",
    ) -> None:
        self._manager = task_manager
        self._bus = bus
        self._run_id = run_id
        self._session_id = session_id

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        subject = str(params["subject"])
        description = str(params.get("description", ""))
        raw_blocked: list[object] = list(params.get("blocked_by") or [])  # type: ignore[call-overload]
        blocked_by = [int(str(x)) for x in raw_blocked]
        raw_capabilities: list[object] = list(params.get("required_capabilities") or [])  # type: ignore[call-overload]
        raw_outputs: list[object] = list(params.get("expected_outputs") or [])  # type: ignore[call-overload]
        raw_acceptance: list[object] = list(params.get("acceptance_criteria") or [])  # type: ignore[call-overload]

        try:
            task = self._manager.create(
                subject,
                description,
                blocked_by,
                task_type=str(params.get("task_type", "general")),
                priority=int(str(params.get("priority", 0))),
                risk=str(params.get("risk", "medium")),
                suggested_agent_level=str(params.get("suggested_agent_level", "")),
                required_capabilities=[str(item) for item in raw_capabilities],
                expected_outputs=[str(item) for item in raw_outputs],
                acceptance_criteria=[str(item) for item in raw_acceptance],
                can_parallelize=bool(params.get("can_parallelize", False)),
                requires_human_review=bool(params.get("requires_human_review", False)),
                estimated_complexity=str(params.get("estimated_complexity", "")),
            )
            if self._bus is not None:
                await self._bus.publish(
                    TaskCreatedEvent(
                        run_id=self._run_id,
                        session_id=self._session_id,
                        task_id=task.id,
                        subject=task.subject,
                        status=task.status,
                        ts=_now(),
                    )
                )
            return ToolResult(content=json.dumps(task.to_dict(), ensure_ascii=False))
        except ValueError as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")
