from __future__ import annotations

import json
from datetime import UTC, datetime

from my_agent.core.bus.events import (
    TaskAssignedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStatusChangedEvent,
    TaskUpdatedEvent,
)
from my_agent.core.events.bus import EventBus
from my_agent.core.task.manager import TaskManager
from my_agent.core.task.model import Task, TaskStatus
from my_agent.core.tools.base import BaseTool, ToolResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskUpdateTool(BaseTool):
    name = "task_update"
    description = (
        "Update a task's status or dependency list. "
        "Set status to 'in_progress' when starting work on a task, "
        "'completed' when finished, or 'failed' when work cannot continue. "
        "Returns the updated task as JSON."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "integer",
                "description": "ID of the task to update.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "failed"],
                "description": "New status for the task.",
            },
            "add_blocked_by": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Task IDs to add to blocked_by.",
            },
            "remove_blocked_by": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Task IDs to remove from blocked_by.",
            },
            "assigned_run_id": {
                "type": "string",
                "description": "Run ID currently assigned to this task.",
            },
            "completed_by_run_id": {
                "type": "string",
                "description": "Run ID that completed this task.",
            },
            "failed_by_run_id": {
                "type": "string",
                "description": "Run ID that failed this task.",
            },
            "failure_reason": {
                "type": "string",
                "description": "Short reason explaining why this task failed.",
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
        "required": ["task_id"],
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
        task_id = int(str(params["task_id"]))
        status: TaskStatus | None = params.get("status")  # type: ignore[assignment]
        raw_add: list[object] = list(params.get("add_blocked_by") or [])  # type: ignore[call-overload]
        raw_remove: list[object] = list(params.get("remove_blocked_by") or [])  # type: ignore[call-overload]
        add_blocked = [int(str(x)) for x in raw_add]
        remove_blocked = [int(str(x)) for x in raw_remove]
        assigned_run_id = self._optional_string(params, "assigned_run_id")
        completed_by_run_id = self._optional_string(params, "completed_by_run_id")
        failed_by_run_id = self._optional_string(params, "failed_by_run_id")
        failure_reason = self._optional_string(params, "failure_reason")
        task_type = self._optional_string(params, "task_type")
        priority = self._optional_int(params, "priority")
        risk = self._optional_string(params, "risk")
        suggested_agent_level = self._optional_string(params, "suggested_agent_level")
        required_capabilities = self._optional_string_list(params, "required_capabilities")
        expected_outputs = self._optional_string_list(params, "expected_outputs")
        acceptance_criteria = self._optional_string_list(params, "acceptance_criteria")
        can_parallelize = self._optional_bool(params, "can_parallelize")
        requires_human_review = self._optional_bool(params, "requires_human_review")
        estimated_complexity = self._optional_string(params, "estimated_complexity")

        try:
            previous_task = self._manager.get(task_id)
            task = self._manager.update(
                task_id,
                status=status,
                add_blocked_by=add_blocked or None,
                remove_blocked_by=remove_blocked or None,
                assigned_run_id=assigned_run_id,
                completed_by_run_id=completed_by_run_id,
                failed_by_run_id=failed_by_run_id,
                failure_reason=failure_reason,
                task_type=task_type,
                priority=priority,
                risk=risk,
                suggested_agent_level=suggested_agent_level,
                required_capabilities=required_capabilities,
                expected_outputs=expected_outputs,
                acceptance_criteria=acceptance_criteria,
                can_parallelize=can_parallelize,
                requires_human_review=requires_human_review,
                estimated_complexity=estimated_complexity,
            )

            if self._bus is not None:
                await self._publish_task_events(
                    task,
                    previous_status=previous_task.status,
                    previous_assigned_run_id=previous_task.assigned_run_id,
                )
            return ToolResult(content=json.dumps(task.to_dict(), ensure_ascii=False))
        except ValueError as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")

    async def _publish_task_events(
        self,
        task: Task,
        *,
        previous_status: TaskStatus,
        previous_assigned_run_id: str,
    ) -> None:
        if self._bus is None:
            return

        now = _now()
        await self._bus.publish(
            TaskUpdatedEvent(
                run_id=self._run_id,
                session_id=self._session_id,
                task_id=task.id,
                subject=task.subject,
                status=task.status,
                ts=now,
            )
        )

        status_changed = previous_status != task.status
        if status_changed:
            await self._bus.publish(
                TaskStatusChangedEvent(
                    run_id=self._run_id,
                    session_id=self._session_id,
                    task_id=task.id,
                    subject=task.subject,
                    previous_status=previous_status,
                    status=task.status,
                    ts=now,
                )
            )

        assigned_changed = (
            task.assigned_run_id != previous_assigned_run_id and bool(task.assigned_run_id)
        )
        if task.status == "in_progress" and (status_changed or assigned_changed):
            await self._bus.publish(
                TaskAssignedEvent(
                    run_id=self._run_id,
                    session_id=self._session_id,
                    task_id=task.id,
                    subject=task.subject,
                    status=task.status,
                    assigned_run_id=task.assigned_run_id,
                    ts=now,
                )
            )

        if task.status == "completed" and status_changed:
            await self._bus.publish(
                TaskCompletedEvent(
                    run_id=self._run_id,
                    session_id=self._session_id,
                    task_id=task.id,
                    subject=task.subject,
                    status=task.status,
                    completed_by_run_id=task.completed_by_run_id,
                    assigned_run_id=task.assigned_run_id,
                    ts=now,
                )
            )

        if task.status == "failed" and status_changed:
            await self._bus.publish(
                TaskFailedEvent(
                    run_id=self._run_id,
                    session_id=self._session_id,
                    task_id=task.id,
                    subject=task.subject,
                    status=task.status,
                    failed_by_run_id=task.failed_by_run_id,
                    assigned_run_id=task.assigned_run_id,
                    failure_reason=task.failure_reason,
                    ts=now,
                )
            )

    def _optional_string(self, params: dict[str, object], key: str) -> str | None:
        if key not in params:
            return None
        return str(params[key])

    def _optional_int(self, params: dict[str, object], key: str) -> int | None:
        if key not in params:
            return None
        return int(str(params[key]))

    def _optional_bool(self, params: dict[str, object], key: str) -> bool | None:
        if key not in params:
            return None
        return bool(params[key])

    def _optional_string_list(self, params: dict[str, object], key: str) -> list[str] | None:
        if key not in params:
            return None
        raw_values: list[object] = list(params[key] or [])  # type: ignore[call-overload]
        return [str(item) for item in raw_values]
