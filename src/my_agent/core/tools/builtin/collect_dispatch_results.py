from __future__ import annotations

import json
from typing import Any

from my_agent.core.events.bus import EventBus
from my_agent.core.subagent.registry import BackgroundSubagentRecord, BackgroundTaskRegistry
from my_agent.core.task.manager import TaskManager
from my_agent.core.task.model import Task
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.builtin.task_update import TaskUpdateTool


class CollectDispatchResultsTool(BaseTool):
    name = "collect_dispatch_results"
    description = (
        "Collect finished background sub-agent results for dispatched tasks and write "
        "task completion or failure state back. Running tasks are left unchanged."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional in-progress task IDs to collect.",
            },
        },
    }

    def __init__(
        self,
        task_manager: TaskManager,
        registry: BackgroundTaskRegistry,
        *,
        bus: EventBus | None = None,
        run_id: str = "",
        session_id: str = "",
    ) -> None:
        self._task_manager = task_manager
        self._registry = registry
        self._bus = bus
        self._run_id = run_id
        self._session_id = session_id

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        selected_ids = self._selected_task_ids(params)
        update_tool = TaskUpdateTool(
            self._task_manager,
            bus=self._bus,
            run_id=self._run_id,
            session_id=self._session_id,
        )
        completed: list[dict[str, object]] = []
        failed: list[dict[str, object]] = []
        running: list[dict[str, object]] = []
        missing: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []

        for task in self._candidate_tasks(selected_ids):
            if task.status != "in_progress" or not task.assigned_run_id:
                skipped.append(
                    {
                        "task_id": task.id,
                        "status": task.status,
                        "reason": "not_assigned_in_progress",
                    }
                )
                continue

            record = self._registry.get_record(task.assigned_run_id)
            if record is None or record.task is None:
                missing.append(
                    {
                        "task_id": task.id,
                        "run_id": task.assigned_run_id,
                        "reason": "missing_background_subagent",
                    }
                )
                continue

            if record.status == "running" and not record.task.done():
                running.append({"task_id": task.id, "run_id": task.assigned_run_id})
                continue

            if record.status == "success" or record.context.status == "success":
                update_result = await update_tool.invoke(
                    {
                        "task_id": task.id,
                        "status": "completed",
                        "assigned_run_id": task.assigned_run_id,
                        "completed_by_run_id": task.assigned_run_id,
                        "failed_by_run_id": "",
                        "failure_reason": "",
                    }
                )
                if update_result.is_error:
                    failed.append(
                        {
                            "task_id": task.id,
                            "run_id": task.assigned_run_id,
                            "reason": update_result.content,
                        }
                    )
                else:
                    completed.append(
                        {
                            "task_id": task.id,
                            "run_id": task.assigned_run_id,
                            "result": _final_answer(record),
                        }
                    )
                continue

            reason = _failure_reason(record)
            update_result = await update_tool.invoke(
                {
                    "task_id": task.id,
                    "status": "failed",
                    "assigned_run_id": task.assigned_run_id,
                    "completed_by_run_id": "",
                    "failed_by_run_id": task.assigned_run_id,
                    "failure_reason": reason,
                }
            )
            failed.append(
                {
                    "task_id": task.id,
                    "run_id": task.assigned_run_id,
                    "reason": update_result.content if update_result.is_error else reason,
                }
            )

        payload = {
            "completed": completed,
            "failed": failed,
            "running": running,
            "missing": missing,
            "skipped": skipped,
        }
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))

    def _candidate_tasks(self, selected_ids: set[int] | None) -> list[Task]:
        tasks = self._task_manager.list_all()
        if selected_ids is None:
            return [task for task in tasks if task.status == "in_progress"]
        return [task for task in tasks if task.id in selected_ids]

    def _selected_task_ids(self, params: dict[str, object]) -> set[int] | None:
        if "task_ids" not in params:
            return None
        raw_values: list[object] = list(params.get("task_ids") or [])  # type: ignore[call-overload]
        return {int(str(value)) for value in raw_values}


def _final_answer(record: BackgroundSubagentRecord) -> str:
    for message in reversed(record.context.messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _failure_reason(record: BackgroundSubagentRecord) -> str:
    if record.status == "timed_out":
        return "timed_out"
    if record.status == "cancelled":
        return "cancelled"
    if record.reason:
        return record.reason
    if record.context.reason:
        return record.context.reason
    return record.context.status or record.status
