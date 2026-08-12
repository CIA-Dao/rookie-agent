from __future__ import annotations

import json
from typing import Any

from my_agent.core.scheduler import SchedulerPlanner
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager
from my_agent.core.task.model import Task
from my_agent.core.tools.base import BaseTool, ToolResult


class OrchestrationSummaryTool(BaseTool):
    name = "orchestration_summary"
    description = (
        "Return a read-only orchestration summary with task counts, failure "
        "recovery guidance, missing background records, scheduler diagnostics, "
        "and final synthesis guidance."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "include_tasks": {
                "type": "boolean",
                "description": "Whether to include per-task details. Defaults to true.",
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
        self._task_manager = task_manager
        self._registry = registry
        self._limits = limits
        self._parent_run_id = parent_run_id
        self._root_run_id = root_run_id
        self._session_id = session_id
        self._depth = depth

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        include_tasks = bool(params.get("include_tasks", True))
        tasks = self._task_manager.list_all()
        plan = SchedulerPlanner(
            self._task_manager,
            self._registry,
            limits=self._limits,
            parent_run_id=self._parent_run_id,
            root_run_id=self._root_run_id,
            session_id=self._session_id,
            depth=self._depth,
        ).plan(run_in_background=True)

        payload: dict[str, object] = {
            "counts": _counts(tasks),
            "scheduler": {
                "ready_task_ids": list(plan.ready_task_ids),
                "dispatchable_task_ids": list(plan.dispatchable_task_ids),
                "blocked_task_ids": list(plan.blocked_task_ids),
                "in_progress_task_ids": list(plan.in_progress_task_ids),
                "failed_task_ids": list(plan.failed_task_ids),
                "should_replan": bool(plan.should_replan),
                "requires_human_review": bool(plan.requires_human_review),
                "diagnostics": list(plan.diagnostics),
                "skipped": [item.to_dict() for item in plan.skipped],
            },
            "recovery": {
                "failed_tasks": _failed_tasks(tasks),
                "missing_background_records": _missing_background_records(
                    tasks,
                    self._registry,
                ),
                "next_actions": _next_actions(tasks, plan),
            },
            "final_synthesis_prompt": _final_synthesis_prompt(tasks),
        }
        if include_tasks:
            payload["tasks"] = [_task_summary(task) for task in tasks]
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))


def _counts(tasks: list[Task]) -> dict[str, int]:
    counts = {"total": len(tasks), "pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
    for task in tasks:
        counts[task.status] += 1
    counts["blocked"] = len(
        [task for task in tasks if task.status == "pending" and task.blocked_by]
    )
    counts["ready"] = len(
        [task for task in tasks if task.status == "pending" and not task.blocked_by]
    )
    return counts


def _task_summary(task: Task) -> dict[str, object]:
    return {
        "id": task.id,
        "subject": task.subject,
        "status": task.status,
        "blocked_by": list(task.blocked_by),
        "assigned_run_id": task.assigned_run_id,
        "completed_by_run_id": task.completed_by_run_id,
        "failed_by_run_id": task.failed_by_run_id,
        "failure_reason": task.failure_reason,
        "risk": task.risk,
        "task_type": task.task_type,
        "priority": task.priority,
    }


def _failed_tasks(tasks: list[Task]) -> list[dict[str, object]]:
    failed: list[dict[str, object]] = []
    for task in tasks:
        if task.status != "failed":
            continue
        failed.append(
            {
                "task_id": task.id,
                "subject": task.subject,
                "assigned_run_id": task.assigned_run_id,
                "failed_by_run_id": task.failed_by_run_id,
                "failure_reason": task.failure_reason,
                "retry_guidance": (
                    "Review the failure reason, decide whether the task needs human "
                    "changes or can be reset to pending, then retry explicitly."
                ),
            }
        )
    return failed


def _missing_background_records(
    tasks: list[Task],
    registry: BackgroundTaskRegistry,
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for task in tasks:
        if task.status != "in_progress" or not task.assigned_run_id:
            continue
        if registry.get_record(task.assigned_run_id) is None:
            missing.append(
                {
                    "task_id": task.id,
                    "subject": task.subject,
                    "assigned_run_id": task.assigned_run_id,
                    "guidance": (
                        "The task is in_progress but the assigned background run is not "
                        "present in the current in-memory registry. Review manually or "
                        "reset/retry explicitly."
                    ),
                }
            )
    return missing


def _next_actions(tasks: list[Task], plan: object) -> list[str]:
    counts = _counts(tasks)
    actions: list[str] = []
    should_replan = bool(getattr(plan, "should_replan", False))
    requires_human_review = bool(getattr(plan, "requires_human_review", False))
    dispatchable_task_ids = list(getattr(plan, "dispatchable_task_ids", []))
    in_progress_task_ids = list(getattr(plan, "in_progress_task_ids", []))

    if not tasks:
        actions.append("create_tasks")
    if should_replan:
        actions.append("replan_task_graph")
    if requires_human_review or counts["failed"]:
        actions.append("review_failed_tasks")
    if dispatchable_task_ids:
        actions.append("dispatch_ready_tasks")
    if in_progress_task_ids:
        actions.append("collect_or_wait_for_running_tasks")
    if counts["completed"] and counts["completed"] + counts["failed"] == counts["total"]:
        actions.append("final_synthesis")
    if not actions:
        actions.append("review_state")
    return actions


def _final_synthesis_prompt(tasks: list[Task]) -> str:
    counts = _counts(tasks)
    return (
        "Summarize orchestration outcome for the user. Include completed task count "
        f"({counts['completed']}/{counts['total']}), failed task count ({counts['failed']}), "
        "remaining running or blocked work, files or checks reported by child agents, "
        "and explicit follow-up decisions needed."
    )
