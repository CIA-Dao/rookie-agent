from __future__ import annotations

import json
from typing import Any

from my_agent.core.events.bus import EventBus
from my_agent.core.scheduler import SchedulerPlan, SchedulerPlanner
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.builtin.collect_dispatch_results import CollectDispatchResultsTool
from my_agent.core.tools.builtin.dispatch_plan import DispatchPlanTool


class OrchestrateTasksTool(BaseTool):
    name = "orchestrate_tasks"
    description = (
        "Run one explicit bounded orchestration tick: collect finished dispatched "
        "task results, generate a fresh scheduler plan, and dispatch a limited "
        "number of safe ready tasks. This tool does not wait, poll, loop, or create tasks."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional task IDs to collect or dispatch. Defaults to all relevant tasks."
                ),
            },
            "max_tasks": {
                "type": "integer",
                "description": (
                    "Maximum number of dispatchable tasks to start in this tick. "
                    "Defaults to 1."
                ),
            },
        },
    }

    def __init__(
        self,
        task_manager: TaskManager,
        registry: BackgroundTaskRegistry,
        dispatch_tool: DispatchPlanTool,
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
        self._dispatch_tool = dispatch_tool
        self._limits = limits
        self._parent_run_id = parent_run_id
        self._root_run_id = root_run_id
        self._session_id = session_id
        self._depth = depth
        self._bus = bus

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        task_ids = _task_ids(params)
        max_tasks = _max_tasks(params)

        collect_tool = CollectDispatchResultsTool(
            self._task_manager,
            self._registry,
            bus=self._bus,
            run_id=self._parent_run_id,
            session_id=self._session_id,
        )
        collection_result = await collect_tool.invoke(_filter_params(task_ids))
        collection = json.loads(collection_result.content)

        plan = self._plan()
        plan_summary = _plan_summary(plan)

        dispatch: dict[str, object] = {
            "dispatched": [],
            "errors": [],
            "skipped_task_ids": list(plan.dispatchable_task_ids),
            "blocked_by": "",
        }
        if plan.should_replan:
            dispatch["blocked_by"] = "replan"
        elif plan.requires_human_review or _has_review_skips(plan):
            dispatch["blocked_by"] = "human_review"
        elif plan.dispatchable_task_ids:
            dispatch_params = _filter_params(task_ids)
            dispatch_params["max_tasks"] = max_tasks
            dispatch_result = await self._dispatch_tool.invoke(dispatch_params)
            dispatch = json.loads(dispatch_result.content)

        payload = {
            "collection": collection,
            "plan_summary": plan_summary,
            "dispatch": dispatch,
            "diagnostics": list(plan.diagnostics),
            "next_action": _next_action(collection, plan_summary, dispatch),
        }
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))

    def _plan(self) -> SchedulerPlan:
        planner = SchedulerPlanner(
            self._task_manager,
            self._registry,
            limits=self._limits,
            parent_run_id=self._parent_run_id,
            root_run_id=self._root_run_id,
            session_id=self._session_id,
            depth=self._depth,
        )
        return planner.plan(run_in_background=True)


def _task_ids(params: dict[str, object]) -> list[int] | None:
    if "task_ids" not in params:
        return None
    raw_values: list[object] = list(params.get("task_ids") or [])  # type: ignore[call-overload]
    return [int(str(value)) for value in raw_values]


def _max_tasks(params: dict[str, object]) -> int:
    if "max_tasks" not in params:
        return 1
    return max(0, int(str(params["max_tasks"])))


def _filter_params(task_ids: list[int] | None) -> dict[str, object]:
    if task_ids is None:
        return {}
    return {"task_ids": list(task_ids)}


def _plan_summary(plan: SchedulerPlan) -> dict[str, object]:
    return {
        "ready_task_ids": list(plan.ready_task_ids),
        "dispatchable_task_ids": list(plan.dispatchable_task_ids),
        "blocked_task_ids": list(plan.blocked_task_ids),
        "in_progress_task_ids": list(plan.in_progress_task_ids),
        "failed_task_ids": list(plan.failed_task_ids),
        "should_replan": bool(plan.should_replan),
        "requires_human_review": bool(plan.requires_human_review),
        "skipped_task_ids": [item.task_id for item in plan.skipped],
    }


def _next_action(
    collection: dict[str, object],
    plan_summary: dict[str, object],
    dispatch: dict[str, object],
) -> str:
    if plan_summary["should_replan"]:
        return "replan"
    if plan_summary["requires_human_review"] or dispatch.get("blocked_by") == "human_review":
        return "human_review"
    errors = dispatch.get("errors", [])
    dispatched = dispatch.get("dispatched", [])
    if errors and not dispatched:
        return "retry_or_review"
    if dispatched or collection.get("running") or plan_summary["in_progress_task_ids"]:
        return "continue"
    if plan_summary["dispatchable_task_ids"]:
        return "continue"
    return "idle"


def _has_review_skips(plan: SchedulerPlan) -> bool:
    return any(
        item.reason in {"requires_human_review", "high_risk_root_review"}
        for item in plan.skipped
    )
