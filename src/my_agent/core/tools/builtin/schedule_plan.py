from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from my_agent.core.bus.events import (
    SchedulerDiagnosisReportedEvent,
    SchedulerDispatchSkippedEvent,
    SchedulerPlanGeneratedEvent,
)
from my_agent.core.events.bus import EventBus
from my_agent.core.scheduler import SchedulerPlanner
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SchedulePlanTool(BaseTool):
    name = "schedule_plan"
    description = (
        "Generate a read-only scheduler plan for current tasks and sub-agent capacity. "
        "This tool diagnoses ready, blocked, failed, and skipped tasks but does not "
        "spawn agents or update task state."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "run_in_background": {
                "type": "boolean",
                "description": "Whether the plan should reserve background sub-agent capacity.",
            }
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
        bus: EventBus | None = None,
    ) -> None:
        self._task_manager = task_manager
        self._registry = registry
        self._limits = limits
        self._parent_run_id = parent_run_id
        self._root_run_id = root_run_id
        self._session_id = session_id
        self._depth = depth
        self._bus = bus

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        planner = SchedulerPlanner(
            self._task_manager,
            self._registry,
            limits=self._limits,
            parent_run_id=self._parent_run_id,
            root_run_id=self._root_run_id,
            session_id=self._session_id,
            depth=self._depth,
        )
        plan = planner.plan(run_in_background=bool(params.get("run_in_background", True)))
        if self._bus is not None:
            plan_id = uuid.uuid4().hex
            now = _now()
            await self._bus.publish(
                SchedulerPlanGeneratedEvent(
                    run_id=self._parent_run_id,
                    session_id=self._session_id,
                    plan_id=plan_id,
                    parent_run_id=self._parent_run_id,
                    root_run_id=self._root_run_id,
                    ready_task_ids=list(plan.ready_task_ids),
                    dispatchable_task_ids=list(plan.dispatchable_task_ids),
                    skipped_task_ids=[item.task_id for item in plan.skipped],
                    should_replan=plan.should_replan,
                    requires_human_review=plan.requires_human_review,
                    diagnostics_count=len(plan.diagnostics),
                    ts=now,
                )
            )
            if plan.diagnostics or plan.should_replan or plan.requires_human_review:
                await self._bus.publish(
                    SchedulerDiagnosisReportedEvent(
                        run_id=self._parent_run_id,
                        session_id=self._session_id,
                        plan_id=plan_id,
                        diagnostics=list(plan.diagnostics),
                        should_replan=plan.should_replan,
                        requires_human_review=plan.requires_human_review,
                        ts=now,
                    )
                )
            if plan.skipped:
                await self._bus.publish(
                    SchedulerDispatchSkippedEvent(
                        run_id=self._parent_run_id,
                        session_id=self._session_id,
                        plan_id=plan_id,
                        skipped=[item.to_dict() for item in plan.skipped],
                        ts=now,
                    )
                )
        return ToolResult(content=json.dumps(plan.to_dict(), ensure_ascii=False))
