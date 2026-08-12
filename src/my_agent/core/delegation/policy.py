from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from my_agent.core.scheduler import SchedulerPlan, SchedulerPlanner
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager

DelegationDecision = Literal[
    "direct",
    "create_task_graph",
    "explicit_orchestration",
    "manual_review",
]


@dataclass(frozen=True)
class DelegationPolicy:
    task_manager: TaskManager
    registry: BackgroundTaskRegistry
    limits: SubagentLimits | None = None
    parent_run_id: str = ""
    root_run_id: str = ""
    session_id: str = ""
    depth: int = 0

    def evaluate(
        self,
        *,
        goal: str = "",
        allow_auto_dispatch: bool = False,
    ) -> dict[str, object]:
        plan = self._plan()
        task_count = len(self.task_manager.list_all())
        decision, reasons, recommended_tools = self._decision(
            plan,
            goal=goal,
            task_count=task_count,
        )
        safe_to_auto_dispatch = bool(
            allow_auto_dispatch
            and decision == "explicit_orchestration"
            and plan.dispatchable_task_ids
            and not plan.should_replan
            and not plan.requires_human_review
        )

        return {
            "decision": decision,
            "auto_delegation_enabled": allow_auto_dispatch,
            "safe_to_auto_dispatch": safe_to_auto_dispatch,
            "reasons": reasons,
            "recommended_tools": recommended_tools,
            "task_count": task_count,
            "scheduler": {
                "ready_task_ids": list(plan.ready_task_ids),
                "dispatchable_task_ids": list(plan.dispatchable_task_ids),
                "blocked_task_ids": list(plan.blocked_task_ids),
                "in_progress_task_ids": list(plan.in_progress_task_ids),
                "failed_task_ids": list(plan.failed_task_ids),
                "should_replan": plan.should_replan,
                "requires_human_review": plan.requires_human_review,
                "diagnostics": list(plan.diagnostics),
                "capacity": plan.capacity.to_dict(),
            },
        }

    def _plan(self) -> SchedulerPlan:
        return SchedulerPlanner(
            self.task_manager,
            self.registry,
            limits=self.limits,
            parent_run_id=self.parent_run_id,
            root_run_id=self.root_run_id,
            session_id=self.session_id,
            depth=self.depth,
        ).plan(run_in_background=True)

    def _decision(
        self,
        plan: SchedulerPlan,
        *,
        goal: str,
        task_count: int,
    ) -> tuple[DelegationDecision, list[str], list[str]]:
        if plan.should_replan or plan.requires_human_review:
            return (
                "manual_review",
                ["scheduler_requires_review"],
                ["orchestration_summary", "schedule_plan"],
            )
        if _has_manual_review_skips(plan):
            return (
                "manual_review",
                ["scheduler_skipped_manual_review_task"],
                ["orchestration_summary", "schedule_plan"],
            )
        if plan.failed_task_ids:
            return (
                "manual_review",
                ["failed_tasks_require_review"],
                ["orchestration_summary"],
            )
        if plan.dispatchable_task_ids:
            return (
                "explicit_orchestration",
                ["dispatchable_task_graph_available"],
                ["orchestrate_until_idle", "orchestration_summary"],
            )
        if plan.in_progress_task_ids:
            return (
                "explicit_orchestration",
                ["running_tasks_need_collection"],
                ["orchestrate_tasks", "orchestration_summary"],
            )
        if task_count == 0 and looks_complex_goal(goal):
            return (
                "create_task_graph",
                ["complex_goal_without_task_graph"],
                ["task_create", "schedule_plan"],
            )
        return ("direct", ["simple_or_no_dispatchable_work"], [])


def looks_complex_goal(goal: str) -> bool:
    normalized = goal.strip()
    if not normalized:
        return False
    if "\n" in normalized:
        return True
    if len(normalized) >= 160:
        return True
    markers = (" and ", " then ", " after ", "first ", "second ", "third ", "1.", "2.")
    lowered = f" {normalized.lower()} "
    return sum(1 for marker in markers if marker in lowered) >= 2


def _has_manual_review_skips(plan: SchedulerPlan) -> bool:
    return any(
        item.reason in {"requires_human_review", "high_risk_root_review"}
        for item in plan.skipped
    )
