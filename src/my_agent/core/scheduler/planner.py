from __future__ import annotations

from dataclasses import dataclass, field

from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager
from my_agent.core.task.model import Task


@dataclass(frozen=True)
class DispatchEnvelope:
    task_id: int
    subject: str
    description: str
    task_type: str
    priority: int
    risk: str
    required_capabilities: list[str]
    expected_outputs: list[str]
    acceptance_criteria: list[str]
    recommended_agent_level: str
    parent_run_id: str
    root_run_id: str
    session_id: str
    result_expectation: str
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "subject": self.subject,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority,
            "risk": self.risk,
            "required_capabilities": self.required_capabilities,
            "expected_outputs": self.expected_outputs,
            "acceptance_criteria": self.acceptance_criteria,
            "recommended_agent_level": self.recommended_agent_level,
            "parent_run_id": self.parent_run_id,
            "root_run_id": self.root_run_id,
            "session_id": self.session_id,
            "result_expectation": self.result_expectation,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class SchedulerCapacity:
    current_depth: int
    max_depth: int
    direct_children_used: int
    direct_children_available: int
    grandchildren_used: int
    grandchildren_available: int
    descendants_used: int
    descendants_available: int
    running_background_used: int
    background_available: int
    can_spawn_more: bool
    limit_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "current_depth": self.current_depth,
            "max_depth": self.max_depth,
            "direct_children_used": self.direct_children_used,
            "direct_children_available": self.direct_children_available,
            "grandchildren_used": self.grandchildren_used,
            "grandchildren_available": self.grandchildren_available,
            "descendants_used": self.descendants_used,
            "descendants_available": self.descendants_available,
            "running_background_used": self.running_background_used,
            "background_available": self.background_available,
            "can_spawn_more": self.can_spawn_more,
            "limit_reasons": self.limit_reasons,
        }


@dataclass(frozen=True)
class SchedulerTaskSkip:
    task_id: int
    reason: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "reason": self.reason,
            "message": self.message,
        }


@dataclass(frozen=True)
class SchedulerTaskDecision:
    task_id: int
    subject: str
    task_type: str
    priority: int
    risk: str
    suggested_agent_level: str
    recommended_agent_level: str
    dispatchable: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "subject": self.subject,
            "task_type": self.task_type,
            "priority": self.priority,
            "risk": self.risk,
            "suggested_agent_level": self.suggested_agent_level,
            "recommended_agent_level": self.recommended_agent_level,
            "dispatchable": self.dispatchable,
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class SchedulerPlan:
    ready_task_ids: list[int]
    blocked_task_ids: list[int]
    in_progress_task_ids: list[int]
    failed_task_ids: list[int]
    dispatchable_task_ids: list[int]
    skipped: list[SchedulerTaskSkip]
    decisions: list[SchedulerTaskDecision]
    diagnostics: list[str]
    capacity: SchedulerCapacity
    should_replan: bool
    requires_human_review: bool
    dispatch_envelopes: list[DispatchEnvelope] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ready_task_ids": self.ready_task_ids,
            "blocked_task_ids": self.blocked_task_ids,
            "in_progress_task_ids": self.in_progress_task_ids,
            "failed_task_ids": self.failed_task_ids,
            "dispatchable_task_ids": self.dispatchable_task_ids,
            "skipped": [item.to_dict() for item in self.skipped],
            "decisions": [item.to_dict() for item in self.decisions],
            "diagnostics": self.diagnostics,
            "capacity": self.capacity.to_dict(),
            "should_replan": self.should_replan,
            "requires_human_review": self.requires_human_review,
            "dispatch_envelopes": [item.to_dict() for item in self.dispatch_envelopes],
        }


class SchedulerPlanner:
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
        self._limits = limits or SubagentLimits()
        self._parent_run_id = parent_run_id
        self._root_run_id = root_run_id or parent_run_id
        self._session_id = session_id
        self._depth = depth

    def plan(self, *, run_in_background: bool = True) -> SchedulerPlan:
        ready_tasks = sorted(
            self._task_manager.ready_tasks(),
            key=lambda task: (-task.priority, task.id),
        )
        blocked_tasks = self._task_manager.blocked_tasks()
        in_progress_tasks = self._task_manager.in_progress_tasks()
        failed_tasks = self._task_manager.failed_tasks()
        cycles = self._task_manager.detect_cycles()
        deadlock = self._task_manager.diagnose_deadlock()
        diagnostics = self._diagnostics(cycles, deadlock, failed_tasks)
        should_replan = bool(cycles or deadlock)
        requires_human_review = bool(failed_tasks)
        capacity = self._capacity(run_in_background=run_in_background)

        dispatchable_task_ids: list[int] = []
        dispatch_envelopes: list[DispatchEnvelope] = []
        skipped: list[SchedulerTaskSkip] = []
        decisions: list[SchedulerTaskDecision] = []
        available_children = capacity.direct_children_available
        available_grandchildren = capacity.grandchildren_available
        available_descendants = capacity.descendants_available
        available_background = capacity.background_available

        for task in ready_tasks:
            recommendation, recommendation_reasons = self._recommend(task)
            task_reasons = list(recommendation_reasons)
            dispatchable = False

            skip_reason = self._skip_reason(
                task,
                recommendation,
                should_replan=should_replan,
                available_children=available_children,
                available_grandchildren=available_grandchildren,
                available_descendants=available_descendants,
                available_background=available_background,
                run_in_background=run_in_background,
            )

            if skip_reason is None:
                dispatchable = True
                dispatchable_task_ids.append(task.id)
                dispatch_envelopes.append(self._dispatch_envelope(task, recommendation))
                available_descendants -= 1
                if run_in_background:
                    available_background -= 1
                if recommendation == "child":
                    available_children -= 1
                if recommendation == "grandchild":
                    available_grandchildren -= 1
                task_reasons.append("dispatchable")
            else:
                skipped.append(skip_reason)
                task_reasons.append(skip_reason.reason)

            decisions.append(
                SchedulerTaskDecision(
                    task_id=task.id,
                    subject=task.subject,
                    task_type=task.task_type,
                    priority=task.priority,
                    risk=task.risk,
                    suggested_agent_level=task.suggested_agent_level,
                    recommended_agent_level=recommendation,
                    dispatchable=dispatchable,
                    reasons=task_reasons,
                )
            )

        return SchedulerPlan(
            ready_task_ids=[task.id for task in ready_tasks],
            blocked_task_ids=[task.id for task in blocked_tasks],
            in_progress_task_ids=[task.id for task in in_progress_tasks],
            failed_task_ids=[task.id for task in failed_tasks],
            dispatchable_task_ids=dispatchable_task_ids,
            skipped=skipped,
            decisions=decisions,
            diagnostics=diagnostics,
            capacity=capacity,
            should_replan=should_replan,
            requires_human_review=requires_human_review,
            dispatch_envelopes=dispatch_envelopes,
        )

    def _capacity(self, *, run_in_background: bool) -> SchedulerCapacity:
        limits = self._limits
        direct_children_used = self._registry.count_direct_children(self._root_run_id)
        grandchildren_used = self._registry.count_direct_children(self._parent_run_id)
        descendants_used = self._registry.count_descendants(self._root_run_id)
        running_background_used = self._registry.count_running_background(self._session_id)
        direct_children_available = max(0, limits.max_children_per_root - direct_children_used)
        grandchildren_available = max(
            0,
            limits.max_grandchildren_per_child - grandchildren_used,
        )
        descendants_available = max(0, limits.max_total_descendants_per_root - descendants_used)
        background_available = (
            max(0, limits.max_concurrent_background_subagents_per_session - running_background_used)
            if run_in_background
            else limits.max_concurrent_background_subagents_per_session
        )
        limit_reasons: list[str] = []

        if self._depth >= limits.max_depth:
            limit_reasons.append("max_depth")
        if self._depth == 0 and direct_children_available <= 0:
            limit_reasons.append("max_children_per_root")
        if self._depth == 1 and grandchildren_available <= 0:
            limit_reasons.append("max_grandchildren_per_child")
        if descendants_available <= 0:
            limit_reasons.append("max_total_descendants_per_root")
        if run_in_background and background_available <= 0:
            limit_reasons.append("max_concurrent_background_subagents_per_session")

        return SchedulerCapacity(
            current_depth=self._depth,
            max_depth=limits.max_depth,
            direct_children_used=direct_children_used,
            direct_children_available=direct_children_available,
            grandchildren_used=grandchildren_used,
            grandchildren_available=grandchildren_available,
            descendants_used=descendants_used,
            descendants_available=descendants_available,
            running_background_used=running_background_used,
            background_available=background_available,
            can_spawn_more=not limit_reasons,
            limit_reasons=limit_reasons,
        )

    def _diagnostics(
        self,
        cycles: list[list[int]],
        deadlock: str | None,
        failed_tasks: list[Task],
    ) -> list[str]:
        diagnostics: list[str] = []
        for cycle in cycles:
            rendered = " -> ".join(str(task_id) for task_id in cycle)
            diagnostics.append(f"cycle detected: {rendered}")
        if deadlock is not None:
            diagnostics.append(deadlock)
        if failed_tasks:
            failed_ids = [task.id for task in failed_tasks]
            diagnostics.append(f"failed tasks require review: {failed_ids}")
        return diagnostics

    def _recommend(self, task: Task) -> tuple[str, list[str]]:
        risk = task.risk.lower()
        task_type = task.task_type.lower()
        suggested = task.suggested_agent_level.lower()

        if task.requires_human_review:
            return "root", ["requires_human_review"]
        if risk == "high":
            return "root", ["high_risk_root_review"]
        if task_type == "planning":
            return "root", ["planning_stays_root"]
        if suggested in {"root", "child", "grandchild"}:
            return suggested, ["planner_suggested_agent_level"]
        if task.can_parallelize:
            return "child", ["parallelizable"]
        return "child", ["default_child_candidate"]

    def _dispatch_envelope(self, task: Task, recommended_agent_level: str) -> DispatchEnvelope:
        return DispatchEnvelope(
            task_id=task.id,
            subject=task.subject,
            description=task.description,
            task_type=task.task_type,
            priority=task.priority,
            risk=task.risk,
            required_capabilities=list(task.required_capabilities),
            expected_outputs=list(task.expected_outputs),
            acceptance_criteria=list(task.acceptance_criteria),
            recommended_agent_level=recommended_agent_level,
            parent_run_id=self._parent_run_id,
            root_run_id=self._root_run_id,
            session_id=self._session_id,
            result_expectation=(
                "Return a concise result summary, files changed if any, tests or checks run, "
                "and whether the task acceptance criteria were met."
            ),
            prompt=self._dispatch_prompt(task, recommended_agent_level),
        )

    def _dispatch_prompt(self, task: Task, recommended_agent_level: str) -> str:
        lines = [
            f"Task #{task.id}: {task.subject}",
            "",
            "Description:",
            task.description or "(none)",
            "",
            "Scheduler context:",
            f"- task_type: {task.task_type}",
            f"- priority: {task.priority}",
            f"- risk: {task.risk}",
            f"- recommended_agent_level: {recommended_agent_level}",
            "",
            "Required capabilities:",
            *self._bullet_lines(task.required_capabilities),
            "",
            "Expected outputs:",
            *self._bullet_lines(task.expected_outputs),
            "",
            "Acceptance criteria:",
            *self._bullet_lines(task.acceptance_criteria),
            "",
            "Result contract:",
            "- Return a concise summary of what was done.",
            "- List files changed, commands run, and verification results when applicable.",
            "- State whether the acceptance criteria were met.",
            "- Do not update task status directly unless the parent explicitly asks.",
        ]
        return "\n".join(lines)

    def _bullet_lines(self, values: list[str]) -> list[str]:
        if not values:
            return ["- (none)"]
        return [f"- {value}" for value in values]

    def _skip_reason(
        self,
        task: Task,
        recommendation: str,
        *,
        should_replan: bool,
        available_children: int,
        available_grandchildren: int,
        available_descendants: int,
        available_background: int,
        run_in_background: bool,
    ) -> SchedulerTaskSkip | None:
        if should_replan:
            return SchedulerTaskSkip(
                task.id,
                "requires_replan",
                "Task is ready, but the graph has cycle or deadlock diagnostics.",
            )
        if task.requires_human_review:
            return SchedulerTaskSkip(
                task.id,
                "requires_human_review",
                "Task requires human review before dispatch.",
            )
        if task.risk.lower() == "high":
            return SchedulerTaskSkip(
                task.id,
                "high_risk_root_review",
                "High-risk task should be handled by root or reviewed manually.",
            )
        if recommendation == "root":
            return SchedulerTaskSkip(
                task.id,
                "root_owned_task",
                "Task is recommended for root handling, not sub-agent dispatch.",
            )
        if self._depth >= self._limits.max_depth:
            return SchedulerTaskSkip(
                task.id,
                "max_depth",
                "Current agent depth cannot spawn further sub-agents.",
            )
        if available_descendants <= 0:
            return SchedulerTaskSkip(
                task.id,
                "max_total_descendants_per_root",
                "Root descendant capacity is exhausted.",
            )
        if run_in_background and available_background <= 0:
            return SchedulerTaskSkip(
                task.id,
                "max_concurrent_background_subagents_per_session",
                "Session background sub-agent capacity is exhausted.",
            )
        if recommendation == "child":
            if self._depth != 0:
                return SchedulerTaskSkip(
                    task.id,
                    "agent_level_unavailable",
                    "Only the root planner can dispatch child-level work.",
                )
            if available_children <= 0:
                return SchedulerTaskSkip(
                    task.id,
                    "max_children_per_root",
                    "Root child capacity is exhausted.",
                )
            return None
        if recommendation == "grandchild":
            if self._depth != 1:
                return SchedulerTaskSkip(
                    task.id,
                    "agent_level_unavailable",
                    "Grandchild dispatch requires a child agent parent.",
                )
            if available_grandchildren <= 0:
                return SchedulerTaskSkip(
                    task.id,
                    "max_grandchildren_per_child",
                    "Child grandchild capacity is exhausted.",
                )
            return None
        return SchedulerTaskSkip(
            task.id,
            "unknown_agent_level",
            f"Unknown recommended agent level: {recommendation}",
        )
