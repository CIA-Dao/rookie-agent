from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from my_agent.core.bus.events import (
    SchedulerDiagnosisReportedEvent,
    SchedulerDispatchSkippedEvent,
    SchedulerPlanGeneratedEvent,
)
from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.scheduler import SchedulerPlanner
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.builtin import SchedulePlanTool


def _collect_events(bus: EventBus) -> list[BaseModel]:
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    return events


def test_scheduler_plan_reports_task_buckets_and_priority_order(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("low priority", priority=1)
    manager.create("blocked", blocked_by=[1])
    manager.create("high priority", priority=5)
    manager.create("running")
    manager.create("failed")
    manager.assign_run(4, "run-4")
    manager.fail_with_run(5, "run-5", "needs review")

    planner = SchedulerPlanner(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    plan = planner.plan()

    assert plan.ready_task_ids == [3, 1]
    assert plan.blocked_task_ids == [2]
    assert plan.in_progress_task_ids == [4]
    assert plan.failed_task_ids == [5]
    assert plan.dispatchable_task_ids == [3, 1]
    assert [envelope.task_id for envelope in plan.dispatch_envelopes] == [3, 1]
    assert plan.requires_human_review is True
    assert "failed tasks require review: [5]" in plan.diagnostics


def test_scheduler_plan_limits_dispatchable_by_child_capacity(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("first", priority=3)
    manager.create("second", priority=2)
    registry = BackgroundTaskRegistry()

    planner = SchedulerPlanner(
        manager,
        registry,
        limits=SubagentLimits(max_children_per_root=1),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    plan = planner.plan()

    assert plan.dispatchable_task_ids == [1]
    assert [skip.to_dict() for skip in plan.skipped] == [
        {
            "task_id": 2,
            "reason": "max_children_per_root",
            "message": "Root child capacity is exhausted.",
        }
    ]


async def test_scheduler_plan_limits_dispatchable_by_background_capacity(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.create("first", priority=3)
    manager.create("second", priority=2)
    registry = BackgroundTaskRegistry()
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    running = asyncio.create_task(wait())
    registry.register(
        "existing-child",
        running,
        ExecutionContext(run_id="existing-child", goal="child", max_steps=3),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
        run_in_background=True,
    )

    planner = SchedulerPlanner(
        manager,
        registry,
        limits=SubagentLimits(max_concurrent_background_subagents_per_session=2),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    plan = planner.plan()

    assert plan.dispatchable_task_ids == [1]
    assert plan.skipped[0].reason == "max_concurrent_background_subagents_per_session"

    release.set()
    await running


def test_scheduler_plan_skips_human_review_and_high_risk_tasks(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("human", requires_human_review=True)
    manager.create("risky", risk="high")
    manager.create("safe", risk="low")

    planner = SchedulerPlanner(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    plan = planner.plan()

    assert plan.dispatchable_task_ids == [3]
    assert [(skip.task_id, skip.reason) for skip in plan.skipped] == [
        (1, "requires_human_review"),
        (2, "high_risk_root_review"),
    ]
    assert [envelope.task_id for envelope in plan.dispatch_envelopes] == [3]


def test_scheduler_plan_builds_dispatch_envelope_with_task_context(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.create(
        "write parser tests",
        description="Cover JSON parser edge cases.",
        task_type="test",
        priority=4,
        risk="low",
        required_capabilities=["read_code", "run_tests"],
        expected_outputs=["pytest coverage"],
        acceptance_criteria=["new tests fail before fix", "pytest passes"],
        can_parallelize=True,
    )

    planner = SchedulerPlanner(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    plan = planner.plan()

    assert len(plan.dispatch_envelopes) == 1
    envelope = plan.dispatch_envelopes[0]
    assert envelope.task_id == 1
    assert envelope.subject == "write parser tests"
    assert envelope.description == "Cover JSON parser edge cases."
    assert envelope.task_type == "test"
    assert envelope.priority == 4
    assert envelope.risk == "low"
    assert envelope.required_capabilities == ["read_code", "run_tests"]
    assert envelope.expected_outputs == ["pytest coverage"]
    assert envelope.acceptance_criteria == ["new tests fail before fix", "pytest passes"]
    assert envelope.recommended_agent_level == "child"
    assert envelope.parent_run_id == "root-run"
    assert envelope.root_run_id == "root-run"
    assert envelope.session_id == "sess-1"
    assert "Task #1: write parser tests" in envelope.prompt
    assert "Cover JSON parser edge cases." in envelope.prompt
    assert "- read_code" in envelope.prompt
    assert "- pytest coverage" in envelope.prompt
    assert "- new tests fail before fix" in envelope.prompt
    assert "State whether the acceptance criteria were met." in envelope.prompt

    data = plan.to_dict()
    envelopes = data["dispatch_envelopes"]
    assert isinstance(envelopes, list)
    assert envelopes[0]["task_id"] == 1
    assert "prompt" in envelopes[0]


def test_scheduler_plan_marks_cycle_as_replan_and_avoids_dispatch(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("ready")
    manager.create("a")
    manager.create("b", blocked_by=[2])
    manager.update(2, add_blocked_by=[3])

    planner = SchedulerPlanner(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    plan = planner.plan()

    assert plan.should_replan is True
    assert plan.dispatchable_task_ids == []
    assert "cycle detected: 2 -> 3 -> 2" in plan.diagnostics
    assert plan.skipped[0].reason == "requires_replan"


def test_scheduler_plan_marks_deadlock_as_replan(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("blocked")
    manager.update(1, add_blocked_by=[99])

    planner = SchedulerPlanner(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    plan = planner.plan()

    assert plan.should_replan is True
    assert plan.ready_task_ids == []
    assert plan.dispatchable_task_ids == []
    assert "deadlock: pending tasks are blocked: [1]" in plan.diagnostics


async def test_schedule_plan_tool_returns_json_without_mutating_tasks(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("alpha", priority=1)
    tool = SchedulePlanTool(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )

    result = await tool.invoke({})
    data = json.loads(result.content)

    assert not result.is_error
    assert data["dispatchable_task_ids"] == [1]
    assert data["dispatch_envelopes"][0]["task_id"] == 1
    assert "Task #1: alpha" in data["dispatch_envelopes"][0]["prompt"]
    assert manager.get(1).status == "pending"
    assert manager.get(1).assigned_run_id == ""


async def test_schedule_plan_tool_publishes_plan_generated_event(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    manager.create("alpha", priority=1)
    tool = SchedulePlanTool(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )

    result = await tool.invoke({})

    data = json.loads(result.content)
    assert not result.is_error
    assert data["dispatchable_task_ids"] == [1]
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SchedulerPlanGeneratedEvent)
    assert event.run_id == "root-run"
    assert event.session_id == "sess-1"
    assert event.ready_task_ids == [1]
    assert event.dispatchable_task_ids == [1]
    assert event.skipped_task_ids == []
    assert event.diagnostics_count == 0
    assert event.plan_id
    assert manager.get(1).status == "pending"


async def test_schedule_plan_tool_publishes_diagnosis_and_skipped_events(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    manager.create("blocked")
    manager.update(1, add_blocked_by=[99])
    tool = SchedulePlanTool(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )

    result = await tool.invoke({})

    data = json.loads(result.content)
    assert not result.is_error
    assert data["should_replan"] is True
    assert [event.type for event in events] == [
        "scheduler.plan.generated",
        "scheduler.diagnosis.reported",
    ]
    generated, diagnosis = events
    assert isinstance(generated, SchedulerPlanGeneratedEvent)
    assert isinstance(diagnosis, SchedulerDiagnosisReportedEvent)
    assert diagnosis.plan_id == generated.plan_id
    assert diagnosis.should_replan is True
    assert "deadlock: pending tasks are blocked: [1]" in diagnosis.diagnostics


async def test_schedule_plan_tool_publishes_skipped_event(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    manager.create("risky", risk="high")
    tool = SchedulePlanTool(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )

    result = await tool.invoke({})

    data = json.loads(result.content)
    assert not result.is_error
    assert data["skipped"][0]["reason"] == "high_risk_root_review"
    assert [event.type for event in events] == [
        "scheduler.plan.generated",
        "scheduler.dispatch.skipped",
    ]
    generated, skipped = events
    assert isinstance(generated, SchedulerPlanGeneratedEvent)
    assert isinstance(skipped, SchedulerDispatchSkippedEvent)
    assert skipped.plan_id == generated.plan_id
    assert skipped.skipped[0]["task_id"] == 1
    assert skipped.skipped[0]["reason"] == "high_risk_root_review"
