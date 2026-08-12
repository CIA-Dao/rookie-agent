from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from my_agent.core.bus.commands import (
    Command,
    SessionCloseCommand,
    SessionCompactCommand,
    SessionCreateCommand,
    SessionGetHistoryCommand,
    SessionSendMessageCommand,
)
from my_agent.core.bus.events import (
    ContextCompactedEvent,
    Event,
    SchedulerDiagnosisReportedEvent,
    SchedulerPlanGeneratedEvent,
    SkillInvokedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    TaskCreatedEvent,
    TaskStatusChangedEvent,
)


def _parse_command(data: dict[str, Any]) -> object:
    return TypeAdapter(Command).validate_python(data)


def _parse_event(data: dict[str, Any]) -> object:
    return TypeAdapter(Event).validate_python(data)


def test_session_create_command_is_discriminated_by_type() -> None:
    command = _parse_command({"type": "session.create", "mode": "chat", "title": "learn"})

    assert isinstance(command, SessionCreateCommand)
    assert command.mode == "chat"
    assert command.title == "learn"


def test_session_send_message_command_is_discriminated_by_type() -> None:
    command = _parse_command(
        {"type": "session.send_message", "session_id": "sess-1", "content": "hello"}
    )

    assert isinstance(command, SessionSendMessageCommand)
    assert command.session_id == "sess-1"
    assert command.content == "hello"


def test_session_get_history_command_is_discriminated_by_type() -> None:
    command = _parse_command({"type": "session.get_history", "session_id": "sess-1"})

    assert isinstance(command, SessionGetHistoryCommand)
    assert command.session_id == "sess-1"


def test_session_close_command_is_discriminated_by_type() -> None:
    command = _parse_command({"type": "session.close", "session_id": "sess-1"})

    assert isinstance(command, SessionCloseCommand)
    assert command.session_id == "sess-1"


def test_session_compact_command_is_discriminated_by_type() -> None:
    command = _parse_command(
        {
            "type": "session.compact",
            "session_id": "sess-1",
            "focus": "preserve permission discussion",
        }
    )

    assert isinstance(command, SessionCompactCommand)
    assert command.session_id == "sess-1"
    assert command.focus == "preserve permission discussion"


def test_context_compacted_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "context.compacted",
            "session_id": "sess-1",
            "run_id": "run-1",
            "original_tokens": 1000,
            "summary_tokens": 120,
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, ContextCompactedEvent)
    assert event.original_tokens == 1000
    assert event.summary_tokens == 120


def test_skill_invoked_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "skill.invoked",
            "skill_name": "review",
            "arguments": "src/foo.py",
            "run_id": "run-1",
            "session_id": "sess-1",
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, SkillInvokedEvent)
    assert event.skill_name == "review"
    assert event.arguments == "src/foo.py"


def test_subagent_started_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "subagent.started",
            "run_id": "child-run",
            "parent_run_id": "parent-run",
            "description": "review code",
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, SubagentStartedEvent)
    assert event.parent_run_id == "parent-run"
    assert event.root_run_id is None
    assert event.depth is None
    assert event.subagent_type == ""


def test_subagent_started_event_accepts_tree_metadata() -> None:
    event = _parse_event(
        {
            "type": "subagent.started",
            "run_id": "grandchild-run",
            "parent_run_id": "child-run",
            "root_run_id": "root-run",
            "depth": 2,
            "subagent_type": "reviewer",
            "description": "review code",
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, SubagentStartedEvent)
    assert event.parent_run_id == "child-run"
    assert event.root_run_id == "root-run"
    assert event.depth == 2
    assert event.subagent_type == "reviewer"


def test_subagent_finished_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "subagent.finished",
            "run_id": "child-run",
            "parent_run_id": "parent-run",
            "status": "success",
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, SubagentFinishedEvent)
    assert event.status == "success"
    assert event.root_run_id is None
    assert event.depth is None
    assert event.subagent_type == ""


def test_subagent_finished_event_accepts_tree_metadata() -> None:
    event = _parse_event(
        {
            "type": "subagent.finished",
            "run_id": "grandchild-run",
            "parent_run_id": "child-run",
            "root_run_id": "root-run",
            "depth": 2,
            "subagent_type": "reviewer",
            "status": "success",
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, SubagentFinishedEvent)
    assert event.parent_run_id == "child-run"
    assert event.root_run_id == "root-run"
    assert event.depth == 2
    assert event.subagent_type == "reviewer"


def test_task_created_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "task.created",
            "run_id": "run-1",
            "session_id": "sess-1",
            "task_id": 1,
            "subject": "write tests",
            "status": "pending",
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, TaskCreatedEvent)
    assert event.task_id == 1
    assert event.subject == "write tests"


def test_task_status_changed_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "task.status_changed",
            "run_id": "run-1",
            "session_id": "sess-1",
            "task_id": 1,
            "subject": "write tests",
            "previous_status": "pending",
            "status": "in_progress",
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, TaskStatusChangedEvent)
    assert event.previous_status == "pending"
    assert event.status == "in_progress"


def test_scheduler_plan_generated_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "scheduler.plan.generated",
            "run_id": "run-1",
            "session_id": "sess-1",
            "plan_id": "plan-1",
            "parent_run_id": "run-1",
            "root_run_id": "run-1",
            "ready_task_ids": [1],
            "dispatchable_task_ids": [1],
            "skipped_task_ids": [],
            "should_replan": False,
            "requires_human_review": False,
            "diagnostics_count": 0,
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, SchedulerPlanGeneratedEvent)
    assert event.plan_id == "plan-1"
    assert event.dispatchable_task_ids == [1]


def test_scheduler_diagnosis_reported_event_is_discriminated_by_type() -> None:
    event = _parse_event(
        {
            "type": "scheduler.diagnosis.reported",
            "run_id": "run-1",
            "session_id": "sess-1",
            "plan_id": "plan-1",
            "diagnostics": ["cycle detected: 1 -> 2 -> 1"],
            "should_replan": True,
            "requires_human_review": False,
            "ts": "2026-01-01T00:00:00+00:00",
        }
    )

    assert isinstance(event, SchedulerDiagnosisReportedEvent)
    assert event.should_replan is True
    assert event.diagnostics == ["cycle detected: 1 -> 2 -> 1"]
