from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from my_agent.core.bus.events import (
    TaskAssignedEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskFailedEvent,
    TaskStatusChangedEvent,
    TaskUpdatedEvent,
)
from my_agent.core.events.bus import EventBus
from my_agent.core.session.store import SessionStore
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.builtin import (
    NoteSaveTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
)


def _collect_events(bus: EventBus) -> list[BaseModel]:
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    return events


async def test_task_create_tool_creates_task(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    tool = TaskCreateTool(manager)

    result = await tool.invoke({"subject": "plan work", "description": "break down"})

    data = json.loads(result.content)
    assert not result.is_error
    assert data["id"] == 1
    assert data["subject"] == "plan work"
    assert data["description"] == "break down"
    assert data["assigned_run_id"] == ""
    assert data["completed_by_run_id"] == ""
    assert data["failed_by_run_id"] == ""
    assert data["failure_reason"] == ""
    assert data["task_type"] == "general"
    assert data["priority"] == 0
    assert data["risk"] == "medium"
    assert data["required_capabilities"] == []
    assert (tmp_path / "task_1.json").exists()


async def test_task_create_tool_publishes_task_created_event(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    tool = TaskCreateTool(manager, bus=bus, run_id="run-1", session_id="sess-1")

    result = await tool.invoke({"subject": "plan work", "description": "break down"})

    assert not result.is_error
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, TaskCreatedEvent)
    assert event.run_id == "run-1"
    assert event.session_id == "sess-1"
    assert event.task_id == 1
    assert event.subject == "plan work"
    assert event.status == "pending"


async def test_task_create_tool_accepts_scheduler_metadata(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    tool = TaskCreateTool(manager)

    result = await tool.invoke(
        {
            "subject": "write parser tests",
            "task_type": "test",
            "priority": 4,
            "risk": "low",
            "suggested_agent_level": "child",
            "required_capabilities": ["read_code", "run_tests"],
            "expected_outputs": ["test file"],
            "acceptance_criteria": ["pytest passes"],
            "can_parallelize": True,
            "requires_human_review": False,
            "estimated_complexity": "small",
        }
    )

    data = json.loads(result.content)
    assert not result.is_error
    assert data["task_type"] == "test"
    assert data["priority"] == 4
    assert data["risk"] == "low"
    assert data["suggested_agent_level"] == "child"
    assert data["required_capabilities"] == ["read_code", "run_tests"]
    assert data["expected_outputs"] == ["test file"]
    assert data["acceptance_criteria"] == ["pytest passes"]
    assert data["can_parallelize"] is True
    assert data["requires_human_review"] is False
    assert data["estimated_complexity"] == "small"


async def test_task_create_tool_returns_error_for_missing_dependency(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    tool = TaskCreateTool(TaskManager(tmp_path), bus=bus, run_id="run-1")

    result = await tool.invoke({"subject": "blocked", "blocked_by": [99]})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "not found" in result.content
    assert events == []


async def test_task_get_tool_returns_task(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    tool = TaskGetTool(manager)

    result = await tool.invoke({"task_id": "1"})

    data = json.loads(result.content)
    assert not result.is_error
    assert data["subject"] == "alpha"


async def test_task_get_tool_returns_error_for_missing_task(tmp_path: Path) -> None:
    tool = TaskGetTool(TaskManager(tmp_path))

    result = await tool.invoke({"task_id": 99})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "not found" in result.content


async def test_task_list_tool_returns_formatted_tasks(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    manager.create("beta")
    manager.update(1, status="completed")
    tool = TaskListTool(manager)

    result = await tool.invoke({})

    assert not result.is_error
    assert "[x] #1: alpha" in result.content
    assert "[ ] #2: beta" in result.content


async def test_task_update_tool_updates_status_and_dependencies(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    manager.create("beta")
    tool = TaskUpdateTool(manager)

    add_result = await tool.invoke({"task_id": 2, "add_blocked_by": ["1"]})
    complete_result = await tool.invoke({"task_id": 1, "status": "completed"})

    add_data = json.loads(add_result.content)
    complete_data = json.loads(complete_result.content)
    assert not add_result.is_error
    assert add_data["blocked_by"] == [1]
    assert not complete_result.is_error
    assert complete_data["status"] == "completed"
    assert manager.get(2).blocked_by == []


async def test_task_update_tool_publishes_update_and_status_events(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    tool = TaskUpdateTool(manager, bus=bus, run_id="run-parent", session_id="sess-1")

    result = await tool.invoke(
        {
            "task_id": 1,
            "status": "in_progress",
            "assigned_run_id": "run-child",
        }
    )

    data = json.loads(result.content)
    assert not result.is_error
    assert data["status"] == "in_progress"
    assert [event.type for event in events] == [
        "task.updated",
        "task.status_changed",
        "task.assigned",
    ]
    updated, changed, assigned = events
    assert isinstance(updated, TaskUpdatedEvent)
    assert updated.task_id == 1
    assert updated.run_id == "run-parent"
    assert isinstance(changed, TaskStatusChangedEvent)
    assert changed.previous_status == "pending"
    assert changed.status == "in_progress"
    assert isinstance(assigned, TaskAssignedEvent)
    assert assigned.assigned_run_id == "run-child"


async def test_task_update_tool_publishes_completed_event(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    manager.assign_run(1, "run-child")
    tool = TaskUpdateTool(manager, bus=bus, run_id="run-parent", session_id="sess-1")

    result = await tool.invoke(
        {
            "task_id": 1,
            "status": "completed",
            "completed_by_run_id": "run-child",
        }
    )

    assert not result.is_error
    assert [event.type for event in events] == [
        "task.updated",
        "task.status_changed",
        "task.completed",
    ]
    completed = events[2]
    assert isinstance(completed, TaskCompletedEvent)
    assert completed.completed_by_run_id == "run-child"
    assert completed.assigned_run_id == "run-child"


async def test_task_update_tool_publishes_failed_event(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    tool = TaskUpdateTool(manager, bus=bus, run_id="run-parent", session_id="sess-1")

    result = await tool.invoke(
        {
            "task_id": 1,
            "status": "failed",
            "failed_by_run_id": "run-child",
            "failure_reason": "tests failed",
        }
    )

    assert not result.is_error
    assert [event.type for event in events] == [
        "task.updated",
        "task.status_changed",
        "task.failed",
    ]
    failed = events[2]
    assert isinstance(failed, TaskFailedEvent)
    assert failed.failed_by_run_id == "run-child"
    assert failed.failure_reason == "tests failed"


async def test_task_update_tool_does_not_publish_status_changed_for_metadata_only(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    tool = TaskUpdateTool(manager, bus=bus, run_id="run-parent", session_id="sess-1")

    result = await tool.invoke({"task_id": 1, "priority": 5})

    assert not result.is_error
    assert [event.type for event in events] == ["task.updated"]


async def test_task_update_tool_accepts_failed_status_and_run_metadata(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    tool = TaskUpdateTool(manager)

    result = await tool.invoke(
        {
            "task_id": 1,
            "status": "failed",
            "assigned_run_id": "run-a",
            "failed_by_run_id": "run-a",
            "failure_reason": "no compatible skill",
        }
    )

    data = json.loads(result.content)
    assert not result.is_error
    assert data["status"] == "failed"
    assert data["assigned_run_id"] == "run-a"
    assert data["failed_by_run_id"] == "run-a"
    assert data["failure_reason"] == "no compatible skill"


async def test_task_update_tool_accepts_scheduler_metadata(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    manager.assign_run(1, "run-a")
    tool = TaskUpdateTool(manager)

    result = await tool.invoke(
        {
            "task_id": 1,
            "task_type": "review",
            "priority": 6,
            "risk": "medium",
            "suggested_agent_level": "root",
            "required_capabilities": ["read_code"],
            "expected_outputs": ["review notes"],
            "acceptance_criteria": ["no blocker findings"],
            "can_parallelize": False,
            "requires_human_review": True,
            "estimated_complexity": "small",
        }
    )

    data = json.loads(result.content)
    assert not result.is_error
    assert data["status"] == "in_progress"
    assert data["assigned_run_id"] == "run-a"
    assert data["task_type"] == "review"
    assert data["priority"] == 6
    assert data["risk"] == "medium"
    assert data["suggested_agent_level"] == "root"
    assert data["required_capabilities"] == ["read_code"]
    assert data["expected_outputs"] == ["review notes"]
    assert data["acceptance_criteria"] == ["no blocker findings"]
    assert data["can_parallelize"] is False
    assert data["requires_human_review"] is True
    assert data["estimated_complexity"] == "small"


async def test_task_update_tool_returns_error_for_missing_task(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    tool = TaskUpdateTool(TaskManager(tmp_path), bus=bus, run_id="run-1")

    result = await tool.invoke({"task_id": 99, "status": "completed"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "not found" in result.content
    assert events == []


async def test_note_save_tool_appends_session_note(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    tool = NoteSaveTool(store, "sess-1", "run-1")

    result = await tool.invoke({"content": "User prefers concise Chinese explanations."})

    notes = store.read_notes("sess-1")
    assert not result.is_error
    assert result.content == "saved"
    assert "User prefers concise Chinese explanations." in notes
    assert "run-1" in notes


async def test_note_save_tool_rejects_empty_content(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    tool = NoteSaveTool(store, "sess-1", "run-1")

    result = await tool.invoke({"content": "   "})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert result.content == "empty content"
    assert store.read_notes("sess-1") == ""
