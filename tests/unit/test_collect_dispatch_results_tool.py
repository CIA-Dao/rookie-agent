from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from my_agent.core.bus.events import TaskCompletedEvent, TaskFailedEvent
from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.subagent.registry import BackgroundTaskRegistry
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.builtin.collect_dispatch_results import CollectDispatchResultsTool


def _collect_events(bus: EventBus) -> list[BaseModel]:
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    return events


def _done_task() -> asyncio.Task[None]:
    async def done() -> None:
        return None

    task = asyncio.create_task(done())
    return task


def _running_task() -> tuple[asyncio.Task[None], asyncio.Event]:
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    return asyncio.create_task(wait()), release


async def test_collect_dispatch_results_completes_successful_task(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("collect me")
    manager.assign_run(task.id, "child-run")
    context = ExecutionContext(run_id="child-run", goal="work", max_steps=3)
    context.messages.append({"role": "assistant", "content": "done"})
    context.mark_success()
    registry = BackgroundTaskRegistry()
    child_task = _done_task()
    await child_task
    registry.register(
        "child-run",
        child_task,
        context,
        status="success",
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    tool = CollectDispatchResultsTool(manager, registry, bus=bus, run_id="root-run")

    result = await tool.invoke({})
    data = json.loads(result.content)

    assert not result.is_error
    assert data["completed"][0]["task_id"] == task.id
    assert data["completed"][0]["run_id"] == "child-run"
    assert data["completed"][0]["result"] == "done"
    updated = manager.get(task.id)
    assert updated.status == "completed"
    assert updated.completed_by_run_id == "child-run"
    assert any(isinstance(event, TaskCompletedEvent) for event in events)


async def test_collect_dispatch_results_fails_failed_task(tmp_path: Path) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("collect failed")
    manager.assign_run(task.id, "child-run")
    context = ExecutionContext(run_id="child-run", goal="work", max_steps=3)
    context.mark_failed("tests failed")
    registry = BackgroundTaskRegistry()
    child_task = _done_task()
    await child_task
    registry.register(
        "child-run",
        child_task,
        context,
        status="failed",
        reason="tests failed",
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )
    tool = CollectDispatchResultsTool(manager, registry, bus=bus, run_id="root-run")

    result = await tool.invoke({})
    data = json.loads(result.content)

    assert not result.is_error
    assert data["failed"][0]["task_id"] == task.id
    assert data["failed"][0]["reason"] == "tests failed"
    updated = manager.get(task.id)
    assert updated.status == "failed"
    assert updated.failed_by_run_id == "child-run"
    assert updated.failure_reason == "tests failed"
    assert any(isinstance(event, TaskFailedEvent) for event in events)


async def test_collect_dispatch_results_leaves_running_task_in_progress(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("still running")
    manager.assign_run(task.id, "child-run")
    context = ExecutionContext(run_id="child-run", goal="work", max_steps=3)
    registry = BackgroundTaskRegistry()
    child_task, release = _running_task()
    registry.register("child-run", child_task, context, status="running")
    tool = CollectDispatchResultsTool(manager, registry, bus=EventBus(), run_id="root-run")

    result = await tool.invoke({})
    data = json.loads(result.content)

    assert data["running"] == [{"task_id": task.id, "run_id": "child-run"}]
    assert manager.get(task.id).status == "in_progress"

    release.set()
    await child_task


async def test_collect_dispatch_results_reports_missing_record_without_mutation(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("missing")
    manager.assign_run(task.id, "missing-run")
    tool = CollectDispatchResultsTool(
        manager,
        BackgroundTaskRegistry(),
        bus=EventBus(),
        run_id="root-run",
    )

    result = await tool.invoke({})
    data = json.loads(result.content)

    assert data["missing"][0]["task_id"] == task.id
    assert manager.get(task.id).status == "in_progress"


async def test_collect_dispatch_results_respects_task_id_filter(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / "tasks")
    first = manager.create("first")
    second = manager.create("second")
    manager.assign_run(first.id, "first-run")
    manager.assign_run(second.id, "second-run")
    registry = BackgroundTaskRegistry()
    for run_id in ("first-run", "second-run"):
        context = ExecutionContext(run_id=run_id, goal="work", max_steps=3)
        context.mark_success()
        child_task = _done_task()
        await child_task
        registry.register(run_id, child_task, context, status="success")
    tool = CollectDispatchResultsTool(manager, registry, bus=EventBus(), run_id="root-run")

    result = await tool.invoke({"task_ids": [second.id]})
    data = json.loads(result.content)

    assert [item["task_id"] for item in data["completed"]] == [second.id]
    assert manager.get(first.id).status == "in_progress"
    assert manager.get(second.id).status == "completed"
