from __future__ import annotations

import asyncio

from my_agent.core.context import ExecutionContext
from my_agent.core.subagent.registry import BackgroundTaskRegistry


async def test_background_task_registry_registers_and_returns_task() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)

    async def noop() -> None:
        return None

    task = asyncio.create_task(noop())

    registry.register("child-run", task, context)

    assert registry.get("child-run") == (task, context)
    assert registry.all() == [(task, context)]
    assert registry.get("missing") is None

    await task


async def test_background_task_registry_stores_subagent_record_metadata() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)

    async def noop() -> None:
        return None

    task = asyncio.create_task(noop())

    registry.register(
        "child-run",
        task,
        context,
        parent_run_id="parent-run",
        root_run_id="root-run",
        depth=2,
        description="Inspect focused area",
        subagent_type="reviewer",
        run_in_background=True,
        created_at="2026-01-01T00:00:00+00:00",
    )

    record = registry.get_record("child-run")

    assert record is not None
    assert record.run_id == "child-run"
    assert record.parent_run_id == "parent-run"
    assert record.root_run_id == "root-run"
    assert record.session_id == ""
    assert record.depth == 2
    assert record.description == "Inspect focused area"
    assert record.subagent_type == "reviewer"
    assert record.run_in_background is True
    assert record.created_at == "2026-01-01T00:00:00+00:00"
    assert record.status == "running"
    assert record.completed_at == ""
    assert record.reason == ""
    assert record.task is task
    assert record.context is context
    assert registry.records() == [record]
    assert registry.get("child-run") == (task, context)
    assert registry.all() == [(task, context)]

    await task


async def test_background_task_registry_counts_tree_and_running_background() -> None:
    registry = BackgroundTaskRegistry()
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    running = asyncio.create_task(wait())

    async def done() -> None:
        return None

    completed = asyncio.create_task(done())
    await completed

    registry.register(
        "child-1",
        task=None,
        context=ExecutionContext(run_id="child-1", goal="child", max_steps=3),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
        description="foreground child",
        run_in_background=False,
        created_at="2026-01-01T00:00:00+00:00",
    )
    registry.register(
        "grandchild-1",
        running,
        ExecutionContext(run_id="grandchild-1", goal="grandchild", max_steps=3),
        parent_run_id="child-1",
        root_run_id="root-run",
        session_id="sess-1",
        depth=2,
        description="running background grandchild",
        run_in_background=True,
        created_at="2026-01-01T00:00:00+00:00",
    )
    registry.register(
        "other-session-child",
        completed,
        ExecutionContext(run_id="other-session-child", goal="child", max_steps=3),
        parent_run_id="other-root",
        root_run_id="other-root",
        session_id="sess-2",
        depth=1,
        description="completed background child",
        run_in_background=True,
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert registry.count_descendants("root-run") == 2
    assert registry.count_direct_children("root-run") == 1
    assert registry.count_direct_children("child-1") == 1
    assert registry.count_running_background("sess-1") == 1
    assert registry.count_running_background("sess-2") == 0

    release.set()
    await running


async def test_background_task_registry_cancel_tree_cancels_descendant_tasks() -> None:
    registry = BackgroundTaskRegistry()
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    parent_task = asyncio.create_task(wait())
    child_task = asyncio.create_task(wait())
    registry.register(
        "parent-run",
        parent_task,
        ExecutionContext(run_id="parent-run", goal="parent", max_steps=3),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
        created_at="2026-01-01T00:00:00+00:00",
    )
    registry.register(
        "child-run",
        child_task,
        ExecutionContext(run_id="child-run", goal="child", max_steps=3),
        parent_run_id="parent-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=2,
        created_at="2026-01-01T00:00:00+00:00",
    )

    cancelled = registry.cancel_tree("parent-run")

    assert cancelled == 2
    assert parent_task.cancelled() or parent_task.cancelling()
    assert child_task.cancelled() or child_task.cancelling()

    release.set()
    await asyncio.gather(parent_task, child_task, return_exceptions=True)


async def test_background_task_registry_prunes_completed_records_only() -> None:
    registry = BackgroundTaskRegistry()
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    running_task = asyncio.create_task(wait())
    registry.register(
        "running",
        running_task,
        ExecutionContext(run_id="running", goal="running", max_steps=3),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
        created_at="2026-01-01T00:00:00+00:00",
    )
    for index in range(3):
        run_id = f"done-{index}"
        registry.register(
            run_id,
            task=None,
            context=ExecutionContext(run_id=run_id, goal=run_id, max_steps=3),
            parent_run_id="root-run",
            root_run_id="root-run",
            session_id="sess-1",
            depth=1,
            created_at=f"2026-01-01T00:00:0{index}+00:00",
            status="success",
            completed_at=f"2026-01-01T00:00:0{index}+00:00",
        )

    removed = registry.prune_completed(1)

    assert removed == 2
    assert registry.get_record("running") is not None
    assert len([record for record in registry.records() if record.status != "running"]) == 1

    release.set()
    await running_task
