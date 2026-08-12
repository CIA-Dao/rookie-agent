from __future__ import annotations

import asyncio
import json
from pathlib import Path

from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse
from my_agent.core.subagent.registry import BackgroundTaskRegistry
from my_agent.core.subagent.tool import SpawnAgentTool
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.builtin.dispatch_plan import DispatchPlanTool
from my_agent.core.tools.builtin.orchestrate_tasks import OrchestrateTasksTool


class _BlockingProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return 10

    def context_window(self) -> int:
        return 64_000

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        await self.release.wait()
        return LlmResponse(stop_reason="end_turn", text="child done")


def _done_task() -> asyncio.Task[None]:
    async def done() -> None:
        return None

    return asyncio.create_task(done())


def _spawn_tool(
    tmp_path: Path,
    *,
    provider: _BlockingProvider,
    bus: EventBus,
    registry: BackgroundTaskRegistry,
) -> SpawnAgentTool:
    return SpawnAgentTool(
        provider=provider,
        parent_bus=bus,
        parent_run_id="root-run",
        permission_manager=None,
        max_steps=3,
        task_registry=registry,
        runs_dir=tmp_path,
        session_id="sess-1",
        workspace_root=str(tmp_path),
        root_run_id="root-run",
    )


def _tool(
    tmp_path: Path,
    *,
    manager: TaskManager,
    registry: BackgroundTaskRegistry,
    provider: _BlockingProvider,
    bus: EventBus,
) -> OrchestrateTasksTool:
    dispatch_tool = DispatchPlanTool(
        manager,
        registry,
        _spawn_tool(tmp_path, provider=provider, bus=bus, registry=registry),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )
    return OrchestrateTasksTool(
        manager,
        registry,
        dispatch_tool,
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )


async def test_orchestrate_tasks_collects_before_planning_and_dispatching(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    provider = _BlockingProvider()
    registry = BackgroundTaskRegistry()
    manager = TaskManager(tmp_path / "tasks")
    first = manager.create("first")
    second = manager.create("second", blocked_by=[first.id])
    manager.assign_run(first.id, "child-first")
    context = ExecutionContext(run_id="child-first", goal="work", max_steps=3)
    context.messages.append({"role": "assistant", "content": "done"})
    context.mark_success()
    child_task = _done_task()
    await child_task
    registry.register(
        "child-first",
        child_task,
        context,
        status="success",
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )

    result = await _tool(
        tmp_path,
        manager=manager,
        registry=registry,
        provider=provider,
        bus=bus,
    ).invoke({})
    data = json.loads(result.content)

    assert not result.is_error
    assert data["collection"]["completed"][0]["task_id"] == first.id
    assert data["plan_summary"]["ready_task_ids"] == [second.id]
    assert data["dispatch"]["dispatched"][0]["task_id"] == second.id
    assert manager.get(first.id).status == "completed"
    assert manager.get(second.id).status == "in_progress"
    assert data["next_action"] == "continue"

    provider.release.set()
    for record in registry.records():
        if record.task is not None and not record.task.done():
            await record.task


async def test_orchestrate_tasks_defaults_to_one_dispatched_task(tmp_path: Path) -> None:
    bus = EventBus()
    provider = _BlockingProvider()
    registry = BackgroundTaskRegistry()
    manager = TaskManager(tmp_path / "tasks")
    manager.create("first", priority=2)
    manager.create("second", priority=1)

    result = await _tool(
        tmp_path,
        manager=manager,
        registry=registry,
        provider=provider,
        bus=bus,
    ).invoke({})
    data = json.loads(result.content)

    assert [item["task_id"] for item in data["dispatch"]["dispatched"]] == [1]
    assert manager.get(1).status == "in_progress"
    assert manager.get(2).status == "pending"

    provider.release.set()
    for record in registry.records():
        if record.task is not None:
            await record.task


async def test_orchestrate_tasks_human_review_plan_does_not_dispatch(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    provider = _BlockingProvider()
    registry = BackgroundTaskRegistry()
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("high risk", risk="high")

    result = await _tool(
        tmp_path,
        manager=manager,
        registry=registry,
        provider=provider,
        bus=bus,
    ).invoke({})
    data = json.loads(result.content)

    assert data["dispatch"]["dispatched"] == []
    assert data["dispatch"]["blocked_by"] == "human_review"
    assert manager.get(task.id).status == "pending"
    assert data["next_action"] == "human_review"


async def test_orchestrate_tasks_replan_plan_does_not_dispatch(tmp_path: Path) -> None:
    bus = EventBus()
    provider = _BlockingProvider()
    registry = BackgroundTaskRegistry()
    manager = TaskManager(tmp_path / "tasks")
    first = manager.create("first")
    second = manager.create("second", blocked_by=[first.id])
    manager.update(first.id, add_blocked_by=[second.id])

    result = await _tool(
        tmp_path,
        manager=manager,
        registry=registry,
        provider=provider,
        bus=bus,
    ).invoke({})
    data = json.loads(result.content)

    assert data["dispatch"]["dispatched"] == []
    assert data["dispatch"]["blocked_by"] == "replan"
    assert data["next_action"] == "replan"
    assert "cycle detected" in data["diagnostics"][0]
