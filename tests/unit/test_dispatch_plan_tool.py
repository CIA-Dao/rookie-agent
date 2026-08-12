from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from my_agent.core.bus.events import TaskAssignedEvent
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse
from my_agent.core.subagent.registry import BackgroundTaskRegistry
from my_agent.core.subagent.tool import SpawnAgentTool
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.builtin.dispatch_plan import DispatchPlanTool


class _BlockingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
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
        self.started.set()
        await self.release.wait()
        return LlmResponse(stop_reason="end_turn", text="child done")


class _FailingSpawnTool(BaseTool):
    name = "spawn_agent"
    description = "failing spawn"
    input_schema: dict[str, object] = {"type": "object", "properties": {}}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(
            content="Subagent limit max_children_per_root reached",
            is_error=True,
            error_type="runtime_error",
        )


def _collect_events(bus: EventBus) -> list[BaseModel]:
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    return events


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


async def test_dispatch_plan_dispatches_only_enveloped_tasks_and_assigns_run(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    events = _collect_events(bus)
    provider = _BlockingProvider()
    registry = BackgroundTaskRegistry()
    manager = TaskManager(tmp_path / "tasks")
    manager.create("safe implementation", priority=2)
    manager.create("high risk review", risk="high")
    tool = DispatchPlanTool(
        manager,
        registry,
        _spawn_tool(tmp_path, provider=provider, bus=bus, registry=registry),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )

    result = await tool.invoke({})
    data = json.loads(result.content)

    assert not result.is_error
    assert len(data["dispatched"]) == 1
    child_run_id = data["dispatched"][0]["run_id"]
    assert data["dispatched"][0]["task_id"] == 1
    assert manager.get(1).status == "in_progress"
    assert manager.get(1).assigned_run_id == child_run_id
    assert manager.get(2).status == "pending"
    assert manager.get(2).assigned_run_id == ""
    assert child_run_id in {record.run_id for record in registry.records()}
    assert any(isinstance(event, TaskAssignedEvent) for event in events)

    provider.release.set()
    record = registry.get_record(child_run_id)
    assert record is not None and record.task is not None
    await record.task


async def test_dispatch_plan_respects_task_id_filter(tmp_path: Path) -> None:
    bus = EventBus()
    provider = _BlockingProvider()
    registry = BackgroundTaskRegistry()
    manager = TaskManager(tmp_path / "tasks")
    manager.create("first", priority=2)
    manager.create("second", priority=1)
    tool = DispatchPlanTool(
        manager,
        registry,
        _spawn_tool(tmp_path, provider=provider, bus=bus, registry=registry),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )

    result = await tool.invoke({"task_ids": [2]})
    data = json.loads(result.content)

    assert not result.is_error
    assert [item["task_id"] for item in data["dispatched"]] == [2]
    assert manager.get(1).status == "pending"
    assert manager.get(2).status == "in_progress"

    provider.release.set()
    for record in registry.records():
        if record.task is not None:
            await record.task


async def test_dispatch_plan_does_not_assign_task_when_spawn_fails(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / "tasks")
    manager.create("safe implementation")
    tool = DispatchPlanTool(
        manager,
        BackgroundTaskRegistry(),
        _FailingSpawnTool(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=EventBus(),
    )

    result = await tool.invoke({})
    data = json.loads(result.content)

    assert result.is_error
    assert data["dispatched"] == []
    assert data["errors"][0]["task_id"] == 1
    assert manager.get(1).status == "pending"
    assert manager.get(1).assigned_run_id == ""
