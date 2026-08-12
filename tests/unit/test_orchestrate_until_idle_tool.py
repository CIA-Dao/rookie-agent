from __future__ import annotations

import asyncio
import json
from pathlib import Path

from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse
from my_agent.core.subagent.registry import BackgroundTaskRegistry
from my_agent.core.subagent.tool import SpawnAgentTool
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.builtin.dispatch_plan import DispatchPlanTool
from my_agent.core.tools.builtin.orchestrate_tasks import OrchestrateTasksTool
from my_agent.core.tools.builtin.orchestrate_until_idle import OrchestrateUntilIdleTool


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


class _ImmediateProvider:
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
        return LlmResponse(stop_reason="end_turn", text="child done")


def _spawn_tool(
    tmp_path: Path,
    *,
    provider: _BlockingProvider | _ImmediateProvider,
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
    provider: _BlockingProvider | _ImmediateProvider,
    bus: EventBus,
) -> OrchestrateUntilIdleTool:
    dispatch_tool = DispatchPlanTool(
        manager,
        registry,
        _spawn_tool(tmp_path, provider=provider, bus=bus, registry=registry),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        bus=bus,
    )
    return OrchestrateUntilIdleTool(
        OrchestrateTasksTool(
            manager,
            registry,
            dispatch_tool,
            parent_run_id="root-run",
            root_run_id="root-run",
            session_id="sess-1",
            bus=bus,
        )
    )


async def test_orchestrate_until_idle_defaults_to_no_wait(tmp_path: Path) -> None:
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

    assert data["stop_reason"] == "running"
    assert data["next_action"] == "call_again_or_review"
    assert data["rounds_run"] == 1
    assert data["bounds"]["max_wait_seconds"] == 0.0
    assert [item["task_id"] for item in data["rounds"][0]["dispatch"]["dispatched"]] == [1]
    assert manager.get(1).status == "in_progress"
    assert manager.get(2).status == "pending"

    provider.release.set()
    for record in registry.records():
        if record.task is not None:
            await record.task


async def test_orchestrate_until_idle_collects_after_explicit_wait(tmp_path: Path) -> None:
    bus = EventBus()
    provider = _ImmediateProvider()
    registry = BackgroundTaskRegistry()
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("first")

    result = await _tool(
        tmp_path,
        manager=manager,
        registry=registry,
        provider=provider,
        bus=bus,
    ).invoke(
        {
            "max_rounds": 3,
            "max_wait_seconds": 1,
            "poll_interval_seconds": 0.01,
        }
    )
    data = json.loads(result.content)

    assert data["stop_reason"] == "idle"
    assert data["next_action"] == "summarize"
    assert data["rounds_run"] >= 2
    assert data["rounds"][0]["dispatch"]["dispatched"][0]["task_id"] == task.id
    assert data["rounds"][1]["collection"]["completed"][0]["task_id"] == task.id
    assert manager.get(task.id).status == "completed"


async def test_orchestrate_until_idle_stops_on_replan_without_dispatching(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    provider = _ImmediateProvider()
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
    ).invoke({"max_wait_seconds": 1})
    data = json.loads(result.content)

    assert data["stop_reason"] == "replan"
    assert data["next_action"] == "replan"
    assert data["rounds_run"] == 1
    assert data["rounds"][0]["dispatch"]["dispatched"] == []
    assert data["rounds"][0]["dispatch"]["blocked_by"] == "replan"
