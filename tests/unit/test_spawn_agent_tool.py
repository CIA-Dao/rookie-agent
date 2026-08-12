from __future__ import annotations

import asyncio
from pathlib import Path

from my_agent.core.agents.loader import AgentProfile
from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits
from my_agent.core.subagent.tool import AgentCancelTool, AgentResultTool, SpawnAgentTool


class _FakeProvider:
    def __init__(self) -> None:
        self.call_count = 0

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
        self.call_count += 1
        return LlmResponse(
            stop_reason="end_turn",
            text="child done",
            usage=UsageStats(input_tokens=10, output_tokens=2),
        )


class _BlockingProvider(_FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

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
        self.call_count += 1
        self.started.set()
        await self.release.wait()
        return LlmResponse(
            stop_reason="end_turn",
            text="child done",
            usage=UsageStats(input_tokens=10, output_tokens=2),
        )


class _NestedSpawnProvider(_FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.run_ids: list[str] = []

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
        self.call_count += 1
        self.run_ids.append(run_id)
        if self.call_count == 1:
            return LlmResponse(
                stop_reason="tool_calls",
                tool_calls=[
                    ToolCallBlock(
                        id="nested-1",
                        name="spawn_agent",
                        input={
                            "description": "Grandchild inspect",
                            "prompt": "Inspect nested detail",
                            "run_in_background": True,
                            "subagent_type": "reviewer",
                        },
                    )
                ],
                usage=UsageStats(input_tokens=10, output_tokens=2),
            )

        return LlmResponse(
            stop_reason="end_turn",
            text="child collected grandchild",
            usage=UsageStats(input_tokens=10, output_tokens=2),
        )


class _CrashProvider(_FakeProvider):
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
        raise RuntimeError("boom")


def _make_tool(
    tmp_path: Path,
    *,
    depth: int = 0,
    provider: _FakeProvider | None = None,
    parent_bus: EventBus | None = None,
    parent_run_id: str = "parent-run",
    task_registry: BackgroundTaskRegistry | None = None,
    root_run_id: str | None = None,
    limits: SubagentLimits | None = None,
) -> SpawnAgentTool:
    return SpawnAgentTool(
        provider=provider or _FakeProvider(),
        parent_bus=parent_bus or EventBus(),
        parent_run_id=parent_run_id,
        permission_manager=None,
        max_steps=3,
        task_registry=task_registry or BackgroundTaskRegistry(),
        runs_dir=tmp_path,
        session_id="sess-1",
        workspace_root=str(tmp_path),
        depth=depth,
        root_run_id=root_run_id,
        limits=limits,
    )


async def test_spawn_agent_tool_creates_child_context(tmp_path: Path) -> None:
    provider = _FakeProvider()
    tool = _make_tool(tmp_path, provider=provider)

    result = await tool.invoke({"description": "review", "prompt": "Review src/foo.py"})

    assert not result.is_error
    assert result.content == "child done"
    assert provider.call_count == 1


async def test_spawn_agent_tool_rejects_nested_depth_limit(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path, depth=2)

    result = await tool.invoke({"description": "nested", "prompt": "Do nested work"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "max_depth" in result.content


async def test_spawn_agent_tool_rejects_child_count_limit(tmp_path: Path) -> None:
    task_registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="existing-child", goal="child", max_steps=3)
    task_registry.register(
        "existing-child",
        task=None,
        context=context,
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
        description="existing",
        created_at="2026-01-01T00:00:00+00:00",
    )
    tool = _make_tool(
        tmp_path,
        parent_run_id="root-run",
        task_registry=task_registry,
        root_run_id="root-run",
        limits=SubagentLimits(max_children_per_root=1),
    )

    result = await tool.invoke({"description": "new child", "prompt": "Do work"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "max_children_per_root" in result.content


async def test_spawn_agent_tool_rejects_grandchild_count_limit(tmp_path: Path) -> None:
    task_registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="existing-grandchild", goal="grandchild", max_steps=3)
    task_registry.register(
        "existing-grandchild",
        task=None,
        context=context,
        parent_run_id="child-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=2,
        description="existing",
        created_at="2026-01-01T00:00:00+00:00",
    )
    tool = SpawnAgentTool(
        provider=_FakeProvider(),
        parent_bus=EventBus(),
        parent_run_id="child-run",
        permission_manager=None,
        max_steps=3,
        task_registry=task_registry,
        runs_dir=tmp_path,
        session_id="sess-1",
        workspace_root=str(tmp_path),
        depth=1,
        root_run_id="root-run",
        limits=SubagentLimits(max_grandchildren_per_child=1),
    )

    result = await tool.invoke({"description": "new grandchild", "prompt": "Do work"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "max_grandchildren_per_child" in result.content


async def test_spawn_agent_tool_rejects_total_descendant_limit(tmp_path: Path) -> None:
    task_registry = BackgroundTaskRegistry()
    for run_id, parent_run_id, depth in [
        ("child-1", "root-run", 1),
        ("child-2", "root-run", 1),
        ("grandchild-1", "child-1", 2),
    ]:
        task_registry.register(
            run_id,
            task=None,
            context=ExecutionContext(run_id=run_id, goal=run_id, max_steps=3),
            parent_run_id=parent_run_id,
            root_run_id="root-run",
            session_id="sess-1",
            depth=depth,
            description=run_id,
            created_at="2026-01-01T00:00:00+00:00",
        )
    tool = SpawnAgentTool(
        provider=_FakeProvider(),
        parent_bus=EventBus(),
        parent_run_id="child-2",
        permission_manager=None,
        max_steps=3,
        task_registry=task_registry,
        runs_dir=tmp_path,
        session_id="sess-1",
        workspace_root=str(tmp_path),
        depth=1,
        root_run_id="root-run",
        limits=SubagentLimits(max_total_descendants_per_root=3),
    )

    result = await tool.invoke({"description": "one more", "prompt": "Do work"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "max_total_descendants_per_root" in result.content


async def test_spawn_agent_tool_rejects_background_concurrency_limit(
    tmp_path: Path,
) -> None:
    task_registry = BackgroundTaskRegistry()
    started = asyncio.Event()
    release = asyncio.Event()

    async def wait_forever() -> None:
        started.set()
        await release.wait()

    task = asyncio.create_task(wait_forever())
    await started.wait()
    task_registry.register(
        "running-child",
        task,
        ExecutionContext(run_id="running-child", goal="child", max_steps=3),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
        description="running",
        run_in_background=True,
        created_at="2026-01-01T00:00:00+00:00",
    )
    tool = _make_tool(
        tmp_path,
        parent_run_id="root-run",
        task_registry=task_registry,
        root_run_id="root-run",
        limits=SubagentLimits(
            max_children_per_root=4,
            max_total_descendants_per_root=8,
            max_concurrent_background_subagents_per_session=1,
        ),
    )

    result = await tool.invoke(
        {
            "description": "new background",
            "prompt": "Do work",
            "run_in_background": True,
        }
    )

    release.set()
    await task

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "max_concurrent_background_subagents_per_session" in result.content


async def test_completed_background_task_does_not_count_against_concurrency(
    tmp_path: Path,
) -> None:
    task_registry = BackgroundTaskRegistry()

    async def done() -> None:
        return None

    task = asyncio.create_task(done())
    await task
    task_registry.register(
        "done-child",
        task,
        ExecutionContext(run_id="done-child", goal="child", max_steps=3),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
        description="done",
        run_in_background=True,
        created_at="2026-01-01T00:00:00+00:00",
    )
    tool = _make_tool(
        tmp_path,
        task_registry=task_registry,
        root_run_id="root-run",
        limits=SubagentLimits(max_concurrent_background_subagents_per_session=1),
    )

    result = await tool.invoke(
        {
            "description": "new background",
            "prompt": "Do work",
            "run_in_background": True,
        }
    )

    assert not result.is_error
    run_id = result.content.rsplit("=", maxsplit=1)[1]
    record = task_registry.get_record(run_id)
    assert record is not None
    assert record.task is not None
    await record.task


def test_build_child_registry_filters_tools_by_profile(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    profile = AgentProfile(
        name="reviewer",
        description="Review",
        system_prompt="Review code",
        allowed_tools=["read_file", "list_dir"],
    )

    registry = tool._build_child_registry(EventBus(), "child-run", profile)
    tool_names = {schema["name"] for schema in registry.tool_schemas()}

    assert tool_names == {"read_file", "list_dir"}


def test_build_child_registry_exposes_nested_subagent_tools_at_depth_zero(
    tmp_path: Path,
) -> None:
    tool = _make_tool(tmp_path, depth=0)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    tool_names = {schema["name"] for schema in registry.tool_schemas()}

    assert "spawn_agent" in tool_names
    assert "agent_result" in tool_names


def test_build_child_registry_hides_nested_subagent_tools_at_depth_one(
    tmp_path: Path,
) -> None:
    tool = _make_tool(tmp_path, depth=1)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    tool_names = {schema["name"] for schema in registry.tool_schemas()}

    assert "spawn_agent" not in tool_names
    assert "agent_result" not in tool_names


def test_builtin_subagent_profiles_do_not_expose_nested_subagent_tools(
    tmp_path: Path,
) -> None:
    tool = _make_tool(tmp_path, depth=0)
    profiles = [
        AgentProfile(
            name="planner",
            description="Plan",
            system_prompt="Plan work",
            allowed_tools=["read_file", "list_dir", "task_create", "task_update", "task_list"],
        ),
        AgentProfile(
            name="executor",
            description="Execute",
            system_prompt="Execute work",
            allowed_tools=[
                "read_file",
                "list_dir",
                "write_file",
                "bash",
                "task_update",
                "task_list",
            ],
        ),
        AgentProfile(
            name="reviewer",
            description="Review",
            system_prompt="Review code",
            allowed_tools=["read_file", "list_dir", "bash"],
        ),
    ]

    for profile in profiles:
        registry = tool._build_child_registry(EventBus(), f"{profile.name}-run", profile)
        tool_names = {schema["name"] for schema in registry.tool_schemas()}
        assert "spawn_agent" not in tool_names
        assert "agent_result" not in tool_names


async def test_spawn_agent_tool_publishes_subagent_events(tmp_path: Path) -> None:
    parent_bus = EventBus()
    events: list[object] = []

    async def capture(event: object) -> None:
        events.append(event)

    parent_bus.subscribe(capture)
    tool = _make_tool(tmp_path, parent_bus=parent_bus)

    result = await tool.invoke({"description": "review", "prompt": "Review src/foo.py"})

    assert not result.is_error
    event_types = [getattr(event, "type", "") for event in events]
    assert "subagent.started" in event_types
    assert "step.started" in event_types
    assert "step.finished" in event_types
    assert "subagent.finished" in event_types


async def test_spawn_agent_tool_events_include_tree_metadata(tmp_path: Path) -> None:
    parent_bus = EventBus()
    events: list[object] = []

    async def capture(event: object) -> None:
        events.append(event)

    parent_bus.subscribe(capture)
    tool = _make_tool(tmp_path, parent_bus=parent_bus, root_run_id="root-run")

    result = await tool.invoke(
        {
            "description": "review",
            "prompt": "Review src/foo.py",
            "subagent_type": "reviewer",
        }
    )

    assert not result.is_error
    started = next(event for event in events if getattr(event, "type", "") == "subagent.started")
    finished = next(event for event in events if getattr(event, "type", "") == "subagent.finished")
    assert started.parent_run_id == "parent-run"
    assert started.root_run_id == "root-run"
    assert started.depth == 1
    assert started.subagent_type == "reviewer"
    assert finished.parent_run_id == "parent-run"
    assert finished.root_run_id == "root-run"
    assert finished.depth == 1
    assert finished.subagent_type == "reviewer"


async def test_spawn_agent_tool_registers_background_task(tmp_path: Path) -> None:
    task_registry = BackgroundTaskRegistry()
    tool = _make_tool(tmp_path, task_registry=task_registry)

    result = await tool.invoke(
        {
            "description": "review",
            "prompt": "Review src/foo.py",
            "run_in_background": True,
        }
    )

    assert not result.is_error
    assert result.content.startswith("Subagent started in background: run_id=")
    run_id = result.content.rsplit("=", maxsplit=1)[1]
    assert (tmp_path / run_id).is_dir()

    registered = task_registry.get(run_id)
    assert registered is not None
    task, context = registered

    await task

    assert context.run_id == run_id
    assert context.status == "success"
    assert context.messages[-1] == {"role": "assistant", "content": "child done"}


async def test_background_spawn_registry_record_includes_tree_metadata(
    tmp_path: Path,
) -> None:
    task_registry = BackgroundTaskRegistry()
    tool = _make_tool(tmp_path, task_registry=task_registry, root_run_id="root-run")

    result = await tool.invoke(
        {
            "description": "background review",
            "prompt": "Review src/foo.py",
            "run_in_background": True,
            "subagent_type": "reviewer",
        }
    )

    assert not result.is_error
    run_id = result.content.rsplit("=", maxsplit=1)[1]
    record = task_registry.get_record(run_id)

    assert record is not None
    assert record.run_id == run_id
    assert record.parent_run_id == "parent-run"
    assert record.root_run_id == "root-run"
    assert record.depth == 1
    assert record.description == "background review"
    assert record.subagent_type == "reviewer"
    assert record.run_in_background is True

    await record.task


async def test_nested_background_spawn_registry_record_is_grandchild(
    tmp_path: Path,
) -> None:
    task_registry = BackgroundTaskRegistry()
    provider = _NestedSpawnProvider()
    tool = _make_tool(
        tmp_path,
        provider=provider,
        task_registry=task_registry,
        root_run_id="root-run",
    )

    result = await tool.invoke({"description": "child", "prompt": "Delegate nested work"})

    assert not result.is_error
    assert result.content == "child collected grandchild"
    assert len(provider.run_ids) >= 2

    records = task_registry.records()
    assert len(records) == 2
    grandchild = next(record for record in records if record.depth == 2)
    assert grandchild.root_run_id == "root-run"
    assert grandchild.depth == 2
    assert grandchild.description == "Grandchild inspect"
    assert grandchild.subagent_type == "reviewer"
    assert grandchild.run_in_background is True
    assert grandchild.parent_run_id == provider.run_ids[0]

    await grandchild.task


async def test_background_spawn_publishes_started_before_child_finishes(
    tmp_path: Path,
) -> None:
    parent_bus = EventBus()
    events: list[object] = []
    provider = _BlockingProvider()
    task_registry = BackgroundTaskRegistry()

    async def capture(event: object) -> None:
        events.append(event)

    parent_bus.subscribe(capture)
    tool = _make_tool(
        tmp_path,
        provider=provider,
        parent_bus=parent_bus,
        task_registry=task_registry,
    )

    result = await tool.invoke(
        {
            "description": "review",
            "prompt": "Review src/foo.py",
            "run_in_background": True,
        }
    )

    event_types = [getattr(event, "type", "") for event in events]
    assert not result.is_error
    assert "subagent.started" in event_types
    assert "subagent.finished" not in event_types

    run_id = result.content.rsplit("=", maxsplit=1)[1]
    registered = task_registry.get(run_id)
    assert registered is not None
    task, _context = registered

    await provider.started.wait()
    provider.release.set()
    await task


async def test_agent_result_tool_returns_not_found_for_unknown_run() -> None:
    tool = AgentResultTool(BackgroundTaskRegistry())

    result = await tool.invoke({"run_id": "missing"})

    assert result.is_error
    assert result.error_type == "not_found"
    assert "missing" in result.content


async def test_agent_result_tool_reports_running_task() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)
    started = asyncio.Event()
    release = asyncio.Event()

    async def wait_forever() -> None:
        started.set()
        await release.wait()

    task = asyncio.create_task(wait_forever())
    await started.wait()
    registry.register("child-run", task, context)
    tool = AgentResultTool(registry)

    result = await tool.invoke({"run_id": "child-run"})

    release.set()
    await task

    assert not result.is_error
    assert result.content == "Subagent is still running: run_id=child-run"


async def test_agent_cancel_tool_cancels_running_background_task() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)
    started = asyncio.Event()
    release = asyncio.Event()

    async def wait_forever() -> None:
        started.set()
        await release.wait()

    task = asyncio.create_task(wait_forever())
    await started.wait()
    registry.register("child-run", task, context)
    cancel_tool = AgentCancelTool(registry)

    result = await cancel_tool.invoke({"run_id": "child-run"})

    assert not result.is_error
    assert "Cancelled" in result.content
    await asyncio.gather(task, return_exceptions=True)


async def test_agent_cancel_tool_returns_not_found_for_missing_run() -> None:
    tool = AgentCancelTool(BackgroundTaskRegistry())

    result = await tool.invoke({"run_id": "missing"})

    assert result.is_error
    assert result.error_type == "not_found"


async def test_agent_cancel_tool_rejects_completed_run() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)

    async def done() -> None:
        return None

    task = asyncio.create_task(done())
    await task
    registry.register("child-run", task, context, status="success")
    tool = AgentCancelTool(registry)

    result = await tool.invoke({"run_id": "child-run"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "not running" in result.content


async def test_agent_cancel_tool_cascades_to_background_descendants() -> None:
    registry = BackgroundTaskRegistry()
    release = asyncio.Event()

    async def wait_forever() -> None:
        await release.wait()

    parent_task = asyncio.create_task(wait_forever())
    child_task = asyncio.create_task(wait_forever())
    registry.register(
        "parent-run",
        parent_task,
        ExecutionContext(run_id="parent-run", goal="parent", max_steps=3),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=1,
    )
    registry.register(
        "child-run",
        child_task,
        ExecutionContext(run_id="child-run", goal="child", max_steps=3),
        parent_run_id="parent-run",
        root_run_id="root-run",
        session_id="sess-1",
        depth=2,
    )
    tool = AgentCancelTool(registry)

    result = await tool.invoke({"run_id": "parent-run"})

    assert not result.is_error
    assert "count=2" in result.content
    release.set()
    await asyncio.gather(parent_task, child_task, return_exceptions=True)


async def test_agent_result_tool_returns_successful_answer() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)
    context.add_assistant_message("final child answer")
    context.mark_success()

    async def done() -> None:
        return None

    task = asyncio.create_task(done())
    await task
    registry.register("child-run", task, context)
    tool = AgentResultTool(registry)

    result = await tool.invoke({"run_id": "child-run"})

    assert not result.is_error
    assert result.content == "final child answer"


async def test_agent_result_tool_returns_failed_status() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)
    context.mark_failed("max_steps_exceeded")

    async def done() -> None:
        return None

    task = asyncio.create_task(done())
    await task
    registry.register("child-run", task, context)
    tool = AgentResultTool(registry)

    result = await tool.invoke({"run_id": "child-run"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert result.content == "Subagent failed: max_steps_exceeded"


async def test_agent_result_tool_returns_cancelled_status() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)

    async def done() -> None:
        return None

    task = asyncio.create_task(done())
    await task
    registry.register("child-run", task, context, status="cancelled", reason="cancelled")
    tool = AgentResultTool(registry)

    result = await tool.invoke({"run_id": "child-run"})

    assert result.is_error
    assert result.error_type == "cancelled"


async def test_agent_result_tool_returns_timeout_status() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)

    async def done() -> None:
        return None

    task = asyncio.create_task(done())
    await task
    registry.register("child-run", task, context, status="timed_out", reason="timed_out")
    tool = AgentResultTool(registry)

    result = await tool.invoke({"run_id": "child-run"})

    assert result.is_error
    assert result.error_type == "timeout"


async def test_agent_result_tool_returns_crashed_task_exception() -> None:
    registry = BackgroundTaskRegistry()
    context = ExecutionContext(run_id="child-run", goal="child goal", max_steps=3)

    async def crash() -> None:
        raise RuntimeError("boom")

    task = asyncio.create_task(crash())
    await asyncio.gather(task, return_exceptions=True)
    registry.register("child-run", task, context, status="failed", reason="subagent_crashed")
    tool = AgentResultTool(registry)

    result = await tool.invoke({"run_id": "child-run"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "boom" in result.content


async def test_background_spawn_times_out_and_agent_result_reports_timeout(
    tmp_path: Path,
) -> None:
    task_registry = BackgroundTaskRegistry()
    provider = _BlockingProvider()
    tool = _make_tool(
        tmp_path,
        provider=provider,
        task_registry=task_registry,
        limits=SubagentLimits(background_timeout_seconds=0.01),
    )

    result = await tool.invoke(
        {
            "description": "slow",
            "prompt": "Wait",
            "run_in_background": True,
        }
    )

    assert not result.is_error
    run_id = result.content.rsplit("=", maxsplit=1)[1]
    record = task_registry.get_record(run_id)
    assert record is not None
    assert record.task is not None
    await record.task

    assert record.status == "timed_out"
    result_tool = AgentResultTool(task_registry)
    timeout_result = await result_tool.invoke({"run_id": run_id})
    assert timeout_result.is_error
    assert timeout_result.error_type == "timeout"
