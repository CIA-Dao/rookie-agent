from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from my_agent.core.bus.events import SchedulerPlanGeneratedEvent, TaskCreatedEvent
from my_agent.core.config import Config
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse, ToolCallBlock
from my_agent.core.runner import AgentRunner
from my_agent.core.session.model import Session
from my_agent.core.session.store import SessionStore
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.catalog import BUILTIN_TOOL_NAMES
from my_agent.core.tools.invocation import invoke_tool
from my_agent.core.tools.registry import ToolRegistry


# 一个简单的假工具，用于测试
class EchoTool(BaseTool):
    name = "echo"
    description = "原样返回输入"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params.get("text", "")))


class FailingTool(BaseTool):
    name = "fail"
    description = "总是失败的工具"
    input_schema = {"type": "object", "properties": {}}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise RuntimeError("故意失败")


class _FakeProvider:
    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return 0

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
        return LlmResponse(stop_reason="end_turn", text="done")


class _FakeMcpManager:
    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = tools

    def get_tools(self) -> list[BaseTool]:
        return list(self._tools)


def test_register_and_get() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    assert registry.get("echo") == tool
    assert registry.get("nonexistent") is None


def test_tool_schemas() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    schemas = registry.tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"
    assert "input_schema" in schemas[0]


def test_agent_runner_registry_includes_task_tools(tmp_path: Path) -> None:
    runner = AgentRunner(Config())
    registry = runner._build_registry(TaskManager(tmp_path))

    tool_names = {schema["name"] for schema in registry.tool_schemas()}

    assert {
        "read_file",
        "task_create",
        "task_get",
        "task_list",
        "task_update",
        "schedule_plan",
        "list_dir",
        "write_file",
        "bash",
        "delegation_policy",
    } <= tool_names
    assert "note_save" not in tool_names


def test_agent_runner_tool_schemas_are_readable_for_llm(tmp_path: Path) -> None:
    runner = AgentRunner(Config())
    registry = runner._build_registry(TaskManager(tmp_path))
    schemas = {str(schema["name"]): schema for schema in registry.tool_schemas()}

    read_file = schemas["read_file"]
    assert "read file" in str(read_file["description"]).lower()
    assert "file path" in str(read_file["input_schema"]).lower()


def test_agent_runner_registry_filters_tools_with_whitelist(tmp_path: Path) -> None:
    runner = AgentRunner(Config())
    registry = runner._build_registry(
        TaskManager(tmp_path),
        tool_whitelist=["read_file", "bash"],
    )

    tool_names = {schema["name"] for schema in registry.tool_schemas()}

    assert tool_names == {"read_file", "bash"}


def test_agent_runner_registry_empty_whitelist_registers_zero_tools(tmp_path: Path) -> None:
    runner = AgentRunner(Config())
    registry = runner._build_registry(
        TaskManager(tmp_path),
        tool_whitelist=[],
    )

    assert registry.tool_schemas() == []


def test_agent_runner_registry_includes_subagent_tools_for_run(tmp_path: Path) -> None:
    provider = _FakeProvider()
    runner = AgentRunner(Config(), provider=provider)
    registry = runner._build_registry(
        TaskManager(tmp_path),
        run_id="run-1",
        bus=EventBus(),
        provider=provider,
    )

    tool_names = {schema["name"] for schema in registry.tool_schemas()}

    assert "spawn_agent" in tool_names
    assert "delegation_policy" in tool_names
    assert "agent_result" in tool_names
    assert "dispatch_plan" in tool_names
    assert "collect_dispatch_results" in tool_names
    assert "orchestrate_tasks" in tool_names
    assert "orchestrate_until_idle" in tool_names
    assert "orchestration_summary" in tool_names


def test_agent_runner_registry_includes_mcp_tools(tmp_path: Path) -> None:
    runner = AgentRunner(Config(), mcp_manager=_FakeMcpManager([EchoTool()]))  # type: ignore[arg-type]

    registry = runner._build_registry(TaskManager(tmp_path))

    assert registry.get("echo") is not None


def test_agent_runner_available_tool_names_include_mcp_tools(tmp_path: Path) -> None:
    runner = AgentRunner(Config(), mcp_manager=_FakeMcpManager([EchoTool()]))  # type: ignore[arg-type]

    names = runner.available_tool_names()

    assert names[: len(BUILTIN_TOOL_NAMES)] == list(BUILTIN_TOOL_NAMES)
    assert "echo" in names
    assert "echo" not in BUILTIN_TOOL_NAMES


def test_agent_runner_registry_filters_mcp_tools_with_whitelist(tmp_path: Path) -> None:
    runner = AgentRunner(Config(), mcp_manager=_FakeMcpManager([EchoTool()]))  # type: ignore[arg-type]

    registry = runner._build_registry(
        TaskManager(tmp_path),
        tool_whitelist=["read_file"],
    )

    assert registry.get("echo") is None


async def test_agent_runner_registry_includes_note_save_for_session_run(tmp_path: Path) -> None:
    runner = AgentRunner(Config())
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="learning",
        created_at="t1",
        updated_at="t1",
    )
    store.write_meta(session)

    registry = runner._build_registry(
        TaskManager(tmp_path / "tasks"),
        session=session,
        store=store,
        run_id="run-1",
    )
    tool = registry.get("note_save")

    assert tool is not None
    result = await tool.invoke({"content": "remember this"})
    assert not result.is_error
    assert "remember this" in store.read_notes(session.id)


async def test_agent_runner_registry_wires_task_tools_to_event_bus(tmp_path: Path) -> None:
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    runner = AgentRunner(Config(), bus=bus)
    registry = runner._build_registry(
        TaskManager(tmp_path / "tasks"),
        run_id="run-1",
        bus=bus,
    )
    tool = registry.get("task_create")

    assert tool is not None
    result = await tool.invoke({"subject": "eventized task"})

    assert not result.is_error
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, TaskCreatedEvent)
    assert event.run_id == "run-1"
    assert event.subject == "eventized task"


async def test_agent_runner_registry_wires_schedule_plan_to_event_bus(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    runner = AgentRunner(Config(), bus=bus)
    task_manager = TaskManager(tmp_path / "tasks")
    task_manager.create("plan me")
    registry = runner._build_registry(
        task_manager,
        run_id="run-1",
        bus=bus,
    )
    tool = registry.get("schedule_plan")

    assert tool is not None
    result = await tool.invoke({})

    assert not result.is_error
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SchedulerPlanGeneratedEvent)
    assert event.run_id == "run-1"
    assert event.dispatchable_task_ids == [1]


async def test_invoke_tool_success() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    bus = EventBus()
    tc = ToolCallBlock(id="c1", name="echo", input={"text": "hello"})
    result = await invoke_tool(registry, tc, bus, "test")
    assert not result.is_error
    assert result.content == "hello"


async def test_invoke_tool_unknown() -> None:
    registry = ToolRegistry()
    bus = EventBus()
    tc = ToolCallBlock(id="c1", name="nonexistent", input={})
    result = await invoke_tool(registry, tc, bus, "test")
    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "unknown tool" in result.content


async def test_invoke_tool_missing_params() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    bus = EventBus()
    tc = ToolCallBlock(id="c1", name="echo", input={})  # 缺少必填 text
    result = await invoke_tool(registry, tc, bus, "test")
    assert result.is_error
    assert result.error_type == "schema_error"
    assert "text" in result.content


async def test_invoke_tool_runtime_error() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())
    bus = EventBus()
    tc = ToolCallBlock(id="c1", name="fail", input={})
    result = await invoke_tool(registry, tc, bus, "test")
    assert result.is_error
    assert result.error_type == "runtime_error"


async def test_invoke_tool_timeout() -> None:
    class SlowTool(BaseTool):
        name = "slow"
        description = "超时工具"
        input_schema = {"type": "object", "properties": {}}

        async def invoke(self, params: dict[str, object]) -> ToolResult:
            import asyncio

            await asyncio.sleep(10)
            return ToolResult(content="done")

    registry = ToolRegistry()
    registry.register(SlowTool())
    bus = EventBus()
    tc = ToolCallBlock(id="c1", name="slow", input={})
    result = await invoke_tool(registry, tc, bus, "test", timeout=0.01)
    assert result.is_error
    assert result.error_type == "timeout"
