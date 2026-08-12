from __future__ import annotations

from pydantic import BaseModel

from my_agent.core.bus.events import PermissionDeniedEvent, PermissionRequestedEvent
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import ToolCallBlock
from my_agent.core.permissions.manager import PermissionManager
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.invocation import invoke_tool
from my_agent.core.tools.registry import ToolRegistry


class _DangerTool(BaseTool):
    name = "bash"
    description = "Fake dangerous tool."
    input_schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self) -> None:
        self.called = False

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.called = True
        return ToolResult(content="executed")


async def test_permission_deny_prevents_tool_execution() -> None:
    tool = _DangerTool()
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    manager = PermissionManager()
    events: list[BaseModel] = []

    async def respond_deny(event: BaseModel) -> None:
        events.append(event)
        if isinstance(event, PermissionRequestedEvent):
            manager.respond(event.tool_use_id, "deny_once")

    bus.subscribe(respond_deny)

    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="tc-1",
            name="bash",
            input={"command": "echo hi"},
        ),
        bus,
        run_id="run-1",
        permission_manager=manager,
        session_id="sess-1",
    )

    assert result.is_error is True
    assert result.error_type == "permission_denied"
    assert tool.called is False
    assert any(isinstance(event, PermissionRequestedEvent) for event in events)
    assert any(isinstance(event, PermissionDeniedEvent) for event in events)


async def test_permission_allow_executes_tool() -> None:
    tool = _DangerTool()
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    manager = PermissionManager()
    events: list[BaseModel] = []

    async def respond_allow(event: BaseModel) -> None:
        events.append(event)
        if isinstance(event, PermissionRequestedEvent):
            manager.respond(event.tool_use_id, "allow_once")

    bus.subscribe(respond_allow)

    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="tc-allow-1",
            name="bash",
            input={"command": "echo hi"},
        ),
        bus,
        run_id="run-1",
        permission_manager=manager,
        session_id="sess-1",
    )

    assert result.is_error is False
    assert result.content == "executed"
    assert tool.called is True
    assert any(isinstance(event, PermissionRequestedEvent) for event in events)
