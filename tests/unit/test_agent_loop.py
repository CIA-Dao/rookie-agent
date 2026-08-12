from __future__ import annotations

from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.token_budget import estimate_prompt_tokens
from my_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from my_agent.core.loop import AgentLoop
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.registry import ToolRegistry


class _SequenceProvider:
    def __init__(self) -> None:
        self.call_count = 0

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return estimate_prompt_tokens(
            messages=messages,
            tool_schemas=tool_schemas,
            system=system,
        )

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
        if self.call_count == 1:
            return LlmResponse(
                stop_reason="end_turn",
                tool_calls=[
                    ToolCallBlock(
                        id="call-1",
                        name="echo",
                        input={"text": "hello"},
                    )
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="The tool returned hello.")


class _EchoTool(BaseTool):
    name = "echo"
    description = "Return the provided text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params["text"]))


class _AlwaysInvalidWriteTool(BaseTool):
    name = "write_file"
    description = "Write a file."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(
            content="missing required parameters: path, content",
            is_error=True,
            error_type="schema_error",
        )


class _RepeatedInvalidWriteProvider:
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
        return LlmResponse(
            stop_reason="tool_use",
            tool_calls=[ToolCallBlock(id=f"call-{step}", name="write_file", input={})],
        )


class _MalformedArgumentsProvider(_RepeatedInvalidWriteProvider):
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
        call_id = f"malformed-{step}"
        return LlmResponse(
            stop_reason="tool_use",
            tool_calls=[ToolCallBlock(id=call_id, name="write_file", input={})],
            tool_call_errors={call_id: "invalid JSON tool arguments"},
        )


class _RecordingCompactor:
    def __init__(self) -> None:
        self.calls = 0

    async def compact(
        self,
        context: ExecutionContext,
        provider: object,
        focus: str = "",
    ) -> object:
        self.calls += 1
        context.messages = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "continue"},
        ]
        context.persist_from = len(context.messages)
        return object()


class _UsageProvider:
    def __init__(self, context_pct: float) -> None:
        self.context_pct = context_pct

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
        return LlmResponse(
            stop_reason="tool_use",
            tool_calls=[
                ToolCallBlock(
                    id="call-1",
                    name="echo",
                    input={"text": "hello"},
                )
            ],
            usage=UsageStats(input_tokens=100, output_tokens=10, context_pct=self.context_pct),
        )


class _CountingProvider(_SequenceProvider):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.token_count = token_count
        self.count_calls = 0

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        self.count_calls += 1
        return self.token_count

    def context_window(self) -> int:
        return 100


async def test_loop_executes_tool_calls_even_when_stop_reason_is_end_turn() -> None:
    provider = _SequenceProvider()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    context = ExecutionContext(run_id="run-1", goal="echo hello", max_steps=3)

    await AgentLoop(provider, registry, EventBus()).run(context)

    assert provider.call_count == 2
    assert context.status == "success"
    assert context.messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "hello",
    }
    assert context.messages[-1] == {
        "role": "assistant",
        "content": "The tool returned hello.",
    }


async def test_loop_stops_after_repeated_invalid_file_write_calls() -> None:
    provider = _RepeatedInvalidWriteProvider()
    registry = ToolRegistry()
    registry.register(_AlwaysInvalidWriteTool())
    context = ExecutionContext(run_id="run-invalid-write", goal="write a file", max_steps=20)

    await AgentLoop(provider, registry, EventBus()).run(context)  # type: ignore[arg-type]

    assert context.status == "failed"
    assert context.step == 3
    assert context.reason is not None
    assert "repeated_tool_error: write_file" in context.reason


async def test_loop_does_not_invoke_tool_for_malformed_arguments() -> None:
    provider = _MalformedArgumentsProvider()
    registry = ToolRegistry()
    registry.register(_AlwaysInvalidWriteTool())
    context = ExecutionContext(run_id="run-protocol-error", goal="write a file", max_steps=1)

    await AgentLoop(provider, registry, EventBus()).run(context)  # type: ignore[arg-type]

    assert context.status == "failed"
    assert context.reason == "max_steps_exceeded"
    assert context.messages[-1]["content"] == (
        "invalid JSON tool arguments; no tool was executed. Retry with a complete "
        "JSON object containing all required tool fields."
    )


async def test_loop_does_not_compact_from_usage_when_ratio_is_disabled() -> None:
    provider = _UsageProvider(context_pct=0.99)
    compactor = _RecordingCompactor()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    context = ExecutionContext(run_id="run-1", goal="echo hello", max_steps=1)

    await AgentLoop(
        provider,
        registry,
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_context_ratio=0.0,
    ).run(context)

    assert compactor.calls == 0


async def test_loop_compacts_after_tool_result_when_usage_ratio_is_high() -> None:
    provider = _UsageProvider(context_pct=0.90)
    compactor = _RecordingCompactor()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    context = ExecutionContext(run_id="run-1", goal="echo hello", max_steps=1)

    await AgentLoop(
        provider,
        registry,
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_context_ratio=0.80,
    ).run(context)

    assert compactor.calls == 1
    assert context.messages == [
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": "continue"},
    ]


async def test_loop_pre_call_compaction_counts_system_prompt() -> None:
    provider = _SequenceProvider()
    compactor = _RecordingCompactor()
    registry = ToolRegistry()
    context = ExecutionContext(
        run_id="run-1",
        goal="short",
        max_steps=1,
        session_notes="x" * 400,
    )

    await AgentLoop(
        provider,
        registry,
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_token_threshold=100,
    ).run(context)

    assert compactor.calls == 1


async def test_loop_uses_provider_token_counter_for_pre_call_compaction() -> None:
    provider = _CountingProvider(token_count=1_000)
    compactor = _RecordingCompactor()
    registry = ToolRegistry()
    context = ExecutionContext(run_id="run-1", goal="short", max_steps=1)

    await AgentLoop(
        provider,
        registry,
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_token_threshold=100,
    ).run(context)

    assert provider.count_calls == 2
    assert compactor.calls == 1


async def test_loop_uses_token_count_fallback_when_usage_is_missing() -> None:
    provider = _CountingProvider(token_count=90)
    compactor = _RecordingCompactor()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    context = ExecutionContext(run_id="run-1", goal="echo hello", max_steps=1)

    await AgentLoop(
        provider,
        registry,
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_token_threshold=1_000,
        compact_context_ratio=0.80,
    ).run(context)

    assert compactor.calls == 1
