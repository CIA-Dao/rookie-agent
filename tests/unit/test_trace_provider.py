from __future__ import annotations

import json
from pathlib import Path

from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse
from my_agent.core.trace.provider import TracingProvider
from my_agent.core.trace.writer import TraceWriter


class _FakeProvider:
    def __init__(self) -> None:
        self.step: int | None = None
        self.count_args: dict[str, object] | None = None

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
        self.step = step
        return LlmResponse(stop_reason="end_turn", text="ok", tool_calls=[], usage=None)

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        self.count_args = {
            "messages": messages,
            "tool_schemas": tool_schemas,
            "system": system,
        }
        return 123

    def context_window(self) -> int:
        return 456


async def test_tracing_provider_records_llm_step(tmp_path: Path) -> None:
    trace_path = tmp_path / "daemon.jsonl"
    writer = TraceWriter(trace_path)
    fake = _FakeProvider()
    provider = TracingProvider(fake, writer, include_payload=False)

    await writer.start()
    try:
        await provider.chat([], [], EventBus(), "run-1", step=3)
    finally:
        await writer.stop()

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert fake.step == 3
    assert [record["kind"] for record in records] == ["api_call", "api_response"]
    assert [record["step"] for record in records] == [3, 3]


def test_tracing_provider_forwards_token_count(tmp_path: Path) -> None:
    provider = TracingProvider(_FakeProvider(), TraceWriter(tmp_path / "daemon.jsonl"))
    messages = [{"role": "user", "content": "hello"}]
    tool_schemas = [{"name": "echo"}]

    count = provider.count_tokens(
        messages=messages,
        tool_schemas=tool_schemas,
        system="system",
    )

    assert count == 123


def test_tracing_provider_forwards_context_window(tmp_path: Path) -> None:
    provider = TracingProvider(_FakeProvider(), TraceWriter(tmp_path / "daemon.jsonl"))

    assert provider.context_window() == 456
