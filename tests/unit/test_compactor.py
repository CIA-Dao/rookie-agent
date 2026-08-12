from __future__ import annotations

from pathlib import Path
from typing import Any

from my_agent.core.bus.events import ContextCompactedEvent, ContextCompactionFailedEvent
from my_agent.core.compact.compactor import Compactor
from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse, UsageStats
from my_agent.core.session.store import SessionStore


class _StubProvider:
    def __init__(self, response: LlmResponse | None = None, *, fail: bool = False) -> None:
        self.response = response or LlmResponse(
            stop_reason="end_turn",
            text="## 1. Original Goal\nTest\n## 2. Completed Steps\n- done",
            usage=UsageStats(input_tokens=100, output_tokens=30),
        )
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: object,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self.calls.append(
            {
                "messages": messages,
                "tool_schemas": tool_schemas,
                "bus": bus,
                "run_id": run_id,
                "step": step,
                "system": system,
            }
        )
        if self.fail:
            raise RuntimeError("LLM failed")
        return self.response


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Please inspect the project."},
        {"role": "assistant", "content": "I will list files."},
        {"role": "tool", "tool_call_id": "call-1", "content": "src/\ntests/"},
    ]


def _compactor(tmp_path: Path) -> Compactor:
    return Compactor(EventBus(), tmp_path, "sess-1")


async def test_compact_messages_calls_provider_without_tools(tmp_path: Path) -> None:
    provider = _StubProvider()
    compactor = _compactor(tmp_path)

    result = await compactor.compact_messages(_messages(), provider)

    assert result is not None
    assert len(provider.calls) == 1
    assert provider.calls[0]["tool_schemas"] == []
    assert provider.calls[0]["run_id"] == "compact"
    assert provider.calls[0]["step"] == 0


async def test_compact_messages_returns_summary_and_usage_tokens(tmp_path: Path) -> None:
    provider = _StubProvider(
        LlmResponse(
            stop_reason="end_turn",
            text="  summary text  ",
            usage=UsageStats(input_tokens=50, output_tokens=12),
        )
    )
    compactor = _compactor(tmp_path)

    result = await compactor.compact_messages(_messages(), provider)

    assert result is not None
    assert result.summary_text == "summary text"
    assert result.summary_tokens == 12
    assert result.original_token_estimate > 0


async def test_compact_messages_includes_focus_in_request(tmp_path: Path) -> None:
    provider = _StubProvider()
    compactor = _compactor(tmp_path)

    await compactor.compact_messages(_messages(), provider, focus="preserve file paths")

    request = provider.calls[0]["messages"][0]["content"]
    assert isinstance(request, str)
    assert "IMPORTANT: Pay special attention to: preserve file paths" in request


async def test_compact_messages_returns_none_when_summary_is_empty(tmp_path: Path) -> None:
    provider = _StubProvider(LlmResponse(stop_reason="end_turn", text="   "))
    compactor = _compactor(tmp_path)

    result = await compactor.compact_messages(_messages(), provider)

    assert result is None


async def test_compact_messages_returns_none_when_provider_fails(tmp_path: Path) -> None:
    provider = _StubProvider(fail=True)
    compactor = _compactor(tmp_path)

    result = await compactor.compact_messages(_messages(), provider)

    assert result is None


async def test_compact_publishes_failed_event_when_summary_is_unavailable(
    tmp_path: Path,
) -> None:
    provider = _StubProvider(fail=True)
    bus = EventBus()
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)
    context = ExecutionContext(run_id="run-1", goal="original goal", max_steps=5)

    result = await Compactor(bus, tmp_path, "sess-1").compact(context, provider)

    assert result is None
    assert len(events) == 1
    assert isinstance(events[0], ContextCompactionFailedEvent)
    assert events[0].session_id == "sess-1"
    assert events[0].run_id == "run-1"


async def test_compact_replaces_context_writes_summary_and_publishes_event(
    tmp_path: Path,
) -> None:
    provider = _StubProvider(
        LlmResponse(
            stop_reason="end_turn",
            text="compact summary",
            usage=UsageStats(input_tokens=50, output_tokens=7),
        )
    )
    bus = EventBus()
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)
    context = ExecutionContext(run_id="run-1", goal="original goal", max_steps=5)
    context.add_assistant_message("assistant history")

    result = await Compactor(bus, tmp_path, "sess-1").compact(context, provider)

    assert result is not None
    assert context.messages == [
        {
            "role": "user",
            "content": "Compacted conversation summary:\n\ncompact summary",
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue from this compacted summary.",
        },
        {"role": "user", "content": "original goal"},
        {"role": "assistant", "content": "assistant history"},
    ]
    summary_files = list(tmp_path.glob("summary_*.md"))
    assert len(summary_files) == 1
    assert summary_files[0].read_text(encoding="utf-8") == "compact summary\n"
    assert len(events) == 1
    assert isinstance(events[0], ContextCompactedEvent)
    assert events[0].session_id == "sess-1"
    assert events[0].run_id == "run-1"
    assert events[0].summary_tokens == 7


async def test_compact_with_store_archives_thread_and_rewrites_active_context(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "old user", run_id="run-old")
    store.append_message("sess-1", "assistant", "old assistant", run_id="run-old")
    provider = _StubProvider(
        LlmResponse(
            stop_reason="end_turn",
            text="store compact summary",
            usage=UsageStats(input_tokens=50, output_tokens=9),
        )
    )
    context = ExecutionContext(run_id="run-2", goal="new goal", max_steps=5)

    result = await Compactor(
        EventBus(),
        store.session_dir("sess-1"),
        "sess-1",
        store=store,
    ).compact(context, provider)

    assert result is not None
    assert (
        tmp_path / "sess-1" / "archive" / "thread-before-run-2.jsonl"
    ).exists()
    assert store.read_messages("sess-1") == [
        {
            "role": "user",
            "content": "Compacted conversation summary:\n\nstore compact summary",
            "compacted": True,
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue from this compacted summary.",
            "compacted": True,
        },
        {"role": "user", "content": "new goal", "compacted_recent": True},
    ]
