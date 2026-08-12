from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

from pytest import MonkeyPatch

from my_agent.core.bus.events import LlmTokenEvent, LlmUsageEvent
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.provider import DeepSeekProvider
from my_agent.core.llm.token_budget import estimate_prompt_tokens


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeStream:
        self.last_kwargs = kwargs
        return _FakeStream(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(content="Hello ", tool_calls=None),
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(content="from the model.", tool_calls=None),
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                ),
            ]
        )


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield chunk


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


async def test_deepseek_provider_publishes_text_as_llm_token(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = DeepSeekProvider("test-model")
    fake_client = _FakeClient()
    provider._client = cast(Any, fake_client)

    bus = EventBus()
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)

    response = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        tool_schemas=[],
        bus=bus,
        run_id="run-test",
        system="custom system",
    )

    token_events = [event for event in events if isinstance(event, LlmTokenEvent)]
    usage_events = [event for event in events if isinstance(event, LlmUsageEvent)]
    assert response.text == "Hello from the model."
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.context_pct == 10 / 64_000
    assert [event.token for event in token_events] == ["Hello ", "from the model."]
    assert len(usage_events) == 1
    assert usage_events[0].input_tokens == 10
    assert usage_events[0].output_tokens == 5
    assert usage_events[0].context_pct == 10 / 64_000
    assert fake_client.completions.last_kwargs is not None
    assert fake_client.completions.last_kwargs["messages"][0] == {
        "role": "system",
        "content": "custom system",
    }


def test_deepseek_provider_counts_prompt_tokens_with_fallback(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = DeepSeekProvider("deepseek-v4-pro")
    messages = [{"role": "user", "content": "hello"}]
    tool_schemas = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object"},
        }
    ]
    system = "custom system"

    assert provider.count_tokens(messages=messages, tool_schemas=tool_schemas, system=system) == (
        estimate_prompt_tokens(messages=messages, tool_schemas=tool_schemas, system=system)
    )


def test_deepseek_provider_exposes_context_window(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = DeepSeekProvider("deepseek-v4-pro")

    assert provider.context_window() == 64_000
