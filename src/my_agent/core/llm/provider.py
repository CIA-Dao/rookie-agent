from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from openai import AsyncOpenAI

from my_agent.core.bus.events import LlmModelSelectedEvent, LlmTokenEvent, LlmUsageEvent
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.token_budget import estimate_prompt_tokens
from my_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from my_agent.core.trace.record import TraceRecord
from my_agent.core.trace.writer import TraceWriter

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v4-pro": 64_000,
}
_DEFAULT_MAX_OUTPUT_TOKENS = 16_384


def _max_output_tokens() -> int:
    raw = os.environ.get("MY_AGENT_LLM_MAX_OUTPUT_TOKENS", str(_DEFAULT_MAX_OUTPUT_TOKENS))
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    return max(1, min(value, 32_768))


def _parse_tool_arguments(raw: str) -> tuple[dict[str, object], str | None]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON tool arguments: {exc.msg}"
    if not isinstance(value, dict):
        return {}, "tool arguments must be a JSON object"
    return value, None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _context_window(model: str) -> int:
    return _MODEL_CONTEXT_WINDOWS.get(model, 64_000)


class DeepSeekProvider:
    def __init__(self, model: str, trace: TraceWriter | None = None) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY not set")
        self._client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model
        self._trace = trace

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
        return _context_window(self._model)

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
        full_messages = [
            {
                "role": "system",
                "content": system or "You are a helpful AI assistant.",
            }
        ] + messages

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tool_schemas
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": _max_output_tokens(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        await bus.publish(
            LlmModelSelectedEvent(
                run_id=run_id,
                model=self._model,
                strategy="static",
                ts=_now(),
            )
        )

        stream = await self._client.chat.completions.create(**kwargs)
        text_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, str]] = {}
        finish = "stop"
        usage: UsageStats | None = None

        async for chunk in stream:
            if chunk.usage:
                context_pct = chunk.usage.prompt_tokens / _context_window(self._model)
                usage = UsageStats(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    context_pct=context_pct,
                )
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            delta = choice.delta

            content = delta.content or ""
            if content:
                text_parts.append(content)
                await bus.publish(LlmTokenEvent(run_id=run_id, token=content, ts=_now()))

            for tool_call in delta.tool_calls or []:
                part = tool_call_parts.setdefault(
                    tool_call.index,
                    {"id": "", "name": "", "arguments": ""},
                )
                if tool_call.id:
                    part["id"] = tool_call.id
                if tool_call.function and tool_call.function.name:
                    part["name"] = tool_call.function.name
                if tool_call.function and tool_call.function.arguments:
                    part["arguments"] += tool_call.function.arguments
                    if self._trace is not None:
                        self._trace.emit(
                            TraceRecord(
                                ts=_now(),
                                direction="LLM->CORE",
                                layer="llm",
                                kind="tool_call_argument_fragment",
                                run_id=run_id,
                                step=step,
                                data={
                                    "index": tool_call.index,
                                    "id": part["id"],
                                    "name": part["name"],
                                    "fragment": tool_call.function.arguments,
                                },
                            )
                        )

        text = "".join(text_parts)
        stop_reason = _map_finish_reason(finish)
        tool_call_errors: dict[str, str] = {}
        tool_calls: list[ToolCallBlock] = []
        for tool_call in tool_call_parts.values():
            arguments, error = _parse_tool_arguments(tool_call["arguments"])
            tool_calls.append(
                ToolCallBlock(
                    id=tool_call["id"],
                    name=tool_call["name"],
                    input=arguments,
                )
            )
            if error:
                tool_call_errors[tool_call["id"]] = error
            if self._trace is not None:
                self._trace.emit(
                    TraceRecord(
                        ts=_now(),
                        direction="LLM->CORE",
                        layer="llm",
                        kind="tool_call_arguments_parsed",
                        run_id=run_id,
                        step=step,
                        data={
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "raw_arguments": tool_call["arguments"],
                            "parsed": error is None,
                            "error": error,
                        },
                    )
                )

        if usage is not None:
            await bus.publish(
                LlmUsageEvent(
                    run_id=run_id,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    context_pct=usage.context_pct,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    cache_creation_input_tokens=usage.cache_creation_input_tokens,
                    ts=_now(),
                )
            )

        return LlmResponse(
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            text=text,
            usage=usage,
            tool_call_errors=tool_call_errors,
        )


def _map_finish_reason(finish: str) -> str:
    if finish == "stop":
        return "end_turn"
    if finish == "tool_calls":
        return "tool_use"
    return finish
