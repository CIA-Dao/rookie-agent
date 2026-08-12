from __future__ import annotations

import dataclasses
import time
from datetime import UTC, datetime
from typing import Any

from my_agent.core.events.bus import EventBus
from my_agent.core.llm.base import LLMProvider
from my_agent.core.llm.types import LlmResponse
from my_agent.core.trace.record import TraceRecord
from my_agent.core.trace.writer import TraceWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TracingProvider:
    def __init__(
        self,
        inner: LLMProvider,
        trace: TraceWriter,
        *,
        include_payload: bool = True,
    ) -> None:
        self._inner = inner
        self._trace = trace
        self._include_payload = include_payload

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return self._inner.count_tokens(
            messages=messages,
            tool_schemas=tool_schemas,
            system=system,
        )

    def context_window(self) -> int:
        return self._inner.context_window()

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
        if self._include_payload:
            call_data: dict[str, Any] = {
                "system": system,
                "messages": messages,
                "tool_schemas": tool_schemas,
            }
        else:
            call_data = {
                "message_count": len(messages),
                "tool_count": len(tool_schemas),
                "has_system": system is not None,
            }

        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE->LLM",
                layer="llm",
                kind="api_call",
                run_id=run_id,
                step=step,
                data=call_data,
            )
        )

        t0 = time.monotonic()
        result = await self._inner.chat(
            messages,
            tool_schemas,
            bus,
            run_id,
            step=step,
            system=system,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        if self._include_payload:
            response_data: dict[str, Any] = {
                "stop_reason": result.stop_reason,
                "text": result.text,
                "tool_calls": [dataclasses.asdict(tc) for tc in result.tool_calls],
                "usage": dataclasses.asdict(result.usage) if result.usage else {},
                "latency_ms": latency_ms,
            }
        else:
            response_data = {
                "stop_reason": result.stop_reason,
                "usage": dataclasses.asdict(result.usage) if result.usage else {},
                "latency_ms": latency_ms,
            }

        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="LLM->CORE",
                layer="llm",
                kind="api_response",
                run_id=run_id,
                step=step,
                data=response_data,
            )
        )

        return result
