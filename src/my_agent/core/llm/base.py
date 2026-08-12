from __future__ import annotations

from typing import Protocol

from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse


class LLMProvider(Protocol):
    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int: ...

    def context_window(self) -> int: ...

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse: ...
