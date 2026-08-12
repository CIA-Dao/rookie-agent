from __future__ import annotations

import json
from typing import Any


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def estimate_prompt_tokens(
    *,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    system: str | None = None,
) -> int:
    payload: dict[str, Any] = {
        "messages": messages,
        "tools": tool_schemas,
    }
    if system is not None:
        payload["system"] = system

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return estimate_text_tokens(text)
