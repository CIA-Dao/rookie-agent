from __future__ import annotations

import json
from typing import Any

TOOL_RESULT_LIMIT = 8_000
TOOL_RESULT_KEEP = 4_000


def truncate_tool_results(
    messages: list[dict[str, Any]],
    limit: int = TOOL_RESULT_LIMIT,
    keep: int = TOOL_RESULT_KEEP,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for message in messages:
        if message.get("role") != "tool":
            result.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, str) or len(content) <= limit:
            result.append(message)
            continue

        ranged = _truncate_ranged_read(content, keep)
        if ranged is not None:
            result.append({**message, "content": ranged})
            continue

        omitted = len(content) - keep
        truncated = (
            content[:keep]
            + f"\n[... {omitted} chars omitted. Full output is available in run events.]"
        )

        result.append({**message, "content": truncated})

    return result


def _truncate_ranged_read(content: str, keep: int) -> str | None:
    """Keep a ranged-read envelope explicit when its content is compacted."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not {"path", "offset", "next_offset"}.issubset(payload):
        return None
    file_content = payload.get("content")
    if not isinstance(file_content, str):
        return None
    payload["content"] = file_content[:keep]
    payload["complete"] = False
    payload["content_truncated"] = True
    payload["continuation_required"] = True
    payload["next_offset"] = payload["offset"]
    payload["sha256"] = None
    return json.dumps(payload, ensure_ascii=False)
