from __future__ import annotations

import json

from my_agent.core.compact.budget import truncate_tool_results


def test_truncate_tool_results_leaves_non_tool_messages_unchanged() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result = truncate_tool_results(messages, limit=10, keep=5)

    assert result == messages


def test_truncate_tool_results_leaves_short_tool_result_unchanged() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": "short"},
    ]

    result = truncate_tool_results(messages, limit=10, keep=5)

    assert result == messages


def test_truncate_tool_results_shortens_long_tool_result_and_keeps_envelope() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "bash",
            "content": "x" * 20,
        },
    ]

    result = truncate_tool_results(messages, limit=10, keep=6)

    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call-1"
    assert result[0]["name"] == "bash"
    assert result[0]["content"] == (
        "x" * 6 + "\n[... 14 chars omitted. Full output is available in run events.]"
    )


def test_truncate_tool_results_does_not_mutate_original_messages() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 20},
    ]

    result = truncate_tool_results(messages, limit=10, keep=6)

    assert messages[0]["content"] == "x" * 20
    assert result[0]["content"] != messages[0]["content"]


def test_truncate_tool_results_marks_ranged_read_as_incomplete() -> None:
    payload = {
        "path": "src/game/engine.js",
        "offset": 0,
        "next_offset": 12000,
        "total_bytes": 20000,
        "complete": False,
        "sha256": None,
        "content": "x" * 9000,
    }
    result = truncate_tool_results(
        [{"role": "tool", "content": json.dumps(payload)}], limit=100, keep=40
    )
    compacted = json.loads(result[0]["content"])
    assert compacted["content_truncated"] is True
    assert compacted["continuation_required"] is True
    assert compacted["complete"] is False
    assert compacted["next_offset"] == 0
    assert compacted["sha256"] is None
