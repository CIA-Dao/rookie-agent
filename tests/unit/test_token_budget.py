from __future__ import annotations

from my_agent.core.llm.token_budget import estimate_prompt_tokens


def test_estimate_prompt_tokens_counts_system_prompt() -> None:
    without_system = estimate_prompt_tokens(
        messages=[{"role": "user", "content": "hello"}],
        tool_schemas=[],
        system=None,
    )
    with_system = estimate_prompt_tokens(
        messages=[{"role": "user", "content": "hello"}],
        tool_schemas=[],
        system="x" * 400,
    )

    assert with_system > without_system


def test_estimate_prompt_tokens_counts_tool_schemas() -> None:
    without_tools = estimate_prompt_tokens(
        messages=[{"role": "user", "content": "hello"}],
        tool_schemas=[],
        system=None,
    )
    with_tools = estimate_prompt_tokens(
        messages=[{"role": "user", "content": "hello"}],
        tool_schemas=[
            {
                "name": "read_file",
                "description": "x" * 400,
                "input_schema": {"type": "object"},
            }
        ],
        system=None,
    )

    assert with_tools > without_tools
