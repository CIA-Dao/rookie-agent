from my_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats


def test_usage_stats_creation() -> None:
    u = UsageStats(input_tokens=100, output_tokens=50)
    assert u.input_tokens == 100
    assert u.output_tokens == 50


def test_tool_call_block_creation() -> None:
    tc = ToolCallBlock(id="call_1", name="read_file", input={"path": "test.txt"})
    assert tc.id == "call_1"
    assert tc.name == "read_file"
    assert tc.input == {"path": "test.txt"}


def test_llm_response_text_only() -> None:
    usage = UsageStats(input_tokens=10, output_tokens=5)
    resp = LlmResponse(stop_reason="end_turn", text="你好", usage=usage)
    assert resp.stop_reason == "end_turn"
    assert resp.text == "你好"
    assert resp.tool_calls == []
    assert resp.usage == usage


def test_llm_response_with_tool_calls() -> None:
    tc = ToolCallBlock(id="call_1", name="read_file", input={"path": "test.txt"})
    resp = LlmResponse(stop_reason="tool_use", tool_calls=[tc], text="")
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_file"
    assert resp.text == ""


def test_llm_response_defaults() -> None:
    resp = LlmResponse(stop_reason="end_turn")
    assert resp.tool_calls == []
    assert resp.text == ""
    assert resp.usage is None
