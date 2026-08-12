from my_agent.core.llm.provider import _parse_tool_arguments


def test_parse_tool_arguments_reports_malformed_json() -> None:
    value, error = _parse_tool_arguments('{"path":"src/App.vue"')

    assert value == {}
    assert error == "invalid JSON tool arguments: Expecting ',' delimiter"


def test_parse_tool_arguments_accepts_complete_object() -> None:
    value, error = _parse_tool_arguments('{"path":"src/App.vue","content":"ok"}')

    assert value == {"path": "src/App.vue", "content": "ok"}
    assert error is None
