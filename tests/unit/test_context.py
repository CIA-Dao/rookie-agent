from my_agent.core.context import ExecutionContext


def test_init_adds_goal_as_first_message() -> None:
    ctx = ExecutionContext(run_id="test", goal="读 README", max_steps=20)
    assert len(ctx.messages) == 1
    assert ctx.messages[0] == {"role": "user", "content": "读 README"}


def test_init_defaults() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)
    assert ctx.step == 0
    assert ctx.status == "running"
    assert ctx.reason is None


def test_add_assistant_message_text_only() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)
    ctx.add_assistant_message(content="你好", tool_calls=None)
    assert ctx.messages[-1] == {"role": "assistant", "content": "你好"}


def test_add_assistant_message_with_tool_calls() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    ]
    ctx.add_assistant_message(content="", tool_calls=tool_calls)
    assert ctx.messages[-1]["role"] == "assistant"
    assert ctx.messages[-1]["tool_calls"] == tool_calls


def test_add_tool_result() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)
    ctx.add_tool_result("call_1", "文件内容是 hello")
    assert ctx.messages[-1]["role"] == "tool"
    assert ctx.messages[-1]["tool_call_id"] == "call_1"
    assert ctx.messages[-1]["content"] == "文件内容是 hello"


def test_mark_success() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)
    ctx.mark_success()
    assert ctx.status == "success"
    assert ctx.is_done()


def test_mark_failed() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)
    ctx.mark_failed("llm_error")
    assert ctx.status == "failed"
    assert ctx.reason == "llm_error"
    assert ctx.is_done()


def test_is_done_while_running() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)
    assert not ctx.is_done()


def test_system_prompt_returns_base_without_session_notes() -> None:
    ctx = ExecutionContext(run_id="test", goal="x", max_steps=5)

    prompt = ctx.system_prompt("base prompt")

    assert prompt.startswith("base prompt")
    assert "Use tools when you need external facts" in prompt
    assert "Do not claim you inspected files" in prompt
    assert "Use list_dir to discover files or directories." in prompt
    assert "Use read_file before explaining or modifying a file you have not seen." in prompt
    assert "Use bash for tests, builds, and commands that genuinely require a shell." in prompt
    assert "For file size/line metadata use file_metadata." in prompt
    assert "On Windows, do not use Unix-only wc, find, or tail for inspection;" in prompt
    assert "When a tool returns an error" in prompt


def test_system_prompt_includes_session_notes() -> None:
    ctx = ExecutionContext(
        run_id="test",
        goal="x",
        max_steps=5,
        session_notes="  user is learning agents  ",
    )

    prompt = ctx.system_prompt("base prompt")

    assert "Session notes:\nuser is learning agents" in prompt
    assert "Use note_save for durable user preferences" in prompt


def test_system_prompt_includes_global_and_project_context() -> None:
    ctx = ExecutionContext(
        run_id="test",
        goal="x",
        max_steps=5,
        global_context="Prefer concise answers.",
        project_context="Use pnpm for this project.",
    )

    prompt = ctx.system_prompt("base prompt")

    assert "Global context:\nPrefer concise answers." in prompt
    assert "Project context:\nUse pnpm for this project." in prompt


def test_system_prompt_override_replaces_base_prompt() -> None:
    ctx = ExecutionContext(
        run_id="test",
        goal="x",
        max_steps=5,
        system_prompt_override="You are a reviewer.",
    )

    prompt = ctx.system_prompt("base prompt")

    assert prompt.startswith("You are a reviewer.")
    assert "base prompt" not in prompt
    assert "Use tools when you need external facts" in prompt
