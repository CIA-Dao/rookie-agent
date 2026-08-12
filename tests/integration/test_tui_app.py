from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Input, RichLog, TextArea

from my_agent.tui.app import (
    AppHeader,
    ChatMessageBlock,
    EventLineBlock,
    LLMStreamBlock,
    MyAgentTuiApp,
    PendingPermission,
    PermissionSelect,
    RunStatusBlock,
    SlashPalette,
    StartupPanel,
    ToolCallBlock,
    _display_width,
)
from my_agent.tui.overlays import ModelSelect, SettingsDialog


@pytest.fixture(autouse=True)
def _no_init_provider_for_tui_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """TUI 集成测试不应因 /init 调用真实 provider 而阻塞。"""
    monkeypatch.setattr(
        "my_agent.core.memory.project_init.create_init_provider", lambda _model: None
    )



def test_tool_call_block_expands_running_params() -> None:
    block = ToolCallBlock("bash", {"command": "uv run pytest", "cwd": "D:/projects/rookie-agent"})

    assert "params" not in str(block.render())

    block.toggle_details()

    text = str(block.render())
    assert "running" in text
    assert "details" in text
    assert "params" in text
    assert '"command": "uv run pytest"' in text
    assert '"cwd": "D:/projects/rookie-agent"' in text


def test_tool_call_block_expands_failed_error_details() -> None:
    block = ToolCallBlock("bash", {"command": "bad command"})
    block.set_result(31, is_error=True, error_message="command not found")

    block.toggle_details()

    text = str(block.render())
    assert "failed" in text
    assert "details" in text
    assert "params" in text
    assert '"command": "bad command"' in text
    assert "error" in text
    assert "command not found" in text


async def test_tui_sends_message_and_renders_llm_tokens(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    session_id = "session-test-123"

    # Skip real bootstrap (we provide a TCP mock server instead).
    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(
            server_version="0.0.1",
            uptime_ms=10,
            latency_ms=1,
        )
        return CoreStartOk(
            pid=0,
            host=host,
            port=port,
            ready=ready,
            already_running=True,
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-test", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-test-123", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                if request["method"] == "permission.respond":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"ok": True},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    return
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": result,
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "run.started", "run_id": "run-test-123", "goal": "hi"},
                        {"type": "step.started", "run_id": "run-test-123", "step": 1},
                        {
                            "type": "llm.model_selected",
                            "run_id": "run-test-123",
                            "model": "deepseek-chat",
                            "strategy": "static",
                        },
                        {
                            "type": "permission.requested",
                            "run_id": "run-test-123",
                            "tool_use_id": "tool-1",
                            "tool_name": "read_file",
                            "param_preview": "path='README.md'",
                        },
                        {
                            "type": "skill.invoked",
                            "run_id": "run-test-123",
                            "skill_name": "review",
                            "arguments": "README.md",
                        },
                        {
                            "type": "subagent.started",
                            "run_id": "child-run-123",
                            "parent_run_id": "run-test-123",
                            "description": "review README.md",
                        },
                        {
                            "type": "subagent.finished",
                            "run_id": "child-run-123",
                            "parent_run_id": "run-test-123",
                            "status": "success",
                        },
                        {
                            "type": "tool.call_started",
                            "run_id": "run-test-123",
                            "tool_use_id": "tool-1",
                            "tool_name": "read_file",
                            "params": {"path": "README.md"},
                        },
                        {
                            "type": "tool.call_finished",
                            "run_id": "run-test-123",
                            "tool_use_id": "tool-1",
                            "tool_name": "read_file",
                            "elapsed_ms": 12,
                        },
                        {"type": "llm.token", "run_id": "run-test-123", "token": "Hello"},
                        {"type": "llm.token", "run_id": "run-test-123", "token": " there"},
                        {
                            "type": "llm.usage",
                            "run_id": "run-test-123",
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "context_pct": 0.42,
                            "cache_read_input_tokens": 8,
                            "cache_creation_input_tokens": 4,
                        },
                        {
                            "type": "context.compacted",
                            "session_id": session_id,
                            "run_id": "run-test-123",
                            "original_tokens": 120000,
                            "summary_tokens": 8000,
                        },
                        {
                            "type": "task.assigned",
                            "run_id": "run-test-123",
                            "task_id": "task-1",
                        },
                        {
                            "type": "scheduler.plan.generated",
                            "run_id": "run-test-123",
                            "plan_id": "plan-1",
                        },
                        {
                            "type": "engine.internal_noise",
                            "run_id": "run-test-123",
                        },
                        {"type": "step.finished", "run_id": "run-test-123", "step": 1},
                        {
                            "type": "run.finished",
                            "run_id": "run-test-123",
                            "status": "success",
                            "steps": 1,
                        },
                    ):
                        event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                        writer.write(event_line.encode())
                    await writer.drain()

            for result in (
                {"summary_tokens": 42, "saved_tokens": 128},
                {"run_id": "run-skill-123", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": result,
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            await asyncio.sleep(0.1)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.2)

            log = app.query_one("#log", RichLog)
            prompt = app.query_one("#prompt", TextArea)

            assert prompt.has_focus
            assert requests[0]["method"] == "event.subscribe"
            # P1-1: subscription must include a topic that matches llm.usage
            topics = requests[0]["params"]["topics"]
            assert "llm.*" in topics or "llm.usage" in topics
            # P1-2/P1-3/P1-4: subscribe context.*, subagent.*, step.*
            assert "context.*" in topics
            assert "subagent.*" in topics
            assert "step.*" in topics
            assert "task.*" in topics
            assert "scheduler.*" in topics
            assert requests[1]["method"] == "session.create"
            assert requests[1]["params"]["mode"] == "chat"
            assert requests[1]["params"]["workspace_root"]

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert any("no tool calls yet" in line.text for line in log.lines)

            await pilot.press("h")
            await pilot.press("i")
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert prompt.text == ""
            assert not prompt.disabled
            assert requests[2]["method"] == "session.send_message"
            assert requests[2]["params"]["session_id"] == session_id
            assert requests[2]["params"]["content"] == "hi"

            event_blocks = list(app.query(EventLineBlock))
            event_texts = [str(b.render()) for b in event_blocks]
            assert any("run" in t and "started: run-test-123" in t for t in event_texts)
            assert any(
                "permission" in t and "read_file" in t and "README.md" in t
                for t in event_texts
            )
            assert any(
                "skill" in t and "review" in t and "README.md" in t
                for t in event_texts
            )
            assert any("run" in t and "completed" in t for t in event_texts)
            # P1-3: subagent events produce EventLineBlock with semantic content
            assert any(
                "SUBAGENT" in t.upper()
                and "started" in t
                and "review README.md" in t
                and "child-run-123" in t
                for t in event_texts
            )
            assert any(
                "SUBAGENT" in t.upper()
                and "finished" in t
                and "child-run-123" in t
                and "success" in t
                for t in event_texts
            )

            # User messages are mounted as ChatMessageBlock instances.
            chat_blocks = list(app.query(ChatMessageBlock))
            user_blocks = [b for b in chat_blocks if b.role == "user"]
            assert len(user_blocks) == 1
            assert user_blocks[0].content == "hi"
            assert "hi" in str(user_blocks[0].render())
            assert any("YOU //" in line.text and "hi" in line.text for line in log.lines)

            status_block = app.query_one(RunStatusBlock)
            status_text = str(status_block.render())
            # P2-0.5: compact status — short id, tools X/Y/Z, ctx N%, no full usage
            assert "success" in status_text
            assert "run-test-123" not in status_text
            assert "test-123" in status_text
            assert "step" in status_text
            assert "1 done" in status_text
            assert "tools 0/1/0" in status_text
            assert "ctx 42%" in status_text
            assert "deepseek-chat" in status_text
            assert "in=100" not in status_text
            assert "out=20" not in status_text
            assert "cache=8/4" not in status_text

            block = app.query_one(LLMStreamBlock)
            assert block.text == "Hello there"
            assert not block.display
            assert any("Hello there" in line.text for line in log.lines)

            # P1-0 noise reduction: low-level run/skill/system events must NOT
            # appear as visible RichLog lines.
            log_text = "\n".join(line.text for line in log.lines)
            assert "YOU //" in log_text
            assert "hi" in log_text
            assert "Hello there" in log_text
            assert "run  started:" not in log_text, (
                f"run.started leaked into visible log: {log_text!r}"
            )
            assert "run  finished:" not in log_text, (
                f"run.finished leaked into visible log: {log_text!r}"
            )
            assert "skill  review" not in log_text, (
                f"skill.invoked leaked into visible log: {log_text!r}"
            )
            # P1-1/P1-2: usage/model event payloads must not leak into chat log.
            assert "llm.usage" not in log_text
            assert "LLM usage" not in log_text
            assert "usage in=100" not in log_text
            assert "llm.model_selected" not in log_text
            assert "model deepseek-chat" not in log_text
            assert "deepseek-chat" not in log_text
            # P1-2/P1-3: important visible events remain visible.
            assert "context compacted" in log_text
            assert "original=120000" in log_text
            assert "summary=8000" in log_text
            assert "saved~=112000" in log_text
            assert "subagent" in log_text.lower() and "started:" in log_text
            assert "review README.md" in log_text
            assert "child=child-run-123" in log_text
            assert "subagent" in log_text.lower() and "finished:" in log_text
            assert "status=success" in log_text
            # G1: future task/scheduler event families are visible; unrelated
            # unknown infrastructure events remain hidden.
            assert "TASK //" in log_text
            assert "assigned #task-1" in log_text
            assert "SCHEDULER //" in log_text
            assert "plan plan=plan-1 ready=0 dispatchable=0 skipped=0" in log_text
            assert "engine.internal_noise" not in log_text
            # P1-4/P2-0.5: step/activity stays out of visible chat log.
            assert "step.started" not in log_text
            assert "step.finished" not in log_text
            assert "step 1 running" not in log_text
            assert "step 1 done" not in log_text
            assert "working" not in log_text
            # P2-0: permission help text moved to PermissionSelect.
            assert "y=allow once" not in log_text

            tool_block = app.query_one(ToolCallBlock)
            assert tool_block.tool_name == "read_file"
            assert "done" in str(tool_block.render())
            assert "README.md" in str(tool_block.render())
            assert "params" not in str(tool_block.render())

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)

            expanded_tool_text = str(tool_block.render())
            assert "details" in expanded_tool_text
            assert "params" in expanded_tool_text
            assert '"path": "README.md"' in expanded_tool_text
            visible_tool_text = "\n".join(line.text for line in log.lines)
            assert "details" in visible_tool_text
            assert "params" in visible_tool_text
            assert '"path": "README.md"' in visible_tool_text

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)

            collapsed_tool_text = str(tool_block.render())
            assert "details" not in collapsed_tool_text
            assert "params" not in collapsed_tool_text
            visible_tool_text = "\n".join(line.text for line in log.lines)
            assert "details" not in visible_tool_text
            assert "params" not in visible_tool_text
            assert '"path": "README.md"' not in visible_tool_text

            prompt.load_text("/too")
            await pilot.pause(0.1)

            palette = app.query_one("#slash-palette", SlashPalette)
            assert palette.display
            assert palette.selected_command is not None
            assert palette.selected_command.command == "/tools"

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert prompt.text == "/tools"

            await pilot.press("enter")
            await pilot.pause(0.1)

            command_tool_text = str(tool_block.render())
            assert "details" in command_tool_text
            assert "params" in command_tool_text
            assert '"path": "README.md"' in command_tool_text
            visible_tool_text = "\n".join(line.text for line in log.lines)
            assert "TOOL" in visible_tool_text
            assert "read_file" in visible_tool_text
            assert "details" in visible_tool_text
            assert "params" in visible_tool_text
            assert '"path": "README.md"' in visible_tool_text

            # A run may finish after Core has already resolved or abandoned the pending
            # permission. In that case the TUI must not leave a stale selector behind.
            permission_select = app.query_one(PermissionSelect)
            assert not permission_select.display

            prompt.load_text("/compact README focus")
            await pilot.press("enter")
            await pilot.pause(0.2)

            compact_requests = [
                request for request in requests if request["method"] == "session.compact"
            ]
            assert len(compact_requests) == 1
            assert compact_requests[0]["params"]["session_id"] == session_id
            assert compact_requests[0]["params"]["focus"] == "README focus"
            assert not prompt.disabled
            assert any(
                "context compacted summary=42 saved~=128" in line.text
                for line in log.lines
            )

            await pilot.press("/")
            await pilot.pause(0.1)

            palette = app.query_one("#slash-palette", SlashPalette)
            assert palette.display
            assert palette.selected_command is not None
            assert palette.selected_command.command.startswith("/compact")
            assert palette.selected_command.insert_text == "/compact "

            await pilot.press("enter")
            await pilot.pause(0.1)

            assert prompt.text == "/compact "
            assert len(requests) == 4

            prompt.load_text("/rev")
            await pilot.pause(0.1)

            assert palette.display
            assert palette.selected_command is not None
            assert palette.selected_command.command == "/review"

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert prompt.text == "/review "

            await pilot.press(*"README.md")
            assert prompt.text == "/review README.md"

            prompt.load_text("/ini")
            await pilot.pause(0.1)

            assert palette.display
            assert palette.selected_command is not None
            assert palette.selected_command.command == "/init"

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert prompt.text == "/init"

            # 斜杠补全后 palette 会重新打开（/init 匹配单行 / 模式）。
            # 手动关闭 palette 然后按 Enter 实际提交 /init。
            await pilot.press("escape")
            await pilot.pause(0.1)

            palette = app.query_one("#slash-palette", SlashPalette)
            assert not palette.display

            # /init is now intercepted client-side (P5) — pressing enter
            # triggers local cmd_init, not a daemon send_message.
            await pilot.press("enter")
            await pilot.pause(0.2)

            # Verify the local workspace files were created.
            assert (Path.cwd() / ".my-agent" / "context.md").is_file()
            assert (Path.cwd() / "AGENTS.md").is_file()
            # Still only 4 requests (no extra send_message for /init).
            assert len(requests) == 4


# ---------------------------------------------------------------------------
# P2-0.6: Streaming wrap — long text does not create unbounded RichLog line
# ---------------------------------------------------------------------------


async def test_tui_long_assistant_text_does_not_create_unbounded_log_line(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """长中文 assistant 文本不会产生超长 RichLog.lines（< 200 chars/line）。"""
    requests: list[dict[str, Any]] = []
    session_id = "session-wrap"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    # 300+ 字无换行中文文本
    long_text = "时光如水，岁月如歌，人生如梦，世事如棋。" * 20

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-w", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-wrap", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result})
                writer.write((resp + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    # 分 3 块推送长文本
                    for chunk in (
                        long_text[:100],
                        long_text[100:200],
                        long_text[200:],
                    ):
                        ev = {"kind": "event",
                              "event": {"type": "llm.token", "token": chunk, "run_id": "run-wrap"}}
                        writer.write((json.dumps(ev) + "\n").encode())
                    # finish
                    fev = {"kind": "event",
                           "event": {"type": "run.finished", "run_id": "run-wrap",
                                     "status": "success", "steps": 1}}
                    writer.write((json.dumps(fev) + "\n").encode())
                    await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.6)

            log = app.query_one("#log", RichLog)
            # 不应该有单条超长 RichLog line
            long_lines = [line.text for line in log.lines if "时光如水" in line.text]
            assert long_lines, "assistant text not found in log"
            assert all(len(line) < 200 for line in long_lines), (
                f"long lines found: {[len(ln) for ln in long_lines]}"
            )


# ---------------------------------------------------------------------------
# P2-0.6.2: Streaming tokens are buffered; final answer renders once
# ---------------------------------------------------------------------------


async def test_tui_streaming_tokens_are_buffered_and_rendered_once_on_finish(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 短 token → 不渲染半成品，run.finished 后只写一次完整回答。"""
    requests: list[dict[str, Any]] = []
    session_id = "session-batch"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    # Generate 100 short tokens
    tokens = [f"t{i:02d}" for i in range(100)]
    expected_text = "".join(tokens)

    render_calls = 0
    from my_agent.tui import app as tui_app
    original = tui_app.MyAgentTuiApp._render_llm_log_line

    def spy(self: Any, content: str) -> None:
        nonlocal render_calls
        render_calls += 1
        original(self, content)

    monkeypatch.setattr(tui_app.MyAgentTuiApp, "_render_llm_log_line", spy)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-b", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-batch", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result})
                writer.write((resp + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for token in tokens:
                        ev = {"kind": "event",
                              "event": {"type": "llm.token", "token": token, "run_id": "run-batch"}}
                        writer.write((json.dumps(ev) + "\n").encode())
                    fev = {"kind": "event",
                           "event": {"type": "run.finished", "run_id": "run-batch",
                                     "status": "success", "steps": 1}}
                    writer.write((json.dumps(fev) + "\n").encode())
                    await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(1.0)

            block = app.query_one(LLMStreamBlock)
            assert block.text == expected_text, (
                f"expected {expected_text!r}, got {block.text!r}"
            )
            # The visible log renders only once, with the completed answer.
            assert render_calls == 1, (
                f"render called {render_calls} times (expected exactly 1)"
            )
# P1-2: context.compaction_failed shows visible error in RichLog.lines
# ---------------------------------------------------------------------------


async def test_tui_context_compaction_failed_shows_visible_error(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """context.compaction_failed event → visible error line in RichLog.lines."""
    requests: list[dict[str, Any]] = []
    session_id = "session-compact-fail"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-x", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-x", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "run.started", "run_id": "run-x", "goal": "hi"},
                        {
                            "type": "context.compaction_failed",
                            "session_id": session_id,
                            "run_id": "run-x",
                            "reason": "summary_unavailable",
                        },
                        {
                            "type": "run.finished",
                            "run_id": "run-x",
                            "status": "success",
                            "steps": 1,
                        },
                    ):
                        event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                        writer.write(event_line.encode())
                    await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.2)

            log = app.query_one("#log", RichLog)
            log_text = "\n".join(line.text for line in log.lines)
            assert "context compaction failed" in log_text, (
                f"compaction_failed not visible: {log_text!r}"
            )
            assert "summary_unavailable" in log_text


# ---------------------------------------------------------------------------
# P1-4: step.finished updates status only; parent run remains busy until run.finished
# ---------------------------------------------------------------------------


async def test_tui_step_finished_does_not_finish_parent_run_or_restore_prompt(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """step.finished must not behave like run.finished."""
    requests: list[dict[str, Any]] = []
    session_id = "session-step-only"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-step", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-step", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "run.started", "run_id": "run-step", "goal": "hi"},
                        {"type": "step.started", "run_id": "run-step", "step": 1},
                        {"type": "step.finished", "run_id": "run-step", "step": 1},
                    ):
                        event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                        writer.write(event_line.encode())
                    await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.3)

            status_text = str(app.query_one(RunStatusBlock).render())
            # P2-0.5: compact format — running status, step, tools, no "status" keyword
            assert "running" in status_text
            assert "step" in status_text
            assert "1 done" in status_text
            assert prompt.disabled
            assert app._busy

            log = app.query_one("#log", RichLog)
            log_text = "\n".join(line.text for line in log.lines)
            assert "step.started" not in log_text
            assert "step.finished" not in log_text
            assert "step 1 done" not in log_text


# ---------------------------------------------------------------------------
# Regression: send_message worker 抛非 IpcError 异常时，UI 必须显示错误并恢复 prompt
# ---------------------------------------------------------------------------


async def test_tui_send_message_recovers_from_non_ipc_exception(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """send_message worker 抛 RuntimeError（非 IpcError）时：
    - log 区显示错误信息（不能被 @work 静默吞掉）
    - prompt 恢复 enabled
    - busy 恢复（后续输入不被锁死）
    """
    requests: list[dict[str, Any]] = []
    session_id = "session-recover-123"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # 响应 event.subscribe + session.create
            for result in (
                {"subscription_id": "sub-recover", "replayed_count": 0},
                {"session_id": session_id},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            # 保持连接打开，但不再处理新请求
            await asyncio.sleep(5.0)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    # Skip real bootstrap; the TCP mock server plays the role of an already-running Core.
    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(
            server_version="0.0.1",
            uptime_ms=10,
            latency_ms=1,
        )
        return CoreStartOk(
            pid=0,
            host=host,
            port=port,
            ready=ready,
            already_running=True,
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)

            log = app.query_one("#log", RichLog)
            prompt = app.query_one("#prompt", TextArea)

            # P5-2.1: connected session id is not written to the chat timeline.
            assert not any(
                f"connected session={session_id}" in line.text for line in log.lines
            ), [line.text for line in log.lines]

            # 把 client.send_command 替换：对 session.send_message 抛非 IpcError 异常
            assert app._client is not None
            original_send = app._client.send_command

            async def failing_send_command(method: str, params: dict[str, Any]) -> dict[str, Any]:
                if method == "session.send_message":
                    raise RuntimeError("simulated non-IpcError failure")
                return await original_send(method, params)

            app._client.send_command = failing_send_command  # type: ignore[method-assign]

            # 输入 hello + Enter
            prompt.load_text("hello")
            await pilot.press("enter")
            await pilot.pause(0.5)

            # 验证：log 区显示错误（不能被吞掉）
            log_text = "\n".join(line.text for line in log.lines)
            assert (
                "simulated non-IpcError failure" in log_text
                or "disconnected" in log_text.lower()
            ), f"error not surfaced in log: {log_text}"

            # 验证：prompt 恢复 enabled
            assert not prompt.disabled, "prompt must be re-enabled after worker error"

            # 验证：busy 已恢复（再次输入不会被锁死）
            app._client.send_command = original_send  # type: ignore[method-assign]
            prompt.load_text("again")
            await pilot.press("enter")
            await pilot.pause(0.3)
            # user message 出现证明 on_input_submitted 没被 busy 拦截
            chat_blocks = list(app.query(ChatMessageBlock))
            user_blocks = [b for b in chat_blocks if b.role == "user"]
            assert any(b.content == "again" for b in user_blocks), [
                b.content for b in user_blocks
            ]


# ---------------------------------------------------------------------------
# Regression: connect_core worker 抛非 IpcError 异常时，UI 必须显示错误并恢复 prompt
# ---------------------------------------------------------------------------


async def test_tui_connect_core_recovers_from_non_ipc_exception(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect_core worker 抛非 IpcError 异常时（如 KeyError）：
    - log 区显示错误信息（不能被 @work 静默吞掉）
    """
    requests: list[dict[str, Any]] = []

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # 响应 event.subscribe 正常
            line = await reader.readline()
            if not line:
                return
            request = json.loads(line)
            requests.append(request)
            response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"subscription_id": "sub-x", "replayed_count": 0},
            }
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()

            # 响应 session.create 但 result 里没有 session_id → 触发 KeyError
            line = await reader.readline()
            if not line:
                return
            request = json.loads(line)
            requests.append(request)
            bad_response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {},  # 缺少 session_id
            }
            writer.write((json.dumps(bad_response) + "\n").encode())
            await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    # Skip real bootstrap
    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(
            server_version="0.0.1",
            uptime_ms=10,
            latency_ms=1,
        )
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async with server:
        async with app.run_test() as pilot:
            # Wait for bootstrap + connect_core to run and fail
            await pilot.pause(1.5)

            log = app.query_one("#log", RichLog)
            log_text = "\n".join(line.text for line in log.lines)

            # 错误必须出现在 log 中（KeyError 不能被 @work 静默吞掉）
            assert (
                "KeyError" in log_text
                or "error" in log_text.lower()
                or "disconnected" in log_text.lower()
            ), f"connect_core error not surfaced: {log_text}"


def _draining_handler(requests: list[dict[str, Any]], session_id: str) -> Any:
    """Build a TCP handler that responds to event.subscribe + session.create only."""

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-boot", "replayed_count": 0},
                {"session_id": session_id},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            await asyncio.sleep(5.0)
        finally:
            writer.close()
            await writer.wait_closed()

    return handle_client


async def test_tui_bootstrap_already_running_connects_directly(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """ensure_core_started_sync returns already_running → TUI 应直接 connect_core。"""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        return CoreStartOk(
            pid=0,
            host=host,
            port=port,
            ready=ready,
            already_running=True,
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-already-running"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp(
        "127.0.0.1",
        free_port,
        global_env_file=tmp_path / ".env",
    )

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)
            log = app.query_one("#log", RichLog)
            # P5-2.1: session id is not written into the chat timeline.
            assert not any(
                f"connected session={session_id}" in line.text for line in log.lines
            ), [line.text for line in log.lines]


async def test_tui_bootstrap_missing_key_enters_setup_mode(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """ensure_core_started_sync 返回 missing_deepseek_key 失败 → TUI 进入 setup mode。"""

    from my_agent.core.lifecycle import CoreStartFailed

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartFailed:
        return CoreStartFailed(
            host=host,
            port=port,
            reason="process_exited",
            exit_code=1,
            stderr_tail="DEEPSEEK_API_KEY not set\nRuntimeError",
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp(
        "127.0.0.1",
        free_port,
        global_env_file=tmp_path / ".env",
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.4)
        log = app.query_one("#log", RichLog)
        dialog = app.screen
        assert isinstance(dialog, SettingsDialog)
        model_picker = dialog.query_one("#settings-model", ModelSelect)
        key_input = dialog.query_one("#settings-key", Input)

        # setup mode 提示
        log_text = "\n".join(line.text for line in log.lines)
        assert "DeepSeek API key" in log_text
        assert "Configure the model" in log_text
        assert model_picker.display is False
        assert key_input.password is True
        # 内部状态：setup mode
        assert app._setup_mode is True


async def test_tui_setup_mode_saves_key_and_retries_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """setup mode 输入 key + Enter → 保存到 env_file + 重新 bootstrap。"""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartFailed, CoreStartOk

    env_file = tmp_path / ".env"
    ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)

    call_count = {"n": 0}

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CoreStartFailed(
                host=host,
                port=port,
                reason="process_exited",
                exit_code=1,
                stderr_tail="DEEPSEEK_API_KEY not set",
            )
        return CoreStartOk(
            pid=12345,
            host=host,
            port=port,
            ready=ready,
            already_running=False,
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-after-key"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp(
        "127.0.0.1",
        free_port,
        global_env_file=env_file,
    )

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)
            # 验证 setup mode 已激活
            assert app._setup_mode is True

            # 先输入 key + Enter，通过格式校验后进入模型选择
            settings_dialog = app.screen
            assert isinstance(settings_dialog, SettingsDialog)
            setup_prompt = settings_dialog.query_one("#settings-key", Input)
            setup_prompt.value = "sk-test-FAKE-1234"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.6)

            # env_file 已写入
            assert env_file.exists()
            content = env_file.read_text(encoding="utf-8")
            assert "DEEPSEEK_API_KEY=sk-test-FAKE-1234" in content

            # bootstrap 被再次调用
            assert call_count["n"] >= 2

            # setup mode 已退出
            assert app._setup_mode is False


async def test_tui_setup_mode_does_not_leak_key_to_log(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """setup mode 输入 key 后：log 区不能出现完整 key。"""
    from my_agent.core.lifecycle import CoreStartFailed

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartFailed:
        return CoreStartFailed(
            host=host,
            port=port,
            reason="process_exited",
            exit_code=1,
            stderr_tail="DEEPSEEK_API_KEY not set",
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    # 后续 save 也不要触发再次 bootstrap（用 flag 拦截）
    app = MyAgentTuiApp(
        "127.0.0.1",
        free_port,
        global_env_file=tmp_path / ".env",
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.4)
        assert app._setup_mode is True

        secret = "sk-LEAK-ME-12345678"
        settings_dialog = app.screen
        assert isinstance(settings_dialog, SettingsDialog)
        setup_prompt = settings_dialog.query_one("#settings-key", Input)
        setup_prompt.value = secret
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause(0.3)

        # 完整 key 不得出现在 log 区任何一行
        log_text = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert secret not in log_text, "API key leaked to log"
        # "API key saved" 这类提示允许出现，但不能带原始 key 值
        # 简单方式：log 区里若提到 key，最多只能出现 "saved" 字样


async def test_tui_bootstrap_non_key_failure_shows_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """bootstrap 失败但不是缺 key：TUI 显示 stderr 摘要和 my-agent-core 调试提示。"""

    from my_agent.core.lifecycle import CoreStartFailed

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartFailed:
        return CoreStartFailed(
            host=host,
            port=port,
            reason="process_exited",
            exit_code=2,
            stderr_tail="ImportError: something broke at startup",
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp(
        "127.0.0.1",
        free_port,
        global_env_file=tmp_path / ".env",
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.4)
        log_text = "\n".join(
            line.text for line in app.query_one("#log", RichLog).lines
        )
        # 不进 setup mode（不是缺 key）
        assert app._setup_mode is False
        # 显示错误摘要
        assert "ImportError" in log_text or "core failed to start" in log_text
        # 提示前台调试
        assert "my-agent-core" in log_text


# ---------------------------------------------------------------------------
# P2-0: PermissionSelect Test — Down + Enter submits selected decision
# ---------------------------------------------------------------------------


async def test_tui_permission_select_enter_submits_selected_decision(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PermissionSelect 打开后，按 Down + Enter → always_allow。"""
    requests: list[dict[str, Any]] = []
    session_id = "session-perm-enter"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # event.subscribe
            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"],
                 "result": {"subscription_id": "sub-ent", "replayed_count": 0}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            # session.create
            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"],
                 "result": {"session_id": session_id}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            # session.send_message
            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"],
                 "result": {"run_id": "run-ent", "status": "started"}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            # push events
            for event in (
                {"type": "run.started", "run_id": "run-ent", "goal": "hi"},
                {
                    "type": "permission.requested",
                    "run_id": "run-ent",
                    "tool_use_id": "tool-aa",
                    "tool_name": "write_file",
                    "param_preview": "path='src/foo.py'",
                },
            ):
                event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                writer.write(event_line.encode())
            await writer.drain()

            # read permission.respond (arrives after events are processed)
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                request = json.loads(line)
                requests.append(request)
                if request["method"] == "permission.respond":
                    payload = json.dumps(
                        {"jsonrpc": "2.0", "id": request["id"],
                         "result": {"ok": True}}
                    )
                    writer.write((payload + "\n").encode())
                    await writer.drain()
            except TimeoutError:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.4)

            permission_select = app.query_one(PermissionSelect)
            assert permission_select.display

            # PermissionSelect 获焦后，Down 移到 Always allow，再 Enter 提交。
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.3)

            perm_requests = [r for r in requests if r["method"] == "permission.respond"]
            assert len(perm_requests) == 1
            assert perm_requests[0]["params"]["tool_use_id"] == "tool-aa"
            assert perm_requests[0]["params"]["decision"] == "always_allow"
            assert not permission_select.display


# ---------------------------------------------------------------------------
# P2-0: PermissionSelect Test — shortcut key submits decision
# ---------------------------------------------------------------------------


async def test_tui_permission_select_shortcut_submits_decision(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按 n 快捷键 → permission.respond 发 deny_once；控件关闭。"""
    requests: list[dict[str, Any]] = []
    session_id = "session-perm-shortcut"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # event.subscribe
            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"],
                 "result": {"subscription_id": "sub-sc", "replayed_count": 0}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            # session.create
            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"],
                 "result": {"session_id": session_id}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            # session.send_message
            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"],
                 "result": {"run_id": "run-sc", "status": "started"}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            # push events
            for event in (
                {"type": "run.started", "run_id": "run-sc", "goal": "hi"},
                {
                    "type": "permission.requested",
                    "run_id": "run-sc",
                    "tool_use_id": "tool-deny",
                    "tool_name": "bash",
                    "param_preview": "command='rm -rf /'",
                },
            ):
                event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                writer.write(event_line.encode())
            await writer.drain()

            # read permission.respond (arrives after events are processed)
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                request = json.loads(line)
                requests.append(request)
                if request["method"] == "permission.respond":
                    payload = json.dumps(
                        {"jsonrpc": "2.0", "id": request["id"],
                         "result": {"ok": True}}
                    )
                    writer.write((payload + "\n").encode())
                    await writer.drain()
            except TimeoutError:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.4)

            permission_select = app.query_one(PermissionSelect)
            assert permission_select.display

            # PermissionSelect 获焦后，直接按 n 快捷键 → deny_once
            await pilot.press("n")
            await pilot.pause(0.3)

            perm_requests = [r for r in requests if r["method"] == "permission.respond"]
            assert len(perm_requests) == 1
            assert perm_requests[0]["params"]["tool_use_id"] == "tool-deny"
            assert perm_requests[0]["params"]["decision"] == "deny_once"
            assert not permission_select.display


async def test_tui_permission_select_works_when_textarea_has_focus(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permission keys must beat TextArea focus: Down + Enter sends permission.respond."""
    requests: list[dict[str, Any]] = []
    session_id = "session-perm-textarea-focus"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-ta-perm", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-ta-perm", "status": "started"},
            ):
                line = await reader.readline()
                request = json.loads(line)
                requests.append(request)
                payload = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
                writer.write((payload + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "run.started", "run_id": "run-ta-perm", "goal": "hi"},
                        {
                            "type": "permission.requested",
                            "run_id": "run-ta-perm",
                            "tool_use_id": "tool-textarea-focus",
                            "tool_name": "bash",
                            "param_preview": "command='pwd'",
                        },
                    ):
                        event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                        writer.write(event_line.encode())
                    await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.4)

            permission_select = app.query_one(PermissionSelect)
            assert permission_select.display

            prompt.focus()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.3)

            perm_requests = [r for r in requests if r["method"] == "permission.respond"]
            assert len(perm_requests) == 1
            assert perm_requests[0]["params"]["tool_use_id"] == "tool-textarea-focus"
            assert perm_requests[0]["params"]["decision"] == "always_allow"
            assert not permission_select.display

            log_text = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
            assert "permission decision" in log_text
            assert "bash" in log_text
            assert "always_allow" in log_text


async def test_tui_permission_select_options_are_not_covered_by_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permission options must be visible above the prompt, not covered by TextArea."""
    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp("127.0.0.1", 9)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)

        app._show_permission(
            PendingPermission(
                tool_use_id="tool-layout",
                tool_name="bash",
                param_preview="command='pwd'",
            )
        )
        await pilot.pause(0.1)

        permission_select = app.query_one(PermissionSelect)
        prompt = app.query_one("#prompt", TextArea)
        rendered = str(permission_select.render())

        assert "PERMISSION REQUIRED" in rendered or "permission required" in rendered.lower()
        assert "y" in rendered and "Allow once" in rendered
        assert "a" in rendered and "Always allow" in rendered
        assert "n" in rendered and "Deny once" in rendered
        assert "d" in rendered and "Always deny" in rendered
        assert permission_select.region.height >= 7
        assert permission_select.region.y + permission_select.region.height <= prompt.region.y


async def test_tui_permission_respond_uses_separate_connection_while_run_waits(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """permission.respond must not wait behind the in-flight session.send_message."""
    requests: list[dict[str, Any]] = []
    permission_seen = asyncio.Event()
    session_id = "session-perm-separate-conn"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            request = json.loads(line)
            requests.append(request)

            if request["method"] == "permission.respond":
                permission_seen.set()
                payload = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}
                )
                writer.write((payload + "\n").encode())
                await writer.drain()
                return

            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"subscription_id": "sub-sep", "replayed_count": 0},
                }
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"], "result": {"session_id": session_id}}
            )
            writer.write((payload + "\n").encode())
            await writer.drain()

            line = await reader.readline()
            request = json.loads(line)
            requests.append(request)
            assert request["method"] == "session.send_message"
            for event in (
                {"type": "run.started", "run_id": "run-sep", "goal": "hi"},
                {
                    "type": "permission.requested",
                    "run_id": "run-sep",
                    "tool_use_id": "tool-separate-conn",
                    "tool_name": "bash",
                    "param_preview": "command='pwd'",
                },
            ):
                event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                writer.write(event_line.encode())
            await writer.drain()

            await asyncio.wait_for(permission_seen.wait(), timeout=2.0)
            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"run_id": "run-sep", "status": "started"},
                }
            )
            writer.write((payload + "\n").encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.4)

            assert app.query_one(PermissionSelect).display
            await pilot.press("y")
            await asyncio.wait_for(permission_seen.wait(), timeout=2.0)

            perm_requests = [r for r in requests if r["method"] == "permission.respond"]
            assert len(perm_requests) == 1
            assert perm_requests[0]["params"]["tool_use_id"] == "tool-separate-conn"
            assert perm_requests[0]["params"]["decision"] == "allow_once"


async def test_tui_run_finished_clears_stale_permission_queue(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a run ends, stale active/queued permission prompts must disappear."""
    requests: list[dict[str, Any]] = []
    session_id = "session-stale-permission"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-stale", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-stale", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    return
                request = json.loads(line)
                requests.append(request)
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "run.started", "run_id": "run-stale", "goal": "hi"},
                        {
                            "type": "permission.requested",
                            "run_id": "run-stale",
                            "tool_use_id": "tool-stale-1",
                            "tool_name": "bash",
                            "param_preview": "command='pwd'",
                        },
                        {
                            "type": "permission.requested",
                            "run_id": "run-stale",
                            "tool_use_id": "tool-stale-2",
                            "tool_name": "bash",
                            "param_preview": "command='pwd'",
                        },
                        {
                            "type": "run.finished",
                            "run_id": "run-stale",
                            "status": "success",
                            "steps": 1,
                        },
                    ):
                        event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                        writer.write(event_line.encode())
                    await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.5)

            permission_select = app.query_one(PermissionSelect)
            assert not permission_select.display

            await pilot.press("y")
            await pilot.pause(0.2)
            assert not [r for r in requests if r["method"] == "permission.respond"]


# ---------------------------------------------------------------------------
# P2-0.5: Activity — status strip shows "working" immediately after submit
# ---------------------------------------------------------------------------


async def test_tui_shows_working_activity_immediately_after_submit(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用户提交消息后、Core 返回 run.started 前，状态栏显示 working。"""
    session_id = "session-activity"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # event.subscribe
            line = await reader.readline()
            request = json.loads(line)
            writer.write(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "id": request["id"],
                         "result": {"subscription_id": "sub-a", "replayed_count": 0}}
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()

            # session.create
            line = await reader.readline()
            request = json.loads(line)
            writer.write(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "id": request["id"],
                         "result": {"session_id": session_id}}
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()

            # session.send_message — delay before responding
            line = await reader.readline()
            request = json.loads(line)
            # respond immediately so TUI doesn't hang, but delay event push
            writer.write(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "id": request["id"],
                         "result": {"run_id": "run-activity", "status": "started"}}
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()

            # Delay event push so the test can check "working" state
            await asyncio.sleep(0.3)

            for event in (
                {"type": "run.started", "run_id": "run-activity", "goal": "hi"},
                {"type": "step.started", "run_id": "run-activity", "step": 1},
                {"type": "run.finished", "run_id": "run-activity", "status": "success", "steps": 1},
            ):
                event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                writer.write(event_line.encode())
            await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)

            prompt = app.query_one("#prompt", TextArea)
            log = app.query_one("#log", RichLog)
            prompt.load_text("hi")
            await pilot.press("enter")
            # Short pause — run.started hasn't arrived yet
            await pilot.pause(0.15)

            status_text = str(app.query_one(RunStatusBlock).render())
            assert "working" in status_text, (
                f"activity not shown before run.started: {status_text!r}"
            )
            assert prompt.disabled

            log_text = "\n".join(line.text for line in log.lines)
            assert "working" not in log_text

            # Wait for run.finished
            await pilot.pause(0.6)
            status_text = str(app.query_one(RunStatusBlock).render())
            assert "success" in status_text


# ---------------------------------------------------------------------------
# P2-0.6.1: CJK streaming preview wraps stably (no 1-3 char fragments)
# ---------------------------------------------------------------------------


async def test_tui_streaming_cjk_preview_wraps_stably_before_finished(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """长中文 streaming preview 不会产生 1-3 字碎行，且 final text 保持原文。"""
    requests: list[dict[str, Any]] = []
    session_id = "session-cjk-stable"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    # 300+ 字无换行中文文本
    long_text = "时光如水，岁月如歌，人生如梦，世事如棋。" * 20

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-cjk", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-cjk-stable", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
                writer.write((resp + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    # Send run.started + streaming CJK tokens in chunks
                    run_started = json.dumps({
                        "kind": "event",
                        "event": {"type": "run.started", "run_id": "run-cjk-stable",
                                  "goal": "test"},
                    }) + "\n"
                    writer.write(run_started.encode())

                    for chunk in (
                        long_text[:100],
                        long_text[100:200],
                        long_text[200:],
                    ):
                        token_ev = json.dumps({
                            "kind": "event",
                            "event": {"type": "llm.token", "run_id": "run-cjk-stable",
                                      "token": chunk},
                        }) + "\n"
                        writer.write(token_ev.encode())
                    await writer.drain()

            # Let TUI flush streaming preview before sending run.finished
            await asyncio.sleep(0.3)

            # Now send run.finished
            finished_ev = json.dumps({
                "kind": "event",
                "event": {"type": "run.finished", "run_id": "run-cjk-stable",
                          "status": "success", "steps": 1},
            }) + "\n"
            writer.write(finished_ev.encode())
            await writer.drain()
            await asyncio.sleep(0.3)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.2)

            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.1)

            log = app.query_one("#log", RichLog)

            # Before run.finished, do not show half-finished CJK/Markdown text.
            assert not any(
                "时光如水" in line.text or "岁月如歌" in line.text
                for line in log.lines
            )

            await pilot.pause(0.6)

            # After run.finished, collect the completed assistant lines.
            assistant_lines = [
                line.text for line in log.lines
                if "时光如水" in line.text or "岁月如歌" in line.text
            ]
            assert assistant_lines, "completed assistant text should be rendered"

            # No line should be a 1-3 char fragment (soft-wrap produces ~64-char lines)
            all_line_texts = "\n".join(assistant_lines).splitlines()
            non_empty = [ln for ln in all_line_texts if ln.strip()]
            short_lines = [ln for ln in non_empty if _display_width(ln) < 4]
            assert not short_lines, (
                f"found {len(short_lines)} ultra-short lines: {short_lines[:5]}"
            )

            # Verify LLMStreamBlock.text is raw original (not wrapped)
            block = app.query_one(LLMStreamBlock)
            assert block.text == long_text, (
                "LLMStreamBlock.text should be raw original text, "
                "not contaminated with soft-wrap newlines"
            )
            assert not block.display, (
                "raw LLMStreamBlock must stay hidden; visible streaming output "
                "is rendered through the stable RichLog preview line only"
            )


# ---------------------------------------------------------------------------
# TDD RED: two consecutive permission.requested events must queue, not overwrite
# ---------------------------------------------------------------------------


async def test_tui_queues_multiple_pending_permissions(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个连续的 permission.requested 进入队列，不互相覆盖，且各自响应发给正确的 tool_use_id。"""
    requests: list[dict[str, Any]] = []
    session_id = "session-perm-queue"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-perm-queue", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-perm-queue", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
                writer.write((resp + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    # Send run.started so TUI enters running state
                    run_started = json.dumps({
                        "kind": "event",
                        "event": {"type": "run.started",
                                  "run_id": "run-perm-queue", "goal": "hi"},
                    }) + "\n"
                    writer.write(run_started.encode())

                    # Push TWO permission.requested back-to-back
                    perm1 = json.dumps({
                        "kind": "event",
                        "event": {
                            "type": "permission.requested",
                            "run_id": "run-perm-queue",
                            "tool_use_id": "tool-read",
                            "tool_name": "read_file",
                            "param_preview": "path='README.md'",
                        },
                    }) + "\n"
                    writer.write(perm1.encode())

                    perm2 = json.dumps({
                        "kind": "event",
                        "event": {
                            "type": "permission.requested",
                            "run_id": "run-perm-queue",
                            "tool_use_id": "tool-bash",
                            "tool_name": "bash",
                            "param_preview": "command='pytest -q'",
                        },
                    }) + "\n"
                    writer.write(perm2.encode())
                    await writer.drain()

            # Now read permission.respond requests
            while True:
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}
                )
                writer.write((resp + "\n").encode())
                await writer.drain()

                # If we've handled 2 permission.respond calls, stop
                if sum(1 for r in requests if r["method"] == "permission.respond") >= 2:
                    break
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port,
                           ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.2)

            # Send user message to trigger session.send_message + permissions
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.4)

            # Assert: PermissionSelect shows FIRST request (read_file)
            permission_select = app.query_one(PermissionSelect)
            assert permission_select.display
            text = str(permission_select.render())
            assert "read_file" in text
            assert "README.md" in text
            assert "bash" not in text  # second should NOT appear yet

            # Respond "y" to first permission
            await pilot.press("y")
            await pilot.pause(0.3)

            # Verify first permission.respond sent to tool-read
            perm_requests = [r for r in requests if r["method"] == "permission.respond"]
            assert len(perm_requests) >= 1, (
                f"Expected at least 1 permission.respond, got {len(perm_requests)}"
            )
            assert perm_requests[0]["params"]["tool_use_id"] == "tool-read"
            assert perm_requests[0]["params"]["decision"] == "allow_once"

            # Assert: PermissionSelect now shows SECOND request (bash)
            assert permission_select.display
            text = str(permission_select.render())
            assert "bash" in text
            assert "pytest -q" in text
            assert "read_file" not in text  # first should be gone

            # Respond "n" to second via shortcut
            await pilot.press("n")
            await pilot.pause(0.3)

            # Verify second permission.respond sent to tool-bash
            perm_requests = [r for r in requests if r["method"] == "permission.respond"]
            assert len(perm_requests) >= 2
            assert perm_requests[1]["params"]["tool_use_id"] == "tool-bash"
            assert perm_requests[1]["params"]["decision"] == "deny_once"

            # Assert: PermissionSelect closed after queue is empty
            assert not permission_select.display

            # Assert: prompt restored
            prompt = app.query_one("#prompt", TextArea)
            assert not prompt.disabled
            assert prompt.has_focus


async def test_tui_ignores_foreign_run_events_when_idle(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle TUI must not render run/LLM events from another session/window."""
    session_id = "session-idle-window"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port,
                           ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-idle", "replayed_count": 0},
                {"session_id": session_id},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                payload = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
                writer.write((payload + "\n").encode())
                await writer.drain()

            for event in (
                {"type": "run.started", "run_id": "foreign-run", "goal": "hello"},
                {"type": "llm.token", "run_id": "foreign-run", "token": "Foreign"},
                {"type": "llm.token", "run_id": "foreign-run", "token": " answer"},
                {
                    "type": "run.finished",
                    "run_id": "foreign-run",
                    "status": "success",
                    "steps": 1,
                },
            ):
                writer.write((json.dumps({"kind": "event", "event": event}) + "\n").encode())
            await writer.drain()
            await asyncio.sleep(0.2)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.5)

            log = app.query_one("#log", RichLog)
            log_text = "\n".join(line.text for line in log.lines)
            assert "Foreign answer" not in log_text
            assert not list(app.query(LLMStreamBlock))


# ---------------------------------------------------------------------------
# P3-1: TextArea — Enter sends multi-line message
# ---------------------------------------------------------------------------


async def test_tui_textarea_enter_sends_multiline_message(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enter sends the complete multi-line TextArea content."""
    requests: list[dict[str, Any]] = []
    session_id = "session-multiline"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-ml", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-ml", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
                writer.write((resp + "\n").encode())
                await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port,
                           ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.2)

            prompt = app.query_one("#prompt", TextArea)
            # Load multi-line content
            prompt.load_text("line one\nline two\nline three")

            await pilot.press("enter")
            await pilot.pause(0.3)

            # Verify session.send_message was called with full multi-line content
            send_msgs = [r for r in requests if r["method"] == "session.send_message"]
            assert len(send_msgs) == 1
            assert send_msgs[0]["params"]["content"] == "line one\nline two\nline three"

            # Verify TextArea is cleared after send
            assert prompt.text == ""


async def test_tui_textarea_ctrl_enter_inserts_newline_without_sending(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl+Enter inserts a newline at the cursor without submitting."""
    requests: list[dict[str, Any]] = []
    session_id = "session-ctrl-j"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-cj", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-cj", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
                writer.write((resp + "\n").encode())
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port,
                           ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.2)

            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("line one")
            prompt.move_cursor((0, len("line one")))
            await pilot.press("ctrl+enter")
            await pilot.press(*"line two")
            await pilot.pause(0.1)

            send_msgs = [r for r in requests if r["method"] == "session.send_message"]
            assert send_msgs == []
            assert prompt.text == "line one\nline two"


# ---------------------------------------------------------------------------
# P3-1: TextArea — removed fallback keys do not submit
# ---------------------------------------------------------------------------


async def test_tui_textarea_ctrl_j_and_alt_enter_do_not_send(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl+J and Alt+Enter no longer submit prompt content."""
    requests: list[dict[str, Any]] = []
    session_id = "session-enter-noop"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-en", "replayed_count": 0},
                {"session_id": session_id},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                resp = json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
                writer.write((resp + "\n").encode())
                await writer.drain()
            await asyncio.sleep(0.3)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port,
                           ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.2)

            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("do not send")
            await pilot.press("ctrl+j")
            await pilot.pause(0.1)
            send_msgs = [r for r in requests if r["method"] == "session.send_message"]
            assert send_msgs == []

            prompt.load_text("still do not send")
            await pilot.press("alt+enter")
            await pilot.pause(0.1)
            send_msgs = [r for r in requests if r["method"] == "session.send_message"]
            assert send_msgs == []


# ---------------------------------------------------------------------------
# P5-1: Startup lifecycle, header, prompt border title, terminal title
# ---------------------------------------------------------------------------


async def test_tui_startup_state_starts_starting_and_prompt_disabled(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """At startup, before bootstrap completes, prompt must be disabled and
    _startup_state must be a non-ready state (starting/connecting)."""

    started = asyncio.Event()

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> Any:
        started.set()
        # Hold bootstrap open so the test can observe the in-flight state.
        await asyncio.sleep(0.5)
        from my_agent.cli.commands.core import CoreProbeResult
        from my_agent.core.lifecycle import CoreStartOk

        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp(
        "127.0.0.1",
        free_port,
        global_env_file=tmp_path / ".env",
    )

    async with app.run_test() as pilot:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await pilot.pause(0.05)

        prompt = app.query_one("#prompt", TextArea)
        assert prompt.disabled, "prompt must stay disabled during connecting"
        # State must not be ready yet.
        assert app._startup_state != "ready"
        assert app._startup_state in {"starting", "connecting", "creating_session"}

        startup = app.query_one("#startup-panel", StartupPanel)
        assert startup.display
        startup_text = str(startup.render()).lower()
        assert "my agent" in startup_text
        assert "loading" in startup_text
        assert "connecting" in startup_text


async def test_tui_startup_panel_shows_welcome_after_session_is_ready(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
) -> None:
    """Ready with no chat shows a welcome banner while RichLog holds layout space."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-startup-panel"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            startup = app.query_one("#startup-panel", StartupPanel)
            assert startup.display
            welcome_text = str(startup.render())
            assert "MY AGENT" in welcome_text
            assert "██" in welcome_text
            assert "Welcome" not in welcome_text
            assert app.query_one("#log", RichLog).display
            prompt = app.query_one("#prompt", TextArea)
            assert prompt.display
            assert prompt.region.y > startup.region.y + startup.region.height


async def test_tui_setup_error_state_when_bootstrap_fails_non_key(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """Bootstrap fails with non-key error → _startup_state becomes setup_error,
    prompt stays disabled, and diagnostics surface in the chat log."""

    from my_agent.core.lifecycle import CoreStartFailed

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartFailed:
        return CoreStartFailed(
            host=host,
            port=port,
            reason="process_exited",
            exit_code=2,
            stderr_tail="ImportError: something broke at startup",
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp(
        "127.0.0.1",
        free_port,
        global_env_file=tmp_path / ".env",
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.4)

        # State must be setup_error (NOT ready, NOT connecting anymore).
        assert app._startup_state == "setup_error"
        prompt = app.query_one("#prompt", TextArea)
        assert prompt.disabled, "prompt must stay disabled in setup_error state"

        log_text = "\n".join(
            line.text for line in app.query_one("#log", RichLog).lines
        )
        assert "ImportError" in log_text or "core failed to start" in log_text


async def test_tui_header_shows_project_session_state_when_ready(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On successful connect, AppHeader must show brand, project name,
    connected status, and short session id."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-header-test-12345678"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            header = app.query_one(AppHeader)
            rendered = str(header.render())
            assert "MY AGENT" in rendered or "My Agent" in rendered
            assert "connected" in rendered.lower()
            # P5-2.1: session id is intentionally kept out of the default ready UI.
            assert session_id[-8:] not in rendered
            assert "session" not in rendered.lower()
            # Project name should be the cwd's last path segment.
            project_name = Path.cwd().name
            assert project_name in rendered

            # App title (terminal title) must reflect ready state with project.
            assert app.title == f"My Agent - {project_name}"


async def test_tui_prompt_border_title_reflects_lifecycle_states(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt border title must change across: connecting → ready → running."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-border-title"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            prompt = app.query_one("#prompt", TextArea)
            # At ready: border title must invite typing.
            assert "type message" in (prompt.border_title or "").lower()

            # Submit a message to flip into running state.
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert "working" in (prompt.border_title or "").lower() or (
                app._startup_state == "running"
            )


async def test_tui_terminal_title_reflects_permission_and_disconnected(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When permission is requested, terminal title must say 'permission required'.
    When disconnected, it must say 'disconnected'."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    app = MyAgentTuiApp("127.0.0.1", free_port)
    project_name = Path.cwd().name

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)

        # Force permission state directly via the helper.
        app._show_permission(
            PendingPermission(
                tool_use_id="tool-title",
                tool_name="bash",
                param_preview="command='ls'",
            )
        )
        await pilot.pause(0.05)
        assert app._startup_state == "waiting_permission"
        assert app.title == f"My Agent - {project_name} - permission required"

        # Now simulate disconnect.
        app._update_startup_state("disconnected")
        await pilot.pause(0.05)
        assert app.title == f"My Agent - {project_name} - disconnected"


async def test_tui_welcome_empty_state_when_first_ready(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First transition to ready with no messages shows a welcome line
    containing project name and slash command hints."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-welcome"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            startup = app.query_one("#startup-panel", StartupPanel)
            rendered = str(startup.render())
            project_name = Path.cwd().name
            assert project_name in rendered, (
                f"welcome banner missing project name: {rendered!r}"
            )
            # Welcome must mention at least one of the documented slash commands.
            assert "/init" in rendered or "/help" in rendered or "/compact" in rendered
            # Infrastructure low-level messages must NOT appear in chat timeline.
            log_text = "\n".join(
                line.text for line in app.query_one("#log", RichLog).lines
            )
            assert "checking core at" not in log_text
            assert "connecting to 127.0.0.1" not in log_text


async def test_tui_running_state_input_restriction_not_regressed(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """During running state, prompt must be disabled. After run.finished,
    prompt must be re-enabled and state must return to ready."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-running-state"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-run", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-running", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    return
                request = json.loads(line)
                requests.append(request)
                writer.write(
                    (
                        json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result})
                        + "\n"
                    ).encode()
                )
                await writer.drain()

                if request["method"] == "session.send_message":
                    # Delay briefly so the test can observe the running state
                    # before run.finished arrives and resets it to ready.
                    await asyncio.sleep(0.3)
                    for event in (
                        {"type": "run.started", "run_id": "run-running", "goal": "hi"},
                        {"type": "run.finished", "run_id": "run-running",
                         "status": "success", "steps": 1},
                    ):
                        writer.write(
                            (json.dumps({"kind": "event", "event": event}) + "\n").encode()
                        )
                    await writer.drain()
            await asyncio.sleep(0.3)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.15)

            # During run: state=running, prompt disabled.
            assert app._startup_state == "running"
            assert prompt.disabled

            await pilot.pause(0.4)

            # After run.finished: state=ready, prompt enabled.
            assert app._startup_state == "ready"
            assert not prompt.disabled


async def test_tui_init_remains_client_side_intercepted(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """/init must continue to be intercepted on the client side and NOT
    be sent as a daemon session.send_message request."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)
    monkeypatch.chdir(tmp_path)

    requests: list[dict[str, Any]] = []
    session_id = "session-init-intercept"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("/init")
            await pilot.press("escape")  # close any slash palette
            await pilot.pause(0.05)
            await pilot.press("enter")
            await pilot.pause(0.3)

            # /init was intercepted locally → workspace files exist
            assert (tmp_path / ".my-agent" / "context.md").is_file()
            assert (tmp_path / "AGENTS.md").is_file()
            # No session.send_message with content == "/init" was sent.
            init_requests = [
                r for r in requests
                if r["method"] == "session.send_message"
                and r["params"].get("content") == "/init"
            ]
            assert init_requests == []


async def test_tui_narrow_terminal_with_cjk_project_name_renders_without_crash(
    monkeypatch: pytest.MonkeyPatch,
    free_port: int,
    tmp_path: Path,
) -> None:
    """In a narrow terminal with a CJK project name, AppHeader must render
    without crashing or producing unreadable overlap."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    cjk_dir = tmp_path / "我的项目目录"
    cjk_dir.mkdir()
    monkeypatch.chdir(cjk_dir)

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-cjk-narrow"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        # size=(40, 24) simulates a very narrow terminal.
        async with app.run_test(size=(40, 24)) as pilot:
            await pilot.pause(0.4)

            # Header must still exist and render without raising.
            header = app.query_one(AppHeader)
            rendered = str(header.render())
            assert "MY AGENT" in rendered or "My Agent" in rendered

            # App title must contain the CJK name.
            assert "我的项目目录" in app.title

            # Prompt border title must be set (not crashed).
            prompt = app.query_one("#prompt", TextArea)
            assert prompt.border_title  # not empty

            # Header must fit within terminal width — single line, no wrap into many lines.
            # We accept up to 2 visual rows after CJK wrapping for the header content.
            header_height = header.region.height
            assert 1 <= header_height <= 4, (
                f"header height out of expected range: {header_height}"
            )


async def test_tui_disconnected_state_input_restriction(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the app enters disconnected state, prompt border title must
    reflect it. Sending a message must not crash and must not leak to daemon."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-disc"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            app._update_startup_state("disconnected")
            await pilot.pause(0.05)

            prompt = app.query_one("#prompt", TextArea)
            assert "waiting for core" in (prompt.border_title or "").lower()
            assert app.title.endswith("- disconnected")


async def test_tui_low_level_bootstrap_logs_do_not_leak_to_chat_timeline(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'checking core at host:port' and 'connecting to host:port' lines
    must NOT appear in the chat timeline anymore. They are infrastructure
    status, now surfaced through the AppHeader instead."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(
            pid=0, host=host, port=port, ready=ready, already_running=True
        )

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-no-lowlevel-logs"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            log_text = "\n".join(
                line.text for line in app.query_one("#log", RichLog).lines
            )
            # Low-level infrastructure status — removed from chat timeline.
            assert "checking core at" not in log_text
            assert "connecting to 127.0.0.1" not in log_text
            # P5-2.1: session id is also removed from the default chat timeline.
            assert "connected session=" not in log_text


# ---------------------------------------------------------------------------
# P5-2: TUI visual polish / product coherence
# ---------------------------------------------------------------------------


async def test_tui_startup_panel_shows_brand_project_and_phase(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """StartupPanel 在启动期间展示品牌、项目名和当前阶段，不含低层地址信息。"""
    from my_agent.core.lifecycle import CoreStartOk

    call_count = {"n": 0}

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        call_count["n"] += 1
        # 第一次返回 OK 后故意不推进到 session.create，让启动面板停留在 connecting。
        from my_agent.cli.commands.core import CoreProbeResult

        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    # 不响应 session.create，让 TUI 停在 CREATING_SESSION 状态。
    async def hanging_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if line:
                request = json.loads(line)
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"subscription_id": "sub-startup", "replayed_count": 0},
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(hanging_handler, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)

            startup = app.query_one(StartupPanel)
            assert startup.display
            rendered = str(startup.render())
            assert "MY AGENT" in rendered
            assert "my-agent" in rendered or app._project_name_raw in rendered
            assert any(
                phase in rendered.lower()
                for phase in ("starting", "connecting", "opening session", "setup required")
            )
            assert "127.0.0.1" not in rendered
            assert ":7437" not in rendered
            assert "socket" not in rendered.lower()
            assert "rpc" not in rendered.lower()


async def test_tui_ready_header_shows_brand_project_session_and_state(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """进入 READY 后，AppHeader 紧凑显示品牌、项目、连接状态、短 session id 和 run 状态。"""
    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-ready-header"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            header = app.query_one(AppHeader)
            rendered = str(header.render())
            assert "MY AGENT" in rendered
            assert app._project_name_raw in rendered
            assert "connected" in rendered.lower()
            # P5-2.1: default ready UI must not show the session id.
            assert session_id[-8:] not in rendered
            assert "session" not in rendered.lower()
            assert "ready" in rendered.lower()
            assert "127.0.0.1" not in rendered
            assert ":7437" not in rendered
            assert "socket" not in rendered.lower()
            assert "rpc" not in rendered.lower()


async def test_tui_prompt_border_title_follows_lifecycle_states(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prompt 的 border_title 会随 starting / ready / running / disconnected 变化。"""
    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-border-title"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)
            prompt = app.query_one("#prompt", TextArea)
            ready_title = prompt.border_title or ""
            assert ready_title and "type" in ready_title.lower()

            app._update_startup_state("running")
            await pilot.pause(0.05)
            assert "working" in (prompt.border_title or "").lower()

            app._update_startup_state("disconnected")
            await pilot.pause(0.05)
            assert "waiting for core" in (prompt.border_title or "").lower()


async def test_tui_welcome_empty_state_keeps_project_and_slash_commands(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次 READY 后，welcome banner 展示在 StartupPanel 里，RichLog 留出内容区。"""
    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-welcome"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)
            startup = app.query_one(StartupPanel)
            assert startup.display
            rendered = str(startup.render())
            assert app._project_name_raw in rendered
            assert "/init" in rendered
            assert "/compact" in rendered
            assert "/tools" in rendered
            assert "/copy" in rendered
            assert "MY AGENT" in rendered
            assert "██" in rendered
            assert "Welcome" not in rendered
            # P5-2.1: welcome banner is not a RichLog line; RichLog stays visible
            # as the empty content area so the prompt remains bottom anchored.
            assert app.query_one("#log", RichLog).display
            prompt = app.query_one("#prompt", TextArea)
            assert prompt.display
            assert prompt.region.y > startup.region.y + startup.region.height
            # No session id in the welcome banner.
            assert session_id not in rendered


async def test_tui_copy_last_exchange_reports_clipboard_failure(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/copy reports a clear failure when the system clipboard is unavailable."""
    requests: list[dict[str, Any]] = []
    session_id = "session-copy-last"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-copy", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-copy", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": result,
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "llm.token", "run_id": "run-copy", "token": "Answer"},
                        {"type": "llm.token", "run_id": "run-copy", "token": " here."},
                        {"type": "run.finished", "run_id": "run-copy", "status": "success"},
                    ):
                        ev = {"kind": "event", "event": event}
                        writer.write((json.dumps(ev) + "\n").encode())
                    await writer.drain()
            await asyncio.sleep(5.0)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.5)

            log = app.query_one("#log", RichLog)
            assert any("Answer here." in line.text for line in log.lines)

            def _raise(_text: str) -> None:
                raise RuntimeError("no clipboard")

            monkeypatch.setattr(app, "_copy_to_system_clipboard", _raise)
            prompt.load_text("/copy")
            await pilot.press("enter")
            await pilot.pause(0.3)

            log_text = "\n".join(line.text for line in log.lines)
            assert "copy failed" in log_text
            assert "last question and answer" in log_text


async def test_tui_copy_last_exchange_includes_user_and_assistant_lines(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/copy sends the latest user question and assistant answer together."""
    requests: list[dict[str, Any]] = []
    session_id = "session-copy-all"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-copy-all", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-copy-all", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": result,
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "llm.token", "run_id": "run-copy-all", "token": "Got it."},
                        {"type": "run.finished", "run_id": "run-copy-all", "status": "success"},
                    ):
                        ev = {"kind": "event", "event": event}
                        writer.write((json.dumps(ev) + "\n").encode())
                    await writer.drain()
            await asyncio.sleep(5.0)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("please summarize")
            await pilot.press("enter")
            await pilot.pause(0.5)

            copied: list[str] = []

            def _capture(text: str) -> None:
                copied.append(text)

            monkeypatch.setattr(app, "_copy_to_system_clipboard", _capture)
            prompt.load_text("/copy")
            await pilot.press("enter")
            await pilot.pause(0.3)

            assert len(copied) == 1
            assert "User:\nplease summarize" in copied[0]
            assert "Assistant:\nGot it." in copied[0]


async def test_tui_copy_ignores_tool_details_and_copies_exchange(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/copy exports the exchange, not the latest tool details."""
    requests: list[dict[str, Any]] = []
    session_id = "session-copy-tool"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-copy-tool", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-copy-tool", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": result,
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {
                            "type": "tool.call_started",
                            "run_id": "run-copy-tool",
                            "tool_use_id": "tool-copy-1",
                            "tool_name": "bash",
                            "params": {"command": "bad"},
                        },
                        {
                            "type": "tool.call_failed",
                            "run_id": "run-copy-tool",
                            "tool_use_id": "tool-copy-1",
                            "tool_name": "bash",
                            "elapsed_ms": 5,
                            "error_message": "not found",
                        },
                        {
                            "type": "llm.token",
                            "run_id": "run-copy-tool",
                            "token": "Please retry.",
                        },
                        {"type": "run.finished", "run_id": "run-copy-tool", "status": "failed"},
                    ):
                        ev = {"kind": "event", "event": event}
                        writer.write((json.dumps(ev) + "\n").encode())
                    await writer.drain()
            await asyncio.sleep(5.0)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("run bad command")
            await pilot.press("enter")
            await pilot.pause(0.2)

            copied: list[str] = []

            def _capture(text: str) -> None:
                copied.append(text)

            monkeypatch.setattr(app, "_copy_to_system_clipboard", _capture)
            prompt.load_text("/copy")
            await pilot.press("enter")
            await pilot.pause(0.3)

            assert len(copied) == 1
            assert "User:\nrun bad command" in copied[0]
            assert "Assistant:\nPlease retry." in copied[0]
            assert "not found" not in copied[0]


async def test_tui_copy_command_shows_visible_hint_when_clipboard_unavailable(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """系统剪贴板不可用时，/copy 会留下明确的可见失败提示。"""
    requests: list[dict[str, Any]] = []
    session_id = "session-copy-hint"

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-copy-hint", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-copy-hint", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": result,
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "llm.token", "run_id": "run-copy-hint", "token": "hint."},
                        {"type": "run.finished", "run_id": "run-copy-hint", "status": "success"},
                    ):
                        ev = {"kind": "event", "event": event}
                        writer.write((json.dumps(ev) + "\n").encode())
                    await writer.drain()
            await asyncio.sleep(5.0)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.5)

            def _raise(_text: str) -> None:
                raise RuntimeError("no clipboard")

            monkeypatch.setattr(app, "_copy_to_system_clipboard", _raise)

            prompt.load_text("/copy")
            await pilot.press("enter")
            await pilot.pause(0.3)

            log_text = "\n".join(
                line.text for line in app.query_one("#log", RichLog).lines
            )
            assert "clipboard" in log_text.lower() or "copied" in log_text.lower()


# ---------------------------------------------------------------------------
# P5-2.1: TUI Banner / Ready State acceptance tests
# ---------------------------------------------------------------------------


async def test_tui_loading_stage_hides_richlog_and_prompt(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading (connecting/creating_session) should hide RichLog and
    the disabled prompt; only the StartupPanel is visible."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    # Do NOT respond to session.create — leave the TUI in CREATING_SESSION.
    async def hanging_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if line:
                request = json.loads(line)
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"subscription_id": "sub-load-test", "replayed_count": 0},
                }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(hanging_handler, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            # StartupPanel must be showing the phase banner.
            startup = app.query_one(StartupPanel)
            assert startup.display
            assert "MY AGENT" in str(startup.render())

            # RichLog must be hidden.
            log = app.query_one("#log", RichLog)
            assert not log.display

            # Prompt must be hidden (disabled input not visible).
            prompt = app.query_one("#prompt", TextArea)
            assert not prompt.display

            # RunStatusBlock must not show idle debug info.
            status = app.query_one(RunStatusBlock)
            assert not status.display


async def test_tui_first_user_message_hides_welcome_banner(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the first user message, the welcome banner must hide and the
    RichLog must become visible."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-banner-hide"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            startup = app.query_one(StartupPanel)
            assert startup.display
            assert "MY AGENT" in str(startup.render())
            assert "██" in str(startup.render())

            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hello")
            await pilot.press("enter")
            await pilot.pause(0.3)

            # Welcome banner must be hidden.
            assert not startup.display, "welcome banner must hide after first message"
            # RichLog must be visible.
            log = app.query_one("#log", RichLog)
            assert log.display
            assert any("hello" in line.text for line in log.lines)


async def test_tui_run_status_block_hidden_in_idle_and_ready(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RunStatusBlock must not be visible at idle/ready; it surfaces only
    when a run or activity is in progress."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-run-status"
    server = await asyncio.start_server(
        _draining_handler(requests, session_id), "127.0.0.1", free_port
    )
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.4)

            status = app.query_one(RunStatusBlock)
            # Idle/ready — not visible.
            assert not status.display

            # Simulate a run via direct start.
            status.start("run-visibility-test")
            assert status.display

            status.finish("success")
            assert not status.display


async def test_tui_skill_tool_compatibility_event_renders_compact_diagnostic(
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P6: skill.tool_compatibility event with aliases and unresolved tools must
    render a compact visible diagnostic line in the chat timeline."""

    from my_agent.cli.commands.core import CoreProbeResult
    from my_agent.core.lifecycle import CoreStartOk

    async def fake_ensure(host: str, port: int, **kwargs: Any) -> CoreStartOk:
        ready = CoreProbeResult(server_version="0.0.1", uptime_ms=10, latency_ms=1)
        return CoreStartOk(pid=0, host=host, port=port, ready=ready, already_running=True)

    monkeypatch.setattr("my_agent.tui.app.ensure_core_started", fake_ensure)

    requests: list[dict[str, Any]] = []
    session_id = "session-skill-compat"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            for result in (
                {"subscription_id": "sub-x", "replayed_count": 0},
                {"session_id": session_id},
                {"run_id": "run-skill", "status": "started"},
            ):
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line)
                requests.append(request)
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

                if request["method"] == "session.send_message":
                    for event in (
                        {"type": "run.started", "run_id": "run-skill", "goal": "hi"},
                        {
                            "type": "skill.tool_compatibility",
                            "skill_name": "review",
                            "run_id": "run-skill",
                            "session_id": session_id,
                            "resolved_tools": ["bash", "read_file"],
                            "aliases": [
                                {"from": "shell", "to": "bash"},
                                {"from": "file.read", "to": "read_file"},
                            ],
                            "unresolved_tools": ["unknown_tool"],
                        },
                        {
                            "type": "run.finished",
                            "run_id": "run-skill",
                            "status": "success",
                            "steps": 1,
                        },
                    ):
                        event_line = json.dumps({"kind": "event", "event": event}) + "\n"
                        writer.write(event_line.encode())
                    await writer.drain()
            await asyncio.sleep(0.5)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    app = MyAgentTuiApp("127.0.0.1", free_port)

    async with server:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            prompt = app.query_one("#prompt", TextArea)
            prompt.load_text("hi")
            await pilot.press("enter")
            await pilot.pause(0.5)

            log = app.query_one("#log", RichLog)
            log_text = "\n".join(line.text for line in log.lines)

            # Skill name + mapped alias + missing tool must all be visible.
            assert "review" in log_text, f"skill name missing: {log_text!r}"
            assert "shell" in log_text and "bash" in log_text, (
                f"alias mapping missing: {log_text!r}"
            )
            assert "file.read" in log_text and "read_file" in log_text, (
                f"capability alias missing: {log_text!r}"
            )
            assert "unknown_tool" in log_text, (
                f"unresolved tool missing: {log_text!r}"
            )
            # Must not leak low-level host/port/RPC details.
            assert "127.0.0.1" not in log_text
            assert ":7437" not in log_text
