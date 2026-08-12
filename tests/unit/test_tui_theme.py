from __future__ import annotations

from my_agent.tui.app import (
    AppHeader,
    ChatMessageBlock,
    EventLineBlock,
    MyAgentTuiApp,
    PermissionSelect,
    PromptTextArea,
    SlashCommand,
    SlashPalette,
    StartupPanel,
    ToolCallBlock,
)

# ---------------------------------------------------------------------------
# P5-2: visual language / theme unit checks
# ---------------------------------------------------------------------------


def test_app_css_contains_theme_tokens_and_scrollbar_rules() -> None:
    """全局 CSS 必须声明近黑底、热粉强调、青蓝信号，并覆盖滚动条颜色。"""
    css = MyAgentTuiApp.CSS
    assert "#0d0d0d" in css or "$background" in css
    assert "#ff149d" in css
    assert "#00e5ff" in css
    assert "scrollbar-background" in css
    assert "scrollbar-color" in css or "scrollbar:" in css


def test_app_css_does_not_use_default_blue_scrollbar() -> None:
    """滚动条相关规则不能保留 Textual 默认的纯蓝色。"""
    css_lower = MyAgentTuiApp.CSS.lower()
    scrollbar_section = css_lower[css_lower.find("scrollbar") :]
    # 主题范围内不应出现默认蓝色（如果用户显式写了 #0000ff 也拒绝）。
    assert "#0000ff" not in scrollbar_section
    assert "blue" not in scrollbar_section


def test_startup_panel_render_includes_brand_project_and_phase() -> None:
    panel = StartupPanel()
    panel.set_state("starting", project_name="my-agent")
    text = str(panel.render())
    assert "MY AGENT" in text
    assert "my-agent" in text
    assert "starting" in text.lower()


def test_startup_panel_render_truncates_long_cjk_project_name() -> None:
    panel = StartupPanel()
    long_name = "我" * 80
    panel.set_state("connecting", project_name=long_name)
    text = str(panel.render())
    assert "MY AGENT" in text
    # 不应在单行条目中塞入 80 个 CJK 字符导致布局爆炸。
    assert len(text) < 300


def test_startup_panel_welcome_uses_big_banner_not_plain_welcome_line() -> None:
    panel = StartupPanel()
    panel.show_welcome(project_name="my-agent")

    text = str(panel.render())

    assert "MY AGENT" in text
    assert "my-agent" in text
    assert "type a message" in text
    assert "Welcome" not in text
    assert "██" in text


def test_header_render_includes_brand_connection_run_state() -> None:
    header = AppHeader()
    header.set_state(
        project_name="my-agent",
        connection_label="connected",
        session_short="abc12345",
        run_state_label="ready",
    )
    text = str(header.render())
    assert "MY AGENT" in text
    assert "my-agent" in text
    assert "connected" in text
    assert "ready" in text.lower()
    # P5-2.1: session id is intentionally kept out of the default ready UI.
    assert "abc12345" not in text
    assert "session" not in text.lower()


def test_header_render_does_not_leak_low_level_details() -> None:
    header = AppHeader()
    header.set_state(
        project_name="my-agent",
        connection_label="connected",
        session_short="abc12345",
        run_state_label="ready",
    )
    text = str(header.render())
    assert "127.0.0.1" not in text
    assert ":7437" not in text
    assert "socket" not in text.lower()
    assert "rpc" not in text.lower()
    assert "daemon" not in text.lower()


def test_chat_message_block_uses_consistent_label() -> None:
    user_block = ChatMessageBlock("user", "hello")
    text = str(user_block.render())
    assert "YOU //" in text
    assert "hello" in text

    assistant_block = ChatMessageBlock("assistant", "hi there")
    text = str(assistant_block.render())
    assert "AGENT //" in text
    assert "hi there" in text


def test_event_line_block_uses_consistent_tag() -> None:
    block = EventLineBlock("permission", "read_file path='README.md'")
    text = str(block.render())
    assert "permission" in text.lower()
    assert "read_file" in text
    assert "README.md" in text


def test_tool_call_block_collapsed_line_uses_status_style() -> None:
    block = ToolCallBlock("read_file", {"path": "README.md"})
    text = str(block.render())
    assert "TOOL" in text or "tool" in text.lower()
    assert "read_file" in text
    assert "running" in text.lower()


def test_tool_call_block_expanded_shows_params_and_error() -> None:
    block = ToolCallBlock("bash", {"command": "bad"})
    block.set_result(12, is_error=True, error_message="not found")
    block.toggle_details()
    text = str(block.render())
    assert "details" in text.lower()
    assert "params" in text.lower()
    assert "error" in text.lower()
    assert "not found" in text


def test_slash_palette_render_looks_like_command_menu() -> None:
    palette = SlashPalette()
    commands = [
        SlashCommand(command="/tools", insert_text="/tools", description="toggle details"),
        SlashCommand(command="/init", insert_text="/init", description="init project"),
    ]
    palette.open(commands, "")
    text = str(palette.render())
    assert "/tools" in text
    assert "toggle details" in text
    assert ">" in text

def test_permission_select_render_highlights_confirmation() -> None:
    ps = PermissionSelect()
    ps.open("write_file", "path='src/foo.py'", queued_count=2)
    text = str(ps.render())
    assert "PERMISSION" in text.upper()
    assert "write_file" in text
    assert "src/foo.py" in text
    assert "2" in text


def test_prompt_text_area_undo_does_not_escape_on_history_error(
    monkeypatch,
) -> None:
    prompt = PromptTextArea(id="prompt")

    def _raise() -> None:
        raise RuntimeError("malformed edit history")

    monkeypatch.setattr(prompt, "undo", _raise)
    prompt.action_undo()


def test_prompt_text_area_undo_on_empty_input_is_a_noop() -> None:
    prompt = PromptTextArea(id="prompt")

    prompt.action_undo()

    assert prompt.text == ""


def test_prompt_text_area_undo_reverts_input_normally() -> None:
    prompt = PromptTextArea(id="prompt")
    prompt.insert("hello")

    prompt.action_undo()

    assert prompt.text == ""
