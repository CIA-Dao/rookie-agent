from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from my_agent.cli.commands import chat as chat_command
from my_agent.core.config import Config


def test_cmd_chat_passes_provided_config_to_chat(
    monkeypatch: MonkeyPatch,
) -> None:
    config = Config(host="127.0.0.9", port=8765)
    received: list[Config] = []

    async def fake_chat(cfg: Config) -> int:
        received.append(cfg)
        return 7

    monkeypatch.setattr(chat_command, "_chat", fake_chat)

    with pytest.raises(SystemExit) as exc_info:
        chat_command.cmd_chat(config)

    assert exc_info.value.code == 7
    assert received == [config]


def test_cmd_chat_connection_failure_prints_actionable_hint(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_chat(config: Config) -> int:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(chat_command, "_chat", fake_chat)

    with pytest.raises(SystemExit) as exc_info:
        chat_command.cmd_chat(Config(host="127.0.0.9", port=8765))

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: cannot connect to my-agent core at 127.0.0.9:8765" in captured.err
    assert "my-agent core status" in captured.err
    assert "my-agent core start" in captured.err


def test_local_slash_command_init_is_handled_locally(tmp_path: Path) -> None:
    """/init 由客户端本地处理，不发送到 daemon。"""
    # helper 尚未实现时会先报错——RED 阶段
    from my_agent.cli.commands.chat import _try_handle_local_slash_command

    result = _try_handle_local_slash_command("/init", tmp_path)
    assert result is True
    # 验证文件确实被创建
    assert (tmp_path / ".my-agent" / "context.md").is_file()
    assert (tmp_path / "AGENTS.md").is_file()


def test_local_slash_command_compact_is_not_local(tmp_path: Path) -> None:
    """/compact 不是本地命令，应由 daemon 处理。"""
    from my_agent.cli.commands.chat import _try_handle_local_slash_command

    result = _try_handle_local_slash_command("/compact focus", tmp_path)
    assert result is False


def test_local_slash_command_plain_text_is_not_local(tmp_path: Path) -> None:
    """普通消息不是本地斜杠命令。"""
    from my_agent.cli.commands.chat import _try_handle_local_slash_command

    result = _try_handle_local_slash_command("hello world", tmp_path)
    assert result is False
