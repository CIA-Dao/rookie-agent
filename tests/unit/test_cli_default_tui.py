from __future__ import annotations

import sys

import pytest
from pytest import MonkeyPatch

import my_agent.cli.main as cli_main


def test_main_no_args_invokes_tui_main(monkeypatch: MonkeyPatch) -> None:
    """`my-agent` 无参数时调用 `my_agent.tui.__main__.main()`。"""
    calls: list[bool] = []

    def fake_tui_main() -> None:
        calls.append(True)

    monkeypatch.setattr(sys, "argv", ["my-agent"])
    # Important: patch the symbol used inside main(), not the module attribute at import time.
    monkeypatch.setattr(
        "my_agent.tui.__main__.main", fake_tui_main, raising=True
    )

    cli_main.main()

    assert calls == [True]


def test_main_help_does_not_invoke_tui(monkeypatch: MonkeyPatch) -> None:
    """`my-agent --help` 不启动 TUI。"""
    calls: list[bool] = []

    def fake_tui_main() -> None:
        calls.append(True)

    monkeypatch.setattr(sys, "argv", ["my-agent", "--help"])
    monkeypatch.setattr("my_agent.tui.__main__.main", fake_tui_main)

    with pytest.raises(SystemExit):
        cli_main.main()

    assert calls == []


def test_main_version_does_not_invoke_tui(monkeypatch: MonkeyPatch) -> None:
    """`my-agent --version` 不启动 TUI。"""
    calls: list[bool] = []

    def fake_tui_main() -> None:
        calls.append(True)

    monkeypatch.setattr(sys, "argv", ["my-agent", "--version"])
    monkeypatch.setattr("my_agent.tui.__main__.main", fake_tui_main)

    cli_main.main()

    assert calls == []


def test_main_core_status_does_not_invoke_tui(monkeypatch: MonkeyPatch) -> None:
    """`my-agent core status` 不启动 TUI。"""
    calls: list[bool] = []

    def fake_tui_main() -> None:
        calls.append(True)

    def fake_cmd_core_status(host: str, port: int) -> None:
        # Stub so we don't actually probe
        pass

    monkeypatch.setattr(sys, "argv", ["my-agent", "core", "status"])
    monkeypatch.setattr("my_agent.tui.__main__.main", fake_tui_main)
    monkeypatch.setattr(cli_main, "cmd_core_status", fake_cmd_core_status)

    cli_main.main()

    assert calls == []


def test_main_ping_does_not_invoke_tui(monkeypatch: MonkeyPatch) -> None:
    """`my-agent ping` 不启动 TUI。"""
    calls: list[bool] = []

    def fake_tui_main() -> None:
        calls.append(True)

    monkeypatch.setattr(sys, "argv", ["my-agent", "ping"])
    monkeypatch.setattr("my_agent.tui.__main__.main", fake_tui_main)

    # Stub get_config and cmd_ping so they don't actually probe a real daemon
    from types import SimpleNamespace

    fake_config = SimpleNamespace(host="127.0.0.1", port=7437)
    monkeypatch.setattr(cli_main, "get_config", lambda: fake_config)
    monkeypatch.setattr(cli_main, "cmd_ping", lambda cfg: None)

    cli_main.main()

    assert calls == []
