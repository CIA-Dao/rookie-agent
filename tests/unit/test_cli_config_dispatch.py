from __future__ import annotations

import sys

from pytest import MonkeyPatch

import my_agent.cli.main as cli_main
from my_agent.core.config import Config


def test_main_passes_config_to_ping(monkeypatch: MonkeyPatch) -> None:
    """main() 对 `my-agent ping` 从 get_config() 取配置传给 cmd_ping。"""
    config = Config(host="127.0.0.9", port=8765)
    received: list[Config] = []

    monkeypatch.setattr(sys, "argv", ["my-agent", "ping"])
    monkeypatch.setattr(cli_main, "get_config", lambda: config)
    monkeypatch.setattr(cli_main, "cmd_ping", lambda cfg: received.append(cfg))

    cli_main.main()

    assert received == [config]


def test_main_passes_config_to_run(monkeypatch: MonkeyPatch) -> None:
    """main() 对 `my-agent run <goal>` 从 get_config() 取配置传给 cmd_run。"""
    config = Config(host="127.0.0.9", port=8765)
    received: list[tuple[str, Config]] = []

    monkeypatch.setattr(sys, "argv", ["my-agent", "run", "hello"])
    monkeypatch.setattr(cli_main, "get_config", lambda: config)
    monkeypatch.setattr(
        cli_main,
        "cmd_run",
        lambda goal, cfg: received.append((goal, cfg)),
    )

    cli_main.main()

    assert received == [("hello", config)]


def test_main_passes_config_to_chat(monkeypatch: MonkeyPatch) -> None:
    """main() 对 `my-agent chat` 从 get_config() 取配置传给 cmd_chat。"""
    config = Config(host="127.0.0.9", port=8765)
    received: list[Config] = []

    monkeypatch.setattr(sys, "argv", ["my-agent", "chat"])
    monkeypatch.setattr(cli_main, "get_config", lambda: config)
    monkeypatch.setattr(cli_main, "cmd_chat", lambda cfg: received.append(cfg))

    cli_main.main()

    assert received == [config]


def test_main_dispatches_init(monkeypatch: MonkeyPatch) -> None:
    """main() 对 `my-agent init` 分发到 cmd_init。"""
    called: list[bool] = []

    monkeypatch.setattr(sys, "argv", ["my-agent", "init"])
    monkeypatch.setattr(
        cli_main,
        "cmd_init",
        lambda: called.append(True),
    )

    cli_main.main()

    assert called == [True]
