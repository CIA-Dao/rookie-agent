from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from pytest import MonkeyPatch

import my_agent.cli.main as cli_main


def test_core_status_dispatches_to_cmd_core_status_with_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """argparse 把 `my-agent core status` 分发给 cmd_core_status(config.host, config.port)。"""

    fake_config = SimpleNamespace(host="10.0.0.1", port=9999)

    def fake_get_config() -> Any:
        return fake_config

    captured: dict[str, Any] = {}

    def fake_cmd_core_status(host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli_main, "get_config", fake_get_config)
    monkeypatch.setattr(cli_main, "cmd_core_status", fake_cmd_core_status)
    monkeypatch.setattr(
        "sys.argv",
        ["my-agent", "core", "status"],
    )

    cli_main.main()

    assert captured == {"host": "10.0.0.1", "port": 9999}


def test_core_start_dispatches_to_cmd_core_start_with_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """argparse 把 `my-agent core start` 分发给 cmd_core_start(config.host, config.port)。"""

    fake_config = SimpleNamespace(host="10.0.0.1", port=9999)
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "get_config", lambda: fake_config)
    monkeypatch.setattr(
        cli_main,
        "cmd_core_start",
        lambda host, port: captured.update({"host": host, "port": port}),
    )
    monkeypatch.setattr(sys, "argv", ["my-agent", "core", "start"])

    cli_main.main()

    assert captured == {"host": "10.0.0.1", "port": 9999}


def test_core_stop_dispatches_to_cmd_core_stop_with_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """argparse 把 `my-agent core stop` 分发给 cmd_core_stop(config.host, config.port)。"""

    fake_config = SimpleNamespace(host="10.0.0.1", port=9999)
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "get_config", lambda: fake_config)
    monkeypatch.setattr(
        cli_main,
        "cmd_core_stop",
        lambda host, port: captured.update({"host": host, "port": port}),
    )
    monkeypatch.setattr(sys, "argv", ["my-agent", "core", "stop"])

    cli_main.main()

    assert captured == {"host": "10.0.0.1", "port": 9999}
