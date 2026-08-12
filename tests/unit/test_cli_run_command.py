from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from my_agent.cli.commands import run as run_command
from my_agent.core.config import Config


def test_cmd_run_connection_failure_prints_actionable_hint(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run(goal: str, config: Config) -> int:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(run_command, "_run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_command.cmd_run("hello", Config(host="127.0.0.9", port=8765))

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: cannot connect to my-agent core at 127.0.0.9:8765" in captured.err
    assert "my-agent core status" in captured.err
    assert "my-agent core start" in captured.err
