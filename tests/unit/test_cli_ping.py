from __future__ import annotations

import json

import pytest
from pytest import MonkeyPatch

from my_agent.cli.commands.ping import _ping, cmd_ping
from my_agent.core.config import Config


class _FakeReader:
    async def readline(self) -> bytes:
        response = {
            "jsonrpc": "2.0",
            "id": "cli-1",
            "result": {
                "server_version": "0.0.1",
                "uptime_ms": 123,
                "received_at": "2026-07-27T12:34:56+00:00",
            },
        }
        return (json.dumps(response) + "\n").encode()


class _FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


async def test_ping_uses_configured_host_and_port(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    async def fake_open_connection(host: str, port: int) -> tuple[_FakeReader, _FakeWriter]:
        calls.append((host, port))
        return _FakeReader(), _FakeWriter()

    monkeypatch.setattr(
        "my_agent.cli.commands.ping.asyncio.open_connection",
        fake_open_connection,
    )
    monkeypatch.setattr("my_agent.cli.commands.ping.time.monotonic", lambda: 100.0)

    await _ping(Config(host="127.0.0.9", port=8765))

    assert calls == [("127.0.0.9", 8765)]


def test_cmd_ping_connection_failure_prints_actionable_hint(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_ping(config: Config) -> None:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("my_agent.cli.commands.ping._ping", fake_ping)

    with pytest.raises(SystemExit) as exc_info:
        cmd_ping(Config(host="127.0.0.9", port=8765))

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: cannot connect to my-agent core at 127.0.0.9:8765" in captured.err
    assert "my-agent core status" in captured.err
    assert "my-agent core start" in captured.err
