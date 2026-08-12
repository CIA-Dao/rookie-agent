from __future__ import annotations

from pytest import MonkeyPatch

from my_agent.core.app import CoreApp


async def test_ping_reports_elapsed_monotonic_uptime_and_current_received_at(
    monkeypatch: MonkeyPatch,
) -> None:
    """ping handler 返回真实 uptime（基于 monotonic）和真实 received_at（基于 _now）。"""
    ticks = iter([100.0, 101.25])
    monkeypatch.setattr("my_agent.core.app.monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        "my_agent.core.app._now",
        lambda: "2026-07-27T12:34:56+00:00",
    )

    app = CoreApp()
    result = await app._ping_handler({})

    assert result.server_version == "0.0.1"
    assert result.uptime_ms == 1250
    assert result.received_at == "2026-07-27T12:34:56+00:00"


async def test_ping_uptime_never_negative_when_clock_moves_backward(
    monkeypatch: MonkeyPatch,
) -> None:
    """构造时 monotonic 为 100.0，handler 调用时回退到 99.5，uptime 应 clamp 到 0。"""
    ticks = iter([100.0, 99.5])
    monkeypatch.setattr("my_agent.core.app.monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        "my_agent.core.app._now",
        lambda: "2026-07-27T12:34:56+00:00",
    )

    app = CoreApp()
    result = await app._ping_handler({})

    assert result.uptime_ms == 0
