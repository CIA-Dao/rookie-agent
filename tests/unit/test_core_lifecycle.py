from __future__ import annotations

from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from my_agent.cli.commands.core import CoreProbeResult
from my_agent.core.lifecycle import (
    CoreStartFailed,
    CoreStartOk,
    ensure_core_started,
)


class _FakeProc:
    """模拟 subprocess.Popen 返回的进程对象。"""

    def __init__(self, pid: int, poll_value: int | None = None) -> None:
        self.pid = pid
        self._poll_value = poll_value
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self._poll_value

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._poll_value = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._poll_value = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        return self._poll_value or 0


# ---------------------------------------------------------------------------
# 已在线：返回 CoreStartOk(already_running=True)，不 spawn
# ---------------------------------------------------------------------------


async def test_ensure_core_started_already_running_returns_ok_without_spawn(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixed = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=1234,
        latency_ms=5,
    )

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        return fixed

    def fail_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("_spawn_core_process should not be called when already running")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fail_spawn)

    pid_file = tmp_path / "core.pid"
    startup_log = tmp_path / "core-startup.err.log"

    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=pid_file,
        startup_log_file=startup_log,
    )

    assert isinstance(result, CoreStartOk)
    assert result.already_running is True
    assert result.host == "127.0.0.1"
    assert result.port == 7437
    assert result.ready.server_version == "0.0.1"
    assert result.ready.uptime_ms == 1234
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# 离线后启动成功：返回 CoreStartOk(already_running=False)，写 PID 文件
# ---------------------------------------------------------------------------


async def test_ensure_core_started_offline_then_ready_returns_ok_with_pid(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    ready = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=42,
        latency_ms=5,
    )
    call_count = {"n": 0}

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionRefusedError("connection refused")
        return ready

    spawn_calls: list[dict[str, Any]] = []

    def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        spawn_calls.append({"args": args, "kwargs": kwargs})
        return _FakeProc(pid=12345)

    async def fake_wait(
        proc: Any,
        host: str,
        port: int,
        timeout_s: float = 5.0,
    ) -> CoreProbeResult:
        return ready

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_spawn)
    monkeypatch.setattr(
        "my_agent.core.lifecycle.wait_for_core_ready_with_proc",
        fake_wait,
    )

    pid_file = tmp_path / "core.pid"
    startup_log = tmp_path / "core-startup.err.log"

    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=pid_file,
        startup_log_file=startup_log,
    )

    assert isinstance(result, CoreStartOk)
    assert result.already_running is False
    assert result.pid == 12345
    assert result.host == "127.0.0.1"
    assert result.port == 7437
    assert result.ready.server_version == "0.0.1"
    assert pid_file.read_text(encoding="utf-8") == "12345\n"


# ---------------------------------------------------------------------------
# 子进程在 ready 前退出：清理 PID，返回 CoreStartFailed(process_exited)
# ---------------------------------------------------------------------------


async def test_ensure_core_started_process_exited_cleans_pid_and_returns_failed(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    from my_agent.cli.commands.core import ProcessExitedError

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(pid=55555, poll_value=1)

    async def fake_wait(
        proc: Any,
        host: str,
        port: int,
        timeout_s: float = 5.0,
    ) -> CoreProbeResult:
        raise ProcessExitedError("core process exited before becoming ready")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_spawn)
    monkeypatch.setattr(
        "my_agent.core.lifecycle.wait_for_core_ready_with_proc",
        fake_wait,
    )

    pid_file = tmp_path / "core.pid"
    startup_log = tmp_path / "core-startup.err.log"
    startup_log.write_text("DEEPSEEK_API_KEY not set\n", encoding="utf-8")

    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=pid_file,
        startup_log_file=startup_log,
    )

    assert isinstance(result, CoreStartFailed)
    assert result.reason == "process_exited"
    assert result.exit_code == 1
    assert "DEEPSEEK_API_KEY not set" in result.stderr_tail
    assert not pid_file.exists()
    assert result.missing_deepseek_key is True


async def test_ensure_core_started_timeout_cleans_pid_and_returns_failed(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    proc = _FakeProc(pid=99999)

    def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    async def fake_wait(
        proc: Any,
        host: str,
        port: int,
        timeout_s: float = 5.0,
    ) -> CoreProbeResult:
        raise TimeoutError("core did not become ready")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_spawn)
    monkeypatch.setattr(
        "my_agent.core.lifecycle.wait_for_core_ready_with_proc",
        fake_wait,
    )

    pid_file = tmp_path / "core.pid"
    startup_log = tmp_path / "core-startup.err.log"

    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=pid_file,
        startup_log_file=startup_log,
    )

    assert isinstance(result, CoreStartFailed)
    assert result.reason == "timeout"
    assert result.exit_code is None
    assert not pid_file.exists()
    assert result.missing_deepseek_key is False
    assert proc.terminate_calls == 1
    assert proc.wait_calls == [2.0]


async def test_ensure_core_started_timeout_final_probe_recovers_boundary_race(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    ready = CoreProbeResult(server_version="0.0.2", uptime_ms=1, latency_ms=1)
    probe_calls = {"count": 0}
    proc = _FakeProc(pid=77777)

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        probe_calls["count"] += 1
        if probe_calls["count"] == 1:
            raise ConnectionRefusedError("connection refused")
        return ready

    async def fake_wait(*args: Any, **kwargs: Any) -> CoreProbeResult:
        raise TimeoutError("core crossed the polling deadline")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", lambda **kwargs: proc)
    monkeypatch.setattr(
        "my_agent.core.lifecycle.wait_for_core_ready_with_proc",
        fake_wait,
    )

    pid_file = tmp_path / "core.pid"
    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=pid_file,
        startup_log_file=tmp_path / "startup.err.log",
    )

    assert isinstance(result, CoreStartOk)
    assert result.pid == 77777
    assert result.ready == ready
    assert pid_file.read_text(encoding="utf-8") == "77777\n"
    assert proc.terminate_calls == 0


async def test_ensure_core_started_spawn_oserror_returns_failed_spawn_error(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    def fail_spawn(*args: Any, **kwargs: Any) -> Any:
        raise OSError("permission denied")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fail_spawn)

    pid_file = tmp_path / "core.pid"
    startup_log = tmp_path / "core-startup.err.log"

    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=pid_file,
        startup_log_file=startup_log,
    )

    assert isinstance(result, CoreStartFailed)
    assert result.reason == "spawn_error"
    assert "permission denied" in result.stderr_tail
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# missing_deepseek_key 启发式：stderr 包含 "DEEPSEEK_API_KEY not set" 时为 True
# ---------------------------------------------------------------------------


def test_missing_deepseek_key_property_true_when_stderr_contains_marker() -> None:
    failed = CoreStartFailed(
        host="127.0.0.1",
        port=7437,
        reason="process_exited",
        exit_code=1,
        stderr_tail="Traceback ...\nDEEPSEEK_API_KEY not set\nRuntimeError",
    )
    assert failed.missing_deepseek_key is True


def test_missing_deepseek_key_property_false_when_stderr_does_not_contain_marker() -> None:
    failed = CoreStartFailed(
        host="127.0.0.1",
        port=7437,
        reason="process_exited",
        exit_code=2,
        stderr_tail="some other error",
    )
    assert failed.missing_deepseek_key is False


# ---------------------------------------------------------------------------
# on_status callback: Core 离线准备启动时通知调用方
# ---------------------------------------------------------------------------


async def test_ensure_core_started_calls_on_status_when_spawning(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Core 离线时 on_status 回调被调用，消息包含 'starting automatically'。"""
    ready = CoreProbeResult(server_version="0.0.1", uptime_ms=1, latency_ms=1)

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    def fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(pid=12345)

    async def fake_wait(*args: Any, **kwargs: Any) -> CoreProbeResult:
        return ready

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_spawn)
    monkeypatch.setattr(
        "my_agent.core.lifecycle.wait_for_core_ready_with_proc", fake_wait
    )

    status_messages: list[str] = []

    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=tmp_path / "core.pid",
        startup_log_file=tmp_path / "startup.err.log",
        on_status=status_messages.append,
    )

    assert isinstance(result, CoreStartOk)
    assert len(status_messages) == 1
    assert "starting automatically" in status_messages[0]


async def test_ensure_core_started_does_not_call_on_status_when_already_running(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Core 已在线时 on_status 回调不被调用。"""
    ready = CoreProbeResult(server_version="0.0.1", uptime_ms=1, latency_ms=1)

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        return ready

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)

    status_messages: list[str] = []

    result = await ensure_core_started(
        "127.0.0.1",
        7437,
        pid_file=tmp_path / "core.pid",
        startup_log_file=tmp_path / "startup.err.log",
        on_status=status_messages.append,
    )

    assert isinstance(result, CoreStartOk)
    assert result.already_running is True
    assert status_messages == []
