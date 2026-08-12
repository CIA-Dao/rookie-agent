from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from my_agent.cli.commands.core import (
    CoreProbeError,
    CoreProbeResult,
    ProcessExitedError,
    _spawn_core_process,
    cmd_core_start,
    cmd_core_status,
    cmd_core_stop,
    probe_core,
)


class _FakeReader:
    async def readline(self) -> bytes:
        return b'{"hello": "not-my-agent-core"}\n'


class _FakeWriter:
    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


async def test_probe_core_rejects_json_that_is_not_a_core_ping_response(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_open_connection(host: str, port: int) -> tuple[_FakeReader, _FakeWriter]:
        return _FakeReader(), _FakeWriter()

    monkeypatch.setattr(
        "my_agent.cli.commands.core.asyncio.open_connection",
        fake_open_connection,
    )

    with pytest.raises(RuntimeError, match="invalid core.ping response"):
        await probe_core("127.0.0.1", 7437)


def test_cmd_core_status_online_prints_running_and_returns_zero(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """在线时打印 'core running at ...' 并正常返回（exit code 0）。"""

    fixed = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=1234,
        latency_ms=5,
    )

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        return fixed

    monkeypatch.setattr(
        "my_agent.cli.commands.core.probe_core",
        fake_probe,
    )

    cmd_core_status("127.0.0.1", 7437)

    out = capsys.readouterr().out
    assert "core running at 127.0.0.1:7437" in out
    assert "server=0.0.1" in out
    assert "uptime=1234ms" in out


def test_cmd_core_status_offline_prints_not_running_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """离线时打印 'core not running at ...' 并 sys.exit(1)。"""

    async def fake_probe_refused(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(
        "my_agent.cli.commands.core.probe_core",
        fake_probe_refused,
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_core_status("127.0.0.1", 7437)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "core not running at 127.0.0.1:7437" in out


def test_cmd_core_status_timeout_prints_not_running_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """超时时与离线使用同一稳定提示，退出码也是 1。"""

    async def fake_probe_timeout(host: str, port: int) -> CoreProbeResult:
        raise TimeoutError("read timeout")

    monkeypatch.setattr(
        "my_agent.cli.commands.core.probe_core",
        fake_probe_timeout,
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_core_status("127.0.0.1", 7437)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "core not running at 127.0.0.1:7437" in out


def test_cmd_core_status_invalid_response_prints_not_running_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_probe_invalid(host: str, port: int) -> CoreProbeResult:
        raise CoreProbeError("invalid core.ping response")

    monkeypatch.setattr(
        "my_agent.cli.commands.core.probe_core",
        fake_probe_invalid,
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_core_status("127.0.0.1", 7437)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "core not running at 127.0.0.1:7437" in out


# ---------------------------------------------------------------------------
# core start —— 已在线时不重复启动
# ---------------------------------------------------------------------------


def test_cmd_core_start_already_running_does_not_spawn(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """已在线时：不调用 Popen，打印 already running，正常返回（退出码 0）。"""

    fixed = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=1234,
        latency_ms=5,
    )

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        return fixed

    def fail_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Popen should not be called when core is already running")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fail_spawn)

    pid_file = tmp_path / "core.pid"
    cmd_core_start("127.0.0.1", 7437, pid_file=pid_file)

    out = capsys.readouterr().out
    assert "core already running at 127.0.0.1:7437" in out
    assert "server=0.0.1" in out
    assert "uptime=1234ms" in out
    # 已在线时不应该写 PID 文件
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# core start —— 离线时启动成功路径
# ---------------------------------------------------------------------------


class _FakeProc:
    """模拟 subprocess.Popen 返回的进程对象。"""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        # None 表示进程仍在运行
        return None

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        return 0


# ---------------------------------------------------------------------------
# core start —— startup log 每次启动必须被截断，避免历史 stderr 干扰诊断
# ---------------------------------------------------------------------------


def test_spawn_core_process_truncates_startup_log(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """_spawn_core_process 必须截断 startup log，不能追加。

    如果用 'ab' 追加模式，前一次失败留下的 'DEEPSEEK_API_KEY not set' 会
    在本次完全不同的错误中被误判为缺 key，触发错误的 setup 引导。
    """
    startup_log = tmp_path / "logs" / "core-startup.err.log"

    # 模拟前一次失败留下的历史 stderr
    startup_log.parent.mkdir(parents=True, exist_ok=True)
    startup_log.write_bytes(b"DEEPSEEK_API_KEY not set\nold traceback line\n")

    # Mock subprocess.Popen 避免真实启动 daemon
    monkeypatch.setattr(
        "my_agent.cli.commands.core.subprocess.Popen",
        lambda *a, **kw: _FakeProc(pid=12345),
    )

    _spawn_core_process(stderr_file=startup_log)

    # 截断后文件应为空（daemon 还没来得及写新内容）
    assert startup_log.read_bytes() == b"", (
        "startup log must be truncated before spawn, not appended. "
        "Stale content causes false missing_deepseek_key detection."
    )


def test_cmd_core_start_offline_spawns_and_writes_pid_and_polls_ready(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """离线时：Popen 启动 daemon、写 PID 文件、轮询成功后打印 started。"""

    ready = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=42,
        latency_ms=5,
    )
    call_count = {"n": 0}

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        call_count["n"] += 1
        # 第一次 probe（cmd_core_start 入口）抛连接拒绝，后续轮询返回 ready
        if call_count["n"] == 1:
            raise ConnectionRefusedError("connection refused")
        return ready

    spawn_calls: list[dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> _FakeProc:
        spawn_calls.append(kwargs)
        return _FakeProc(pid=12345)

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_spawn)
    # wait_for_core_ready_with_proc is imported from cli.commands.core into lifecycle,
    # so its internal probe_core call resolves in the cli.commands.core namespace.
    # Patch both namespaces to keep the test deterministic (no real TCP).
    monkeypatch.setattr("my_agent.cli.commands.core.probe_core", fake_probe)

    pid_file = tmp_path / "core.pid"
    cmd_core_start("127.0.0.1", 7437, pid_file=pid_file)

    # PID 文件内容
    assert pid_file.read_text(encoding="utf-8") == "12345\n"
    # _spawn_core_process 被调用
    assert len(spawn_calls) == 1
    # 输出包含地址和 pid
    out = capsys.readouterr().out
    assert "core started at 127.0.0.1:7437" in out
    assert "pid=12345" in out
    assert "server=0.0.1" in out


# ---------------------------------------------------------------------------
# core start —— 超时时清理 PID 文件并退出 1
# ---------------------------------------------------------------------------


def test_cmd_core_start_timeout_cleans_pid_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """启动后轮询超时：清理本次写入的 PID 文件，打印 error，sys.exit(1)。"""

    call_count = {"n": 0}

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        call_count["n"] += 1
        # 入口 probe 抛连接拒绝（离线）；后续 wait_for_core_ready 持续超时
        raise ConnectionRefusedError("connection refused")

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(pid=99999)

    async def fake_wait(
        proc: _FakeProc,
        host: str,
        port: int,
        timeout_s: float = 5.0,
    ) -> CoreProbeResult:
        raise TimeoutError("core did not become ready")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_popen)
    monkeypatch.setattr(
        "my_agent.core.lifecycle.wait_for_core_ready_with_proc", fake_wait
    )

    pid_file = tmp_path / "core.pid"
    with pytest.raises(SystemExit) as exc_info:
        cmd_core_start("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    # PID 文件应该被清理
    assert not pid_file.exists()
    out = capsys.readouterr().out
    assert "error: core did not become ready at 127.0.0.1:7437" in out


def test_cmd_core_start_popen_oserror_prints_error_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Popen 抛 OSError：打印错误，sys.exit(1)，不写 PID。"""

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    def fail_popen(*args: Any, **kwargs: Any) -> Any:
        raise OSError("permission denied")

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fail_popen)

    pid_file = tmp_path / "core.pid"
    with pytest.raises(SystemExit) as exc_info:
        cmd_core_start("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "error: failed to start core" in out
    assert "permission denied" in out
    assert not pid_file.exists()


def test_cmd_core_start_process_exits_before_ready_cleans_pid_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """子进程在 ready 前退出：清理 PID，打印 error，sys.exit(1)。"""

    class _DeadProc:
        def __init__(self) -> None:
            self.pid = 55555

        def poll(self) -> int | None:
            # 进程已退出，返回非 0 退出码
            return 1

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    def fake_popen(*args: Any, **kwargs: Any) -> _DeadProc:
        return _DeadProc()

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_popen)
    # 注意：不 mock wait_for_core_ready_with_proc，让真实函数检测到
    # proc.poll() != None 并抛 ProcessExitedError

    pid_file = tmp_path / "core.pid"
    with pytest.raises(SystemExit) as exc_info:
        cmd_core_start("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    assert not pid_file.exists()
    out = capsys.readouterr().out
    assert "error: core process exited before becoming ready" in out


def test_cmd_core_start_process_exits_before_ready_prints_stderr_tail(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """子进程在 ready 前退出：打印 exit_code、stderr 摘要、~/.my-agent/.env 提示、前台调试提示。"""

    class _DeadProc:
        def __init__(self) -> None:
            self.pid = 55555

        def poll(self) -> int | None:
            return 1

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    async def fake_wait(*args: Any, **kwargs: Any) -> CoreProbeResult:
        # 模拟 wait_for_core_ready_with_proc 检测到 proc.poll() != None
        raise ProcessExitedError("core process exited before becoming ready")

    def fake_popen(*args: Any, **kwargs: Any) -> _DeadProc:
        return _DeadProc()

    monkeypatch.setattr("my_agent.core.lifecycle.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.core.lifecycle._spawn_core_process", fake_popen)
    monkeypatch.setattr(
        "my_agent.core.lifecycle.wait_for_core_ready_with_proc", fake_wait
    )

    # 预先写入 startup log，模拟 daemon 子进程写出的真实错误
    startup_log = tmp_path / "core-startup.err.log"
    startup_log.write_text("DEEPSEEK_API_KEY not set\n", encoding="utf-8")

    pid_file = tmp_path / "core.pid"
    with pytest.raises(SystemExit) as exc_info:
        cmd_core_start(
            "127.0.0.1",
            7437,
            pid_file=pid_file,
            startup_log_file=startup_log,
        )

    assert exc_info.value.code == 1
    assert not pid_file.exists()
    out = capsys.readouterr().out
    assert "error: core process exited before becoming ready" in out
    assert "exit_code=1" in out
    assert "stderr:" in out
    assert "DEEPSEEK_API_KEY not set" in out
    assert "~/.my-agent/.env" in out or ".my-agent/.env" in out
    assert "my-agent-core" in out


# ---------------------------------------------------------------------------
# core stop —— 无 PID 文件且离线时幂等返回 0
# ---------------------------------------------------------------------------


def test_cmd_core_stop_no_pid_and_offline_prints_not_running_and_returns_zero(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """无 PID 文件且 Core 离线：打印 'core not running at ...'，正常返回（exit 0）。"""

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("my_agent.cli.commands.core.probe_core", fake_probe)

    pid_file = tmp_path / "missing.pid"
    cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    out = capsys.readouterr().out
    assert "core not running at 127.0.0.1:7437" in out
    assert not pid_file.exists()


def test_cmd_core_stop_no_pid_but_online_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """无 PID 文件但 Core 在线：不尝试停止，打印 error，sys.exit(1)。"""

    fixed = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=1234,
        latency_ms=5,
    )

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        return fixed

    terminate_calls: list[int] = []

    def fake_terminate(pid: int) -> None:
        terminate_calls.append(pid)

    monkeypatch.setattr("my_agent.cli.commands.core.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.cli.commands.core.terminate_process", fake_terminate)

    pid_file = tmp_path / "missing.pid"
    with pytest.raises(SystemExit) as exc_info:
        cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    assert terminate_calls == []
    out = capsys.readouterr().out
    assert (
        "error: core is running at 127.0.0.1:7437 but no pid file was found" in out
    )


def test_cmd_core_stop_stale_pid_and_offline_cleans_pid_and_returns_zero(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """PID 文件指向不存在的进程且 Core 离线：清理 stale PID 文件，返回 0。"""

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("my_agent.cli.commands.core.probe_core", fake_probe)
    monkeypatch.setattr(
        "my_agent.cli.commands.core.is_process_running",
        lambda pid: False,
    )

    pid_file = tmp_path / "core.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert not pid_file.exists()
    out = capsys.readouterr().out
    assert "core not running at 127.0.0.1:7437" in out
    assert "removed stale pid file" in out


def test_cmd_core_stop_stale_pid_but_online_keeps_pid_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """stale PID + Core 在线：保留 PID、不乱杀、sys.exit(1)、打印可行动排查提示。"""

    fixed = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=1234,
        latency_ms=5,
    )

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        return fixed

    terminate_calls: list[int] = []

    def fake_terminate(pid: int) -> None:
        terminate_calls.append(pid)

    monkeypatch.setattr("my_agent.cli.commands.core.probe_core", fake_probe)
    monkeypatch.setattr(
        "my_agent.cli.commands.core.is_process_running",
        lambda pid: False,
    )
    monkeypatch.setattr("my_agent.cli.commands.core.terminate_process", fake_terminate)

    # 使用与默认 PID 文件同名的临时路径，让提示中的文件名更具代表性
    pid_file = tmp_path / ".my-agent" / "my-agent-core.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("12345\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    assert pid_file.exists()
    assert terminate_calls == []
    out = capsys.readouterr().out
    # 保留原有 error 主文案关键信息
    assert "error:" in out
    assert "core is running at 127.0.0.1:7437" in out
    assert "pid file points to dead pid=12345" in out
    assert "cannot stop safely" in out
    # 新增的可行动提示
    assert "stale pid file" in out
    assert "my-agent will not stop a process by port automatically" in out
    assert "Get-NetTCPConnection" in out
    assert "Get-CimInstance Win32_Process" in out
    assert "Set-Content" in out
    assert "my-agent-core.pid" in out
    assert "$actual_pid" in out


def test_cmd_core_stop_invalid_pid_and_offline_cleans_pid_and_returns_zero(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """PID 文件内容非法且 Core 离线：清理 invalid PID 文件，返回 0。"""

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("my_agent.cli.commands.core.probe_core", fake_probe)

    pid_file = tmp_path / "core.pid"
    pid_file.write_text("not-a-pid\n", encoding="utf-8")

    cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert not pid_file.exists()
    out = capsys.readouterr().out
    assert "core not running at 127.0.0.1:7437" in out
    assert "removed invalid pid file" in out


def test_cmd_core_stop_invalid_pid_but_online_keeps_pid_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """PID 文件内容非法但 Core 在线：保留 PID 文件，不乱杀，sys.exit(1)。"""

    fixed = CoreProbeResult(
        server_version="0.0.1",
        uptime_ms=1234,
        latency_ms=5,
    )

    async def fake_probe(host: str, port: int) -> CoreProbeResult:
        return fixed

    terminate_calls: list[int] = []

    def fake_terminate(pid: int) -> None:
        terminate_calls.append(pid)

    monkeypatch.setattr("my_agent.cli.commands.core.probe_core", fake_probe)
    monkeypatch.setattr("my_agent.cli.commands.core.terminate_process", fake_terminate)

    pid_file = tmp_path / "core.pid"
    pid_file.write_text("not-a-pid\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    assert pid_file.exists()
    assert terminate_calls == []
    out = capsys.readouterr().out
    assert "error:" in out
    assert "pid file is invalid" in out
    assert "cannot stop safely" in out


def test_cmd_core_stop_happy_path_terminates_waits_cleans_pid_and_prints_stopped(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """PID 文件指向活跃进程：发 SIGTERM、等进程退出、确认 Core 离线、清 PID、打印 stopped。"""

    running = {"yes": True}
    terminate_calls: list[int] = []
    process_exit_calls: list[tuple[int, float]] = []
    core_offline_calls: list[tuple[str, int, float]] = []

    def fake_is_running(pid: int) -> bool:
        return running["yes"]

    def fake_terminate(pid: int) -> None:
        terminate_calls.append(pid)
        # 模拟 terminate 后进程消失
        running["yes"] = False

    async def fake_wait_exit(pid: int, timeout_s: float = 5.0) -> None:
        process_exit_calls.append((pid, timeout_s))
        return None

    async def fake_wait_offline(host: str, port: int, timeout_s: float = 5.0) -> None:
        core_offline_calls.append((host, port, timeout_s))
        return None

    monkeypatch.setattr(
        "my_agent.cli.commands.core.is_process_running",
        fake_is_running,
    )
    monkeypatch.setattr("my_agent.cli.commands.core.terminate_process", fake_terminate)
    monkeypatch.setattr(
        "my_agent.cli.commands.core.wait_for_process_exit",
        fake_wait_exit,
    )
    monkeypatch.setattr(
        "my_agent.cli.commands.core.wait_for_core_offline",
        fake_wait_offline,
    )

    pid_file = tmp_path / "core.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert terminate_calls == [12345]
    assert process_exit_calls == [(12345, 5.0)]
    assert core_offline_calls == [("127.0.0.1", 7437, 5.0)]
    assert not pid_file.exists()
    out = capsys.readouterr().out
    assert "core stopped at 127.0.0.1:7437 pid=12345" in out


def test_cmd_core_stop_process_did_not_exit_keeps_pid_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """wait_for_process_exit 抛 TimeoutError：保留 PID，打印 error，sys.exit(1)。"""

    terminate_calls: list[int] = []
    offline_calls: list[tuple[str, int]] = []

    def fake_is_running(pid: int) -> bool:
        return True

    def fake_terminate(pid: int) -> None:
        terminate_calls.append(pid)

    async def fake_wait_exit_timeout(pid: int, timeout_s: float = 5.0) -> None:
        raise TimeoutError(f"process {pid} did not exit")

    async def fake_wait_offline(host: str, port: int, timeout_s: float = 5.0) -> None:
        offline_calls.append((host, port))

    monkeypatch.setattr("my_agent.cli.commands.core.is_process_running", fake_is_running)
    monkeypatch.setattr("my_agent.cli.commands.core.terminate_process", fake_terminate)
    monkeypatch.setattr(
        "my_agent.cli.commands.core.wait_for_process_exit",
        fake_wait_exit_timeout,
    )
    monkeypatch.setattr(
        "my_agent.cli.commands.core.wait_for_core_offline",
        fake_wait_offline,
    )

    pid_file = tmp_path / "core.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    assert terminate_calls == [12345]
    # 进程没退出，不应该再尝试 probe Core
    assert offline_calls == []
    # PID 文件保留
    assert pid_file.exists()
    out = capsys.readouterr().out
    assert "error: core process 12345 did not exit" in out


def test_cmd_core_stop_process_exited_but_core_still_online_keeps_pid_and_exits_one(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """进程已退出但 Core 仍在线（probe 超时）：保留 PID，打印 error，sys.exit(1)。"""

    terminate_calls: list[int] = []
    process_exit_calls: list[int] = []

    def fake_is_running(pid: int) -> bool:
        return True

    def fake_terminate(pid: int) -> None:
        terminate_calls.append(pid)

    async def fake_wait_exit(pid: int, timeout_s: float = 5.0) -> None:
        process_exit_calls.append(pid)
        return None

    async def fake_wait_offline_timeout(
        host: str,
        port: int,
        timeout_s: float = 5.0,
    ) -> None:
        raise TimeoutError(f"core still running at {host}:{port}")

    monkeypatch.setattr("my_agent.cli.commands.core.is_process_running", fake_is_running)
    monkeypatch.setattr("my_agent.cli.commands.core.terminate_process", fake_terminate)
    monkeypatch.setattr(
        "my_agent.cli.commands.core.wait_for_process_exit",
        fake_wait_exit,
    )
    monkeypatch.setattr(
        "my_agent.cli.commands.core.wait_for_core_offline",
        fake_wait_offline_timeout,
    )

    pid_file = tmp_path / "core.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cmd_core_stop("127.0.0.1", 7437, pid_file=pid_file)

    assert exc_info.value.code == 1
    assert terminate_calls == [12345]
    assert process_exit_calls == [12345]
    # PID 文件保留
    assert pid_file.exists()
    out = capsys.readouterr().out
    assert (
        "error: core still running at 127.0.0.1:7437 after stopping pid=12345"
        in out
    )
