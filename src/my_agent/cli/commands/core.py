from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class CoreProbeError(RuntimeError):
    """The endpoint did not return a valid response to our core.ping request."""


@dataclass(frozen=True)
class CoreProbeResult:
    """core.ping 探测的结构化结果。"""

    server_version: str
    uptime_ms: int
    latency_ms: int


async def probe_core(host: str, port: int) -> CoreProbeResult:
    """连上 daemon 发 core.ping，返回结构化结果。

    纯通信逻辑：不打印、不 sys.exit。
    供后续 core start 复用。
    """
    t0 = time.monotonic()
    reader, writer = await asyncio.open_connection(host, port)
    try:
        req = {
            "jsonrpc": "2.0",
            "id": "core-status-probe",
            "method": "core.ping",
            "params": {"client": "my-agent-cli/0.0.1"},
        }
        writer.write((json.dumps(req) + "\n").encode())
        await writer.drain()

        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionResetError, OSError):
            pass

    latency_ms = int((time.monotonic() - t0) * 1000)
    try:
        resp = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CoreProbeError("invalid core.ping response") from exc

    if (
        not isinstance(resp, dict)
        or resp.get("jsonrpc") != "2.0"
        or resp.get("id") != "core-status-probe"
        or not isinstance(resp.get("result"), dict)
    ):
        raise CoreProbeError("invalid core.ping response")

    result = resp["result"]
    server_version = result.get("server_version")
    uptime_ms = result.get("uptime_ms")
    if not isinstance(server_version, str) or type(uptime_ms) is not int:
        raise CoreProbeError("invalid core.ping response")

    return CoreProbeResult(
        server_version=server_version,
        uptime_ms=uptime_ms,
        latency_ms=latency_ms,
    )


def cmd_core_status(host: str, port: int) -> None:
    """执行 `my-agent core status`：探测 + 打印 + 设置退出码。"""
    try:
        result = asyncio.run(probe_core(host, port))
    except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError):
        print(f"core not running at {host}:{port}")
        sys.exit(1)

    print(
        f"core running at {host}:{port} "
        f"server={result.server_version} "
        f"uptime={result.uptime_ms}ms"
    )


# ---------------------------------------------------------------------------
# core start
# ---------------------------------------------------------------------------

DEFAULT_CORE_PID_FILE = Path("~/.my-agent/my-agent-core.pid").expanduser()

# 启动后等待 ready 的总时间。
DEFAULT_READY_TIMEOUT_S = 5.0

# daemon 子进程 stderr 重定向到这里，便于 ready 前退出时诊断真实原因
DEFAULT_CORE_STARTUP_LOG_FILE = Path("~/.my-agent/logs/core-startup.err.log").expanduser()


async def wait_for_core_ready(
    host: str,
    port: int,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
) -> CoreProbeResult:
    """轮询 probe_core 直到可连接或超时。

    纯通信逻辑：成功返回 CoreProbeResult，失败抛 TimeoutError。
    """
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return await probe_core(host, port)
        except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError) as exc:
            last_error = exc
            await asyncio.sleep(0.05)
    raise TimeoutError(f"core did not become ready at {host}:{port}") from last_error


async def wait_for_core_ready_with_proc(
    proc: subprocess.Popen[bytes],
    host: str,
    port: int,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
) -> CoreProbeResult:
    """轮询 probe_core，同时检查子进程是否已提前退出。

    - 进程已退出（poll() 返回非 None）：抛 ProcessExitedError。
    - 超时：抛 TimeoutError。
    - 成功：返回 CoreProbeResult。
    """
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ProcessExitedError("core process exited before becoming ready")
        try:
            return await probe_core(host, port)
        except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError) as exc:
            last_error = exc
            await asyncio.sleep(0.05)
    raise TimeoutError(f"core did not become ready at {host}:{port}") from last_error


class ProcessExitedError(RuntimeError):
    """子进程在 ready 之前已经退出。"""


def _spawn_core_process(
    *,
    stderr_file: Path = DEFAULT_CORE_STARTUP_LOG_FILE,
) -> subprocess.Popen[bytes]:
    """后台启动 `python -m my_agent.core`，返回 Popen 对象。

    stderr 重定向到 stderr_file，便于 daemon 在 ready 前退出时复盘真实原因。
    文件句柄在 Popen 启动后关闭——子进程持有自己的继承句柄。
    """
    stderr_file.parent.mkdir(parents=True, exist_ok=True)
    # 用 "wb" 截断模式：每次启动清空旧日志，避免历史 stderr 干扰本次诊断
    # （否则前一次的 DEEPSEEK_API_KEY not set 会让本次不同错误的
    # missing_deepseek_key 启发式误判为 True）
    stderr_handle = stderr_file.open("wb")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "my_agent.core"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
            env=os.environ.copy(),
            cwd=str(Path.cwd()),
            start_new_session=True,
        )
    finally:
        stderr_handle.close()


# 读 path 末尾 max_chars 字符的文本；文件不存在或为空返回 ""
def _read_text_tail(path: Path, max_chars: int = 4000) -> str:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    return text[-max_chars:].strip()


def cmd_core_start(
    host: str,
    port: int,
    *,
    pid_file: Path = DEFAULT_CORE_PID_FILE,
    startup_log_file: Path = DEFAULT_CORE_STARTUP_LOG_FILE,
) -> None:
    """执行 `my-agent core start`。

    委托 lifecycle.ensure_core_started_sync() 完成实际工作，本函数只负责把
    结构化结果格式化为用户可见文本，并设置退出码。

    - 已在线：打印 already running，返回（退出 0）。
    - 离线后启动成功：打印 started。
    - 子进程在 ready 前退出：打印 exit_code + stderr 摘要 + 修复提示，sys.exit(1)。
    - timeout / spawn_error：打印 error，sys.exit(1)。
    """
    from my_agent.core.lifecycle import (
        CoreStartFailed,
        CoreStartOk,
        ensure_core_started_sync,
    )

    result = ensure_core_started_sync(
        host,
        port,
        pid_file=pid_file,
        startup_log_file=startup_log_file,
    )

    if isinstance(result, CoreStartOk):
        if result.already_running:
            print(
                f"core already running at {host}:{port} "
                f"server={result.ready.server_version} "
                f"uptime={result.ready.uptime_ms}ms"
            )
        else:
            print(
                f"core started at {host}:{port} "
                f"pid={result.pid} "
                f"server={result.ready.server_version} "
                f"uptime={result.ready.uptime_ms}ms"
            )
        return

    # CoreStartFailed 分支：根据 reason 打印不同错误信息
    assert isinstance(result, CoreStartFailed)

    if result.reason == "process_exited":
        print("error: core process exited before becoming ready")
        if result.exit_code is not None:
            print(f"exit_code={result.exit_code}")
        if result.stderr_tail:
            print("stderr:")
            print(result.stderr_tail)
        print(
            "hint: put shared secrets in ~/.my-agent/.env "
            "or set them as environment variables."
        )
        print("hint: run `my-agent-core` in the foreground to debug startup.")
        sys.exit(1)

    if result.reason == "timeout":
        print(f"error: core did not become ready at {host}:{port}")
        sys.exit(1)

    if result.reason == "spawn_error":
        print(f"error: failed to start core: {result.stderr_tail}")
        sys.exit(1)

    # Defensive: unknown reason
    print(f"error: core failed to start: {result.reason}")
    sys.exit(1)


def _safe_unlink(path: Path) -> None:
    """删除文件，不存在时静默忽略。"""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# core stop
# ---------------------------------------------------------------------------


class InvalidPidFileError(RuntimeError):
    """PID 文件存在但内容不是正整数。"""


def read_core_pid(pid_file: Path = DEFAULT_CORE_PID_FILE) -> int | None:
    """读取 PID 文件。返回正整数 PID；文件不存在返回 None；内容非法抛 InvalidPidFileError。"""
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        pid = int(raw)
    except ValueError as exc:
        raise InvalidPidFileError(f"invalid pid file: {pid_file}") from exc
    if pid <= 0:
        raise InvalidPidFileError(f"invalid pid file: {pid_file}")
    return pid


def is_process_running(pid: int) -> bool:
    """os.kill(pid, 0) 探测进程是否存在（不会终止进程）。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def terminate_process(pid: int) -> None:
    """向进程发送 SIGTERM。"""
    os.kill(pid, signal.SIGTERM)


async def wait_for_process_exit(pid: int, timeout_s: float = 5.0) -> None:
    """轮询进程是否已退出，超时抛 TimeoutError。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_process_running(pid):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"process {pid} did not exit")


async def wait_for_core_offline(host: str, port: int, timeout_s: float = 5.0) -> None:
    """轮询 probe_core 直到不可连接或超时。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            await probe_core(host, port)
        except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"core still running at {host}:{port}")


# 打印 stale PID + Core online 场景的可行动排查提示；不退出，退出策略由调用方控制
def _print_stale_pid_online_hint(
    host: str,
    port: int,
    dead_pid: int,
    pid_file: Path,
) -> None:
    print(
        f"error: core is running at {host}:{port} but pid file "
        f"points to dead pid={dead_pid}; cannot stop safely"
    )
    print("hint: stale pid file detected while the core port is still online.")
    print("hint: my-agent will not stop a process by port automatically.")
    print("hint: inspect the listening process:")
    print(
        f"  $listener = Get-NetTCPConnection -LocalAddress {host} "
        f"-LocalPort {port} -State Listen"
    )
    print("  $actual_pid = $listener.OwningProcess")
    print(
        '  (Get-CimInstance Win32_Process -Filter '
        '"ProcessId=$actual_pid").CommandLine'
    )
    print(
        "hint: if the command line is `python -m my_agent.core`, "
        "repair the pid file:"
    )
    print(f'  Set-Content -Path "{pid_file}" -Value $actual_pid -Encoding ascii')
    print("  my-agent core stop")


def cmd_core_stop(
    host: str,
    port: int,
    *,
    pid_file: Path = DEFAULT_CORE_PID_FILE,
) -> None:
    """执行 `my-agent core stop`。

    无 PID 文件且离线：打印 not running，返回（退出 0）。
    """
    try:
        pid = read_core_pid(pid_file)
    except InvalidPidFileError:
        try:
            asyncio.run(probe_core(host, port))
        except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError):
            _safe_unlink(pid_file)
            print(
                f"core not running at {host}:{port}; removed invalid pid file"
            )
            return
        print(
            f"error: core is running at {host}:{port} but pid file is invalid; "
            "cannot stop safely"
        )
        sys.exit(1)

    if pid is None:
        try:
            asyncio.run(probe_core(host, port))
        except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError):
            print(f"core not running at {host}:{port}")
            return
        print(
            f"error: core is running at {host}:{port} but no pid file was found"
        )
        sys.exit(1)

    if not is_process_running(pid):
        # PID 文件指向的进程已不存在：根据 Core 当前状态决定
        try:
            asyncio.run(probe_core(host, port))
        except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError):
            _safe_unlink(pid_file)
            print(
                f"core not running at {host}:{port}; removed stale pid file"
            )
            return
        _print_stale_pid_online_hint(host, port, pid, pid_file)
        sys.exit(1)

    # 进程存在：发停止信号 -> 等进程退出 -> 等 Core 离线 -> 清 PID
    terminate_process(pid)
    try:
        asyncio.run(wait_for_process_exit(pid))
    except TimeoutError:
        print(f"error: core process {pid} did not exit")
        sys.exit(1)
    try:
        asyncio.run(wait_for_core_offline(host, port))
    except TimeoutError:
        print(
            f"error: core still running at {host}:{port} after stopping pid={pid}"
        )
        sys.exit(1)
    _safe_unlink(pid_file)
    print(f"core stopped at {host}:{port} pid={pid}")
