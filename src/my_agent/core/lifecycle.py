from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Re-use the low-level helpers that already live in cli/commands/core.py.
# cli/commands/core.py never imports lifecycle, so there's no circular import.
from my_agent.cli.commands.core import (
    DEFAULT_CORE_PID_FILE,
    DEFAULT_CORE_STARTUP_LOG_FILE,
    CoreProbeError,
    CoreProbeResult,
    ProcessExitedError,
    _read_text_tail,
    _safe_unlink,
    _spawn_core_process,
    probe_core,
    wait_for_core_ready_with_proc,
)


@dataclass(frozen=True)
class CoreStartOk:
    """Core is running and reachable."""

    pid: int
    host: str
    port: int
    ready: CoreProbeResult
    already_running: bool = False


@dataclass(frozen=True)
class CoreStartFailed:
    """Structured failure result. Never prints or sys.exits."""

    host: str
    port: int
    reason: str
    exit_code: int | None = None
    stderr_tail: str = ""
    startup_log_file: Path | None = None

    @property
    def missing_deepseek_key(self) -> bool:
        return "DEEPSEEK_API_KEY not set" in self.stderr_tail


async def ensure_core_started(
    host: str,
    port: int,
    *,
    pid_file: Path = DEFAULT_CORE_PID_FILE,
    startup_log_file: Path = DEFAULT_CORE_STARTUP_LOG_FILE,
    on_status: Callable[[str], None] | None = None,
) -> CoreStartOk | CoreStartFailed:
    """Ensure Core daemon is running and reachable.

    Behavior:
    - If probe_core() reports online: return CoreStartOk(already_running=True).
    - Otherwise spawn the daemon, write PID file, wait until ready.
    - On success: return CoreStartOk(already_running=False).
    - On process-exited-before-ready: clean PID, return CoreStartFailed(process_exited).
    - On timeout: clean PID, return CoreStartFailed(timeout).
    - On Popen OSError: return CoreStartFailed(spawn_error) (no PID to clean).

    If on_status is provided, it will be called with a human-readable status
    string at key moments (e.g. right before spawning the daemon).

    This function never prints user-visible text and never calls sys.exit.
    """
    # 1. Already running?
    try:
        ready = await probe_core(host, port)
    except (ConnectionRefusedError, CoreProbeError, OSError, TimeoutError):
        pass
    else:
        return CoreStartOk(
            pid=0,
            host=host,
            port=port,
            ready=ready,
            already_running=True,
        )

    # 2. Spawn daemon process.
    if on_status is not None:
        on_status("core not running, starting automatically...")
    try:
        proc = _spawn_core_process(stderr_file=startup_log_file)
    except OSError as exc:
        return CoreStartFailed(
            host=host,
            port=port,
            reason="spawn_error",
            stderr_tail=str(exc),
            startup_log_file=startup_log_file,
        )

    pid = proc.pid

    # 3. Write PID file (create parent dir).
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{pid}\n", encoding="utf-8")

    # 4. Poll until ready / process exits / timeout.
    try:
        ready = await wait_for_core_ready_with_proc(proc, host, port)
    except ProcessExitedError:
        _safe_unlink(pid_file)
        exit_code = proc.poll()
        stderr_tail = _read_text_tail(startup_log_file)
        return CoreStartFailed(
            host=host,
            port=port,
            reason="process_exited",
            exit_code=exit_code,
            stderr_tail=stderr_tail,
            startup_log_file=startup_log_file,
        )
    except TimeoutError:
        _safe_unlink(pid_file)
        stderr_tail = _read_text_tail(startup_log_file)
        return CoreStartFailed(
            host=host,
            port=port,
            reason="timeout",
            stderr_tail=stderr_tail,
            startup_log_file=startup_log_file,
        )

    return CoreStartOk(
        pid=pid,
        host=host,
        port=port,
        ready=ready,
        already_running=False,
    )


def ensure_core_started_sync(
    host: str,
    port: int,
    *,
    pid_file: Path = DEFAULT_CORE_PID_FILE,
    startup_log_file: Path = DEFAULT_CORE_STARTUP_LOG_FILE,
) -> CoreStartOk | CoreStartFailed:
    """Sync wrapper around ensure_core_started for non-async callers."""
    return asyncio.run(
        ensure_core_started(
            host,
            port,
            pid_file=pid_file,
            startup_log_file=startup_log_file,
        )
    )
