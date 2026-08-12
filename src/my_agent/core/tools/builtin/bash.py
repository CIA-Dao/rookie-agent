from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from my_agent.core.runtime.windows import WindowsRuntime
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import workspace_root_or_cwd

_MAX_OUTPUT_BYTES = 64 * 1024
_DEFAULT_TIMEOUT = 30.0
_MAX_TIMEOUT = 120.0
_CD_RE = re.compile(r"(?:^|[;&|]\s*)(?:cd|Set-Location)\s+([^;&|\n]+)", re.IGNORECASE)
_UNIX_INSPECTION_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:wc|find|tail|head)\b", re.IGNORECASE
)


def _decode_process_output(data: bytes) -> str:
    """Backward-compatible wrapper for the shared runtime decoder."""
    return WindowsRuntime.decode_output(data)


class BashTool(BaseTool):
    name = "bash"
    description = "Execute a non-interactive command and return stdout and stderr."
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "string"},
                ],
                "description": (
                    "Command and arguments to execute. Use an array for a direct "
                    "process invocation or a string for shell syntax."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Maximum seconds to wait.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = workspace_root
        self._runtime = WindowsRuntime(workspace_root)

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        command = params.get("command") or []
        try:
            timeout = float(str(params.get("timeout") or _DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            return ToolResult(
                content="timeout must be a number",
                is_error=True,
                error_type="runtime_error",
            )


        if not isinstance(command, (list, str)):
            return ToolResult(
                content="command must be a list of arguments or a shell string",
                is_error=True,
                error_type="schema_error",
            )
        if not command or (isinstance(command, list) and not command):
            return ToolResult(
                content="command must not be empty",
                is_error=True,
                error_type="runtime_error",
            )

        if timeout > _MAX_TIMEOUT:
            return ToolResult(
                content=f"timeout must be <= {_MAX_TIMEOUT}",
                is_error=True,
                error_type="runtime_error",
            )
        cwd = workspace_root_or_cwd(self._workspace_root)
        if isinstance(command, str):
            if os.name == "nt" and _UNIX_INSPECTION_RE.search(command):
                return ToolResult(
                    "unsupported-platform-command: use file_metadata/file_search "
                    "or an explicit PowerShell inspection command on Windows",
                    is_error=True,
                    error_type="unsupported-platform-command",
                )
            invalid_cwd = _outside_requested_cwd(command, cwd)
            if invalid_cwd is not None:
                return ToolResult(
                    content=(
                        f"command requested cwd outside workspace: {invalid_cwd}; "
                        f"effective cwd is {cwd}"
                    ),
                    is_error=True,
                    error_type="path_error",
                )
        try:
            if isinstance(command, str):
                if (
                    os.name == "nt"
                    and re.search(r"(?:^|\s)bash(?:\.exe)?\s+-lc\b", command)
                    and not await self._runtime.bash_is_usable()
                ):
                    return ToolResult(
                        "shell_unavailable: bash.exe is not installed; use PowerShell "
                        "or a structured command",
                        True,
                        "shell_unavailable",
                    )
                shell_argv = self._runtime.shell_argv(command)
                if shell_argv is None:
                    process = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=cwd,
                    )
                else:
                    process = await asyncio.create_subprocess_exec(
                        *shell_argv,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=cwd,
                    )
            else:
                process = await asyncio.create_subprocess_exec(
                *self._runtime.normalize_argv([str(part) for part in command]),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult(
                    content=f"[timeout after {timeout}s]",
                    is_error=True,
                    error_type="timeout",
                )
            output = self._runtime.decode_output(stdout_bytes)
            return_code = process.returncode or 0
            if return_code != 0:
                error_type = (
                    "process_start_error" if return_code == 3_221_225_794 else "runtime_error"
                )
                return ToolResult(
                    content=f"[cwd {cwd}] [exit {return_code}]\n{output}",
                    is_error=True,
                    error_type=error_type,
                )
            return ToolResult(content=output or "[no output]")

        except OSError as e:
            return ToolResult(
                content=f"[cwd {cwd}] {e}",
                is_error=True,
                error_type="process_start_error",
            )


def _outside_requested_cwd(command: str, workspace: Path) -> str | None:
    for match in _CD_RE.finditer(command):
        raw = match.group(1).strip().strip('"\'')
        if not raw:
            continue
        requested = Path(raw).expanduser()
        if not requested.is_absolute():
            requested = workspace / requested
        try:
            requested.resolve().relative_to(workspace.resolve())
        except ValueError:
            return raw
    return None


def _prepare_shell_command(command: str) -> str:
    """Make Windows cmd shell output deterministic for UTF-8 diagnostics."""
    return WindowsRuntime().prepare_shell_command(command)
