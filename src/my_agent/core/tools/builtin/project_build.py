from __future__ import annotations

import asyncio
import json
from pathlib import Path

from my_agent.core.runtime.windows import WindowsRuntime
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import WorkspacePathError, resolve_workspace_path

_DEFAULT_TIMEOUT = 120.0
_MAX_OUTPUT = 16_000


class ProjectBuildTool(BaseTool):
    name = "project_build"
    description = (
        "Run a project's declared build script using its package manager. This is "
        "a bounded one-shot delivery check, not a long-running dev server."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Project directory, default '.'."},
            "script": {"type": "string", "description": "Package script, default 'build'."},
            "timeout": {"type": "number", "minimum": 1, "maximum": 600},
        },
        "required": [],
    }

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = workspace_root
        self._runtime = WindowsRuntime(workspace_root)

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        path_str = str(params.get("path") or ".")
        script = str(params.get("script") or "build")
        try:
            timeout = float(str(params.get("timeout") or _DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            return ToolResult("timeout must be a number", True, "schema_error")
        if not 1 <= timeout <= 600 or not script.strip():
            return ToolResult(
                "script must be non-empty and timeout must be 1..600",
                True,
                "schema_error",
            )
        try:
            project = resolve_workspace_path(self._workspace_root, path_str)
        except WorkspacePathError as exc:
            return ToolResult(str(exc), True, "runtime_error")
        package_json = project / "package.json"
        if not package_json.is_file():
            return ToolResult(
                "package.json not found; project build adapter unavailable",
                True,
                "runtime_error",
            )
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return ToolResult(f"unable to read package.json: {exc}", True, "runtime_error")
        scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
        if not isinstance(scripts, dict) or script not in scripts:
            return ToolResult(f"missing project script: {script}", True, "missing-project-script")

        executable = _package_manager(project)
        command = self._runtime.normalize_argv([executable, "run", script])
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=project,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                output_bytes, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult(
                    json.dumps(_result(command, timeout, None, "timeout", "")),
                    True,
                    "timeout",
                )
        except OSError as exc:
            return ToolResult(str(exc), True, "process_start_error")
        output = output_bytes.decode("utf-8", errors="replace")
        exit_code = process.returncode or 0
        payload = _result(
            command,
            timeout,
            exit_code,
            None if exit_code == 0 else "non-zero-exit",
            output,
        )
        return ToolResult(
            json.dumps(payload, ensure_ascii=False),
            is_error=exit_code != 0,
            error_type=None if exit_code == 0 else "runtime_error",
        )


def _package_manager(project: Path) -> str:
    if (project / "pnpm-lock.yaml").is_file():
        return "pnpm"
    if (project / "yarn.lock").is_file():
        return "yarn"
    return "npm"


def _result(
    command: list[str],
    timeout: float,
    exit_code: int | None,
    error: str | None,
    output: str,
) -> dict[str, object]:
    return {
        "command": command,
        "timeout_seconds": timeout,
        "exit_code": exit_code,
        "error": error,
        "output": output[-_MAX_OUTPUT:],
    }
