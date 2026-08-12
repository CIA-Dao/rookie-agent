from __future__ import annotations

import platform
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CommandEvidence(BaseModel):
    """Optional execution context attached to the existing tool-call events."""

    model_config = ConfigDict(extra="ignore")

    run_id: str | None = None
    step: int | None = None
    tool_use_id: str | None = None
    intent: str | None = None
    description: str | None = None
    original_command: str | list[str] | None = None
    normalized_executable: str | None = None
    normalized_argv: list[str] = Field(default_factory=list)
    shell_mode: bool = False
    cwd: str | None = None
    platform: str | None = None
    attempt: int = 1
    process_started: bool = False
    error_type: str | None = None
    recovery: str | None = None


def evidence_for_tool_call(
    tool_name: str,
    params: dict[str, Any],
    *,
    run_id: str | None = None,
    step: int | None = None,
    tool_use_id: str | None = None,
    workspace_root: Path | str | None = None,
) -> CommandEvidence | None:
    if tool_name != "bash" or "command" not in params:
        return None
    command = params["command"]
    if isinstance(command, list):
        argv = [str(part) for part in command]
    else:
        try:
            argv = shlex.split(str(command), posix=False)
        except ValueError:
            argv = [str(command)]
    return CommandEvidence(
        run_id=run_id,
        step=step,
        tool_use_id=tool_use_id,
        original_command=command if isinstance(command, (str, list)) else str(command),
        normalized_executable=argv[0] if argv else None,
        normalized_argv=argv,
        shell_mode=isinstance(command, str),
        cwd=str(Path(workspace_root).resolve()) if workspace_root is not None else None,
        platform=platform.system().lower(),
    )
