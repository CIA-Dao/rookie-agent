from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import (
    WorkspacePathError,
    is_denied_internal_path,
    resolve_workspace_path,
)

_DEFAULT_MAX_RESULTS = 200


class FileSearchTool(BaseTool):
    name = "file_search"
    description = (
        "Search workspace filenames with glob patterns. Use this instead of shell "
        "find/dir commands; results are workspace-relative and deterministic."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to search, default '.'."},
            "patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filename globs such as ['*.vue', '*.js'].",
            },
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["patterns"],
    }

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = workspace_root

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        path_str = str(params.get("path") or ".")
        patterns_value = params.get("patterns")
        if not isinstance(patterns_value, list) or not patterns_value or not all(
            isinstance(pattern, str) and pattern for pattern in patterns_value
        ):
            return ToolResult("patterns must be a non-empty string array", True, "schema_error")
        max_results = params.get("max_results", _DEFAULT_MAX_RESULTS)
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or not 1 <= max_results <= 1000
        ):
            return ToolResult("max_results must be an integer from 1 to 1000", True, "schema_error")
        try:
            root = resolve_workspace_path(self._workspace_root, path_str)
        except WorkspacePathError as exc:
            return ToolResult(str(exc), True, "runtime_error")
        if not root.is_dir():
            return ToolResult(f"directory does not exist: {path_str}", True, "runtime_error")

        workspace = Path(self._workspace_root or Path.cwd()).resolve()
        matches: list[str] = []
        truncated = False
        try:
            for candidate in root.rglob("*"):
                if not candidate.is_file() or is_denied_internal_path(workspace, candidate):
                    continue
                relative = candidate.relative_to(workspace).as_posix()
                if any(
                    fnmatch.fnmatch(candidate.name, pattern)
                    or fnmatch.fnmatch(relative, pattern)
                    for pattern in patterns_value
                ):
                    matches.append(relative)
                    if len(matches) >= max_results:
                        truncated = True
                        break
        except PermissionError:
            return ToolResult(
                f"permission denied searching directory: {path_str}",
                True,
                "runtime_error",
            )
        matches.sort()
        return ToolResult(
            json.dumps(
                {
                    "path": path_str,
                    "patterns": patterns_value,
                    "matches": matches,
                    "truncated": truncated,
                },
                ensure_ascii=False,
            )
        )
