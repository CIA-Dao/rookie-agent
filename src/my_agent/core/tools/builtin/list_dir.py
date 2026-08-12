from __future__ import annotations

from pathlib import Path

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import (
    WorkspacePathError,
    is_denied_internal_path,
    resolve_workspace_path,
)

_MAX_DEPTH = 4
_MAX_ENTRIES = 200


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List the contents of a directory as a tree."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory path."},
            "max_depth": {"type": "integer", "description": "Maximum recursion depth."},
        },
        "required": [],
    }

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = workspace_root

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        path_str = str(params.get("path") or ".")
        try:
            path = resolve_workspace_path(self._workspace_root, path_str)
        except WorkspacePathError as e:
            return ToolResult(
                content=str(e),
                is_error=True,
                error_type="runtime_error",
            )

        try:
            raw_max_depth = params.get("max_depth") or 2
            max_depth = int(str(raw_max_depth))
        except (TypeError, ValueError):
            return ToolResult(
                content="max_depth must be an integer",
                is_error=True,
                error_type="runtime_error",
            )

        max_depth = min(max_depth, _MAX_DEPTH)

        if not path.exists():
            return ToolResult(
                content=f"no such directory: {path_str}",
                is_error=True,
                error_type="runtime_error",
            )

        if not path.is_dir():
            return ToolResult(
                content=f"not a directory: {path_str}",
                is_error=True,
                error_type="runtime_error",
            )

        lines: list[str] = [path_str + "/"]
        count = 0

        def walk(directory: Path, depth: int, prefix: str) -> None:
            nonlocal count
            if depth > max_depth or count >= _MAX_ENTRIES:
                return

            entries = sorted(directory.iterdir(), key=lambda entry: (entry.is_file(), entry.name))

            for index, entry in enumerate(entries):
                if count >= _MAX_ENTRIES:
                    lines.append(f"{prefix}... (truncated)")
                    return

                # P6: redact denied children from the listing entirely. We must not
                # reveal the entry name when it lands on an internal/sensitive path.
                if is_denied_internal_path(self._workspace_root, entry.resolve()):
                    continue

                is_last = index == len(entries) - 1
                connector = "`-- " if is_last else "|-- "
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{prefix}{connector}{entry.name}{suffix}")
                count += 1

                if entry.is_dir() and depth < max_depth:
                    extension = "    " if is_last else "|   "
                    walk(entry, depth + 1, prefix + extension)

        walk(path, 1, "")
        return ToolResult(content="\n".join(lines))
