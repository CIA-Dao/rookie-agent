from __future__ import annotations

from pathlib import Path

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import WorkspacePathError, resolve_workspace_path

_MAX_BYTES = 1 * 1024 * 1024


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write text content to a file."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the file."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
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

        content = str(params.get("content") or "")

        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_BYTES:
            return ToolResult(
                content=f"content too large: {len(encoded)} bytes (limit 1 MB)",
                is_error=True,
                error_type="runtime_error",
            )
        if path.exists() and path.is_dir():
            return ToolResult(
                content=f"not a file: {path_str}",
                is_error=True,
                error_type="runtime_error",
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(
                content=f"write file error: {e}",
                is_error=True,
                error_type="runtime_error",
            )
        return ToolResult(content=f"wrote {len(encoded)} bytes to {path_str}")
