from __future__ import annotations

import json
import platform
from hashlib import sha256
from pathlib import Path

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import WorkspacePathError, resolve_workspace_path


class FileMetadataTool(BaseTool):
    name = "file_metadata"
    description = (
        "Inspect a workspace file without shell commands. Returns size, line count, "
        "SHA-256, and modification time."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path."}},
        "required": ["path"],
    }

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = workspace_root

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        path_str = str(params.get("path") or "")
        try:
            path = resolve_workspace_path(self._workspace_root, path_str)
        except WorkspacePathError as exc:
            return ToolResult(str(exc), True, "runtime_error")
        if not path.is_file():
            return ToolResult(f"file does not exist: {path_str}", True, "runtime_error")
        try:
            data = path.read_bytes()
            stat = path.stat()
        except PermissionError:
            return ToolResult(f"permission denied reading file: {path_str}", True, "runtime_error")
        result = {
            "path": path_str,
            "size_bytes": len(data),
            "line_count": len(data.splitlines()),
            "sha256": sha256(data).hexdigest(),
            "modified_ns": stat.st_mtime_ns,
            "platform": platform.system().lower(),
        }
        return ToolResult(json.dumps(result, ensure_ascii=False))
