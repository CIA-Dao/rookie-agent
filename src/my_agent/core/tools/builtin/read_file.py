from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import WorkspacePathError, resolve_workspace_path

_MAX_RANGE_BYTES = 12_000


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read file content. Small files return text; large files and ranged requests "
        "return a UTF-8-safe envelope. Continue with next_offset until complete."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read."},
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "UTF-8 byte offset returned by the previous read.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_RANGE_BYTES,
                "description": "Maximum UTF-8 bytes to return.",
            },
            "max_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_RANGE_BYTES,
                "description": "Compatibility alias for limit.",
            },
            "snapshot_id": {
                "type": "string",
                "description": "Snapshot identity returned by an earlier range read.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = workspace_root

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        path_str = str(params.get("path") or "")
        offset = params.get("offset", 0)
        limit_value = params.get("limit", params.get("max_bytes", _MAX_RANGE_BYTES))
        snapshot_id = params.get("snapshot_id")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return ToolResult("offset must be a non-negative integer", True, "schema_error")
        if (
            not isinstance(limit_value, int)
            or isinstance(limit_value, bool)
            or not 1 <= limit_value <= _MAX_RANGE_BYTES
        ):
            return ToolResult(
                "limit must be an integer from 1 to 12000", True, "schema_error"
            )
        if snapshot_id is not None and not isinstance(snapshot_id, str):
            return ToolResult("snapshot_id must be a string", True, "schema_error")
        try:
            path = resolve_workspace_path(self._workspace_root, path_str)
        except WorkspacePathError as exc:
            return ToolResult(str(exc), True, "runtime_error")
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return ToolResult(f"file does not exist: {path_str}", True, "runtime_error")
        except PermissionError:
            return ToolResult(
                f"permission denied reading file: {path_str}", True, "runtime_error"
            )

        digest = sha256(data).hexdigest()
        if snapshot_id is not None and snapshot_id != digest:
            return ToolResult(
                f"file changed during ranged read: expected snapshot {snapshot_id}, got {digest}",
                True,
                "runtime_error",
            )
        if offset > len(data):
            return ToolResult(
                f"offset {offset} is beyond file size {len(data)}", True, "schema_error"
            )
        if offset == 0 and len(data) <= _MAX_RANGE_BYTES and "offset" not in params:
            try:
                return ToolResult(data.decode("utf-8"))
            except UnicodeDecodeError:
                return ToolResult(f"file is not valid UTF-8: {path_str}", True, "runtime_error")

        try:
            data[:offset].decode("utf-8")
        except UnicodeDecodeError:
            hint = offset
            while hint > 0 and data[hint] & 0xC0 == 0x80:
                hint -= 1
            return ToolResult(
                f"offset {offset} is not a UTF-8 character boundary; "
                f"retry with the previous returned next_offset or valid_offset_hint={hint}",
                True,
                "schema_error",
            )
        end = min(offset + limit_value, len(data))
        while end > offset:
            try:
                content = data[offset:end].decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            return ToolResult(
                "limit is too small for the next UTF-8 character", True, "schema_error"
            )
        return ToolResult(
            json.dumps(
                {
                    "path": path_str,
                    "offset": offset,
                    "next_offset": end,
                    "total_bytes": len(data),
                    "complete": end == len(data),
                    "continuation_required": end != len(data),
                    "sha256": digest if end == len(data) else None,
                    "snapshot_id": digest,
                    "content": content,
                },
                ensure_ascii=False,
            )
        )
