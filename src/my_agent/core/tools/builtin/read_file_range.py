from __future__ import annotations

from pathlib import Path

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.builtin.read_file import ReadFileTool


class ReadFileRangeTool(BaseTool):
    """Compatibility alias for the unified read_file pagination contract."""

    name = "read_file_range"
    description = "Compatibility alias for read_file with offset and limit."
    input_schema = ReadFileTool.input_schema

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._read_file = ReadFileTool(workspace_root)

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return await self._read_file.invoke({"offset": 0, **params})
