from __future__ import annotations

import json
from pathlib import Path

from my_agent.core.tools.builtin import ReadFileRangeTool


async def test_read_file_range_reconstructs_utf8_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    content = "alpha\n坦克大战\n" * 300
    (workspace / "engine.js").write_bytes(content.encode("utf-8"))
    tool = ReadFileRangeTool(workspace)
    offset = 0
    pieces: list[str] = []
    digest: str | None = None

    while True:
        result = await tool.invoke({"path": "engine.js", "offset": offset, "max_bytes": 97})
        assert not result.is_error
        payload = json.loads(result.content)
        assert payload["offset"] == offset
        pieces.append(payload["content"])
        offset = payload["next_offset"]
        if payload["complete"]:
            digest = payload["sha256"]
            break

    assert "".join(pieces) == content
    assert digest


async def test_read_file_range_rejects_invalid_cursor(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.txt").write_text("坦克", encoding="utf-8")
    tool = ReadFileRangeTool(workspace)

    beyond = await tool.invoke({"path": "data.txt", "offset": 99})
    assert beyond.is_error
    assert "beyond" in beyond.content

    boundary = await tool.invoke({"path": "data.txt", "offset": 1})
    assert boundary.is_error
    assert "UTF-8" in boundary.content
    assert "valid_offset_hint=0" in boundary.content


async def test_read_file_range_preserves_small_file_tool_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.txt").write_text("small", encoding="utf-8")
    result = await ReadFileRangeTool(workspace).invoke({"path": "data.txt"})
    payload = json.loads(result.content)
    assert payload["complete"] is True
    assert payload["content"] == "small"
