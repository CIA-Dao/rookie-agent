from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from my_agent.core.tools.builtin import FileMetadataTool, FileSearchTool, ProjectBuildTool


async def test_file_metadata_is_platform_neutral(tmp_path: Path) -> None:
    source = tmp_path / "src" / "App.vue"
    source.parent.mkdir()
    source.write_text("<template>ok</template>\n", encoding="utf-8")

    result = await FileMetadataTool(tmp_path).invoke({"path": "src/App.vue"})

    assert not result.is_error
    metadata = json.loads(result.content)
    assert metadata["path"] == "src/App.vue"
    assert metadata["size_bytes"] > 0
    assert metadata["line_count"] == 1
    assert len(metadata["sha256"]) == 64


async def test_file_search_replaces_find_and_returns_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.vue").write_text("", encoding="utf-8")
    (tmp_path / "src" / "main.js").write_text("", encoding="utf-8")
    (tmp_path / ".my-agent").mkdir()
    (tmp_path / ".my-agent" / "hidden.vue").write_text("", encoding="utf-8")

    result = await FileSearchTool(tmp_path).invoke(
        {"path": ".", "patterns": ["*.vue", "*.js"]}
    )

    assert not result.is_error
    payload = json.loads(result.content)
    assert payload["matches"] == ["src/App.vue", "src/main.js"]
    assert payload["truncated"] is False


async def test_file_search_rejects_invalid_pattern_input(tmp_path: Path) -> None:
    result = await FileSearchTool(tmp_path).invoke({"patterns": []})

    assert result.is_error
    assert result.error_type == "schema_error"


async def test_file_tools_reject_paths_outside_workspace(tmp_path: Path) -> None:
    result = await FileMetadataTool(tmp_path).invoke({"path": ".."})

    assert result.is_error
    assert "outside workspace" in result.content


async def test_project_build_uses_declared_script_and_structured_result(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"build": "vite build"}}), encoding="utf-8"
    )
    calls: list[tuple[object, ...]] = []

    async def fake_spawn(*command: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)

        async def communicate() -> tuple[bytes, bytes]:
            return b"build ok", b""

        return SimpleNamespace(returncode=0, communicate=communicate)

    monkeypatch.setattr(
        "my_agent.core.tools.builtin.project_build.asyncio.create_subprocess_exec",
        fake_spawn,
    )
    result = await ProjectBuildTool(tmp_path).invoke({"path": "."})

    assert not result.is_error
    assert calls and calls[0][0:3] == ("npm.cmd", "run", "build")
    assert json.loads(result.content)["exit_code"] == 0


async def test_project_build_reports_missing_script_without_spawning(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {}}), encoding="utf-8")

    result = await ProjectBuildTool(tmp_path).invoke({})

    assert result.is_error
    assert result.error_type == "missing-project-script"
