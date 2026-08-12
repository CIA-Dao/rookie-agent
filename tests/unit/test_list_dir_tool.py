from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from my_agent.core.tools.builtin import ListDirTool


async def test_list_dir_tool_lists_directory_tree(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    result = await ListDirTool().invoke({"path": ".", "max_depth": 2})

    assert not result.is_error
    assert "./" in result.content
    assert "README.md" in result.content
    assert "src/" in result.content
    assert "main.py" in result.content


async def test_list_dir_tool_rejects_path_traversal() -> None:
    result = await ListDirTool().invoke({"path": "../secret"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "outside workspace" in result.content


async def test_list_dir_tool_uses_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("# demo", encoding="utf-8")

    result = await ListDirTool(workspace).invoke({"path": ".", "max_depth": 1})

    assert not result.is_error
    assert "README.md" in result.content


# ── P6: workspace security denylist ───────────────────────────────────────────


async def test_list_dir_tool_denies_internal_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".my-agent").mkdir(parents=True)
    (workspace / ".git").mkdir(parents=True)
    (workspace / "runs").mkdir(parents=True)

    for path in (".my-agent", ".git", "runs"):
        result = await ListDirTool(workspace).invoke({"path": path, "max_depth": 1})
        assert result.is_error, f"expected denial for {path}"
        assert result.error_type == "runtime_error"
        assert "denied" in result.content


async def test_list_dir_tool_redacts_denied_children_in_allowed_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Mix of allowed and denied children at the workspace root.
    (workspace / "README.md").write_text("ok", encoding="utf-8")
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("x", encoding="utf-8")
    (workspace / ".my-agent").mkdir()
    (workspace / ".my-agent" / "context.md").write_text("private", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / "runs").mkdir()
    (workspace / ".env").write_text("sk-x", encoding="utf-8")
    (workspace / "secret.key").write_text("k", encoding="utf-8")

    result = await ListDirTool(workspace).invoke({"path": ".", "max_depth": 1})

    assert not result.is_error
    # Allowed children remain visible
    assert "README.md" in result.content
    assert "src/" in result.content
    # Denied entries must NOT be exposed by name in a way that leaks their existence
    # as concrete entries in the tree; allow a compact redacted marker instead.
    assert ".my-agent/" not in result.content
    assert ".git/" not in result.content
    assert "runs/" not in result.content
    assert ".env" not in result.content
    assert "secret.key" not in result.content


async def test_list_dir_tool_keeps_listing_ordinary_project_logs(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "logs").mkdir(parents=True)
    (workspace / "logs" / "app.log").write_text("line\n", encoding="utf-8")

    result = await ListDirTool(workspace).invoke({"path": "logs", "max_depth": 1})

    assert not result.is_error
    assert "app.log" in result.content
