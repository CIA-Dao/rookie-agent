from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from my_agent.core.tools.builtin import WriteFileTool


async def test_write_file_tool_writes_text_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = await WriteFileTool().invoke({"path": "notes/hello.txt", "content": "hello"})

    assert not result.is_error
    assert (tmp_path / "notes" / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert "wrote" in result.content
    assert "notes/hello.txt" in result.content


async def test_write_file_tool_rejects_path_traversal() -> None:
    result = await WriteFileTool().invoke({"path": "../secret.txt", "content": "nope"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "outside workspace" in result.content


async def test_write_file_tool_uses_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = await WriteFileTool(workspace).invoke(
        {"path": "notes/hello.txt", "content": "hello"}
    )

    assert not result.is_error
    assert (workspace / "notes" / "hello.txt").read_text(encoding="utf-8") == "hello"


async def test_write_file_tool_rejects_large_content(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = await WriteFileTool().invoke(
        {"path": "big.txt", "content": "x" * (1024 * 1024 + 1)}
    )

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "content too large" in result.content
    assert not (tmp_path / "big.txt").exists()


# ── P6: workspace security denylist ───────────────────────────────────────────


async def test_write_file_tool_denies_internal_paths_without_creating(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for path in (
        ".my-agent/context.md",
        ".git/config",
        "runs/events.jsonl",
        ".env",
        "secret.key",
        "id_rsa",
    ):
        result = await WriteFileTool(workspace).invoke({"path": path, "content": "nope"})
        assert result.is_error, f"expected denial for {path}"
        assert result.error_type == "runtime_error"
        assert "denied" in result.content

    # Critical: write_file must NOT create parent dirs or files for denied paths.
    assert not (workspace / ".my-agent" / "context.md").exists()
    assert not (workspace / ".my-agent").exists() or not (
        workspace / ".my-agent" / "context.md"
    ).exists()
    assert not (workspace / ".git").exists() or not (workspace / ".git" / "config").exists()
    assert not (workspace / "runs").exists() or not (
        workspace / "runs" / "events.jsonl"
    ).exists()
    assert not (workspace / ".env").exists()
    assert not (workspace / "secret.key").exists()
    assert not (workspace / "id_rsa").exists()


async def test_write_file_tool_keeps_writing_ordinary_project_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = await WriteFileTool(workspace).invoke(
        {"path": "src/app.py", "content": "print('hi')"}
    )

    assert not result.is_error
    assert (workspace / "src" / "app.py").read_text(encoding="utf-8") == "print('hi')"


async def test_write_file_tool_keeps_writing_project_logs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = await WriteFileTool(workspace).invoke(
        {"path": "logs/app.log", "content": "line1\n"}
    )

    assert not result.is_error
    assert (workspace / "logs" / "app.log").read_text(encoding="utf-8") == "line1\n"
