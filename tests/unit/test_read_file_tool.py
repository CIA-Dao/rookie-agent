from __future__ import annotations

import json
from pathlib import Path

from my_agent.core.tools.builtin import ReadFileTool


async def test_read_file_tool_reads_from_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("# demo", encoding="utf-8")

    result = await ReadFileTool(workspace).invoke({"path": "README.md"})

    assert not result.is_error
    assert result.content == "# demo"


async def test_read_file_tool_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = await ReadFileTool(workspace).invoke({"path": "../secret.txt"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "outside workspace" in result.content


# ── P6: workspace security denylist ───────────────────────────────────────────


async def test_read_file_tool_denies_my_agent_internal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".my-agent").mkdir(parents=True)
    (workspace / ".my-agent" / "context.md").write_text("private context", encoding="utf-8")

    result = await ReadFileTool(workspace).invoke({"path": ".my-agent/context.md"})

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "denied" in result.content
    # Must not leak the actual file content
    assert "private context" not in result.content


async def test_read_file_tool_denies_dotgit_runs_env(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "config").write_text("[user]", encoding="utf-8")
    (workspace / "runs").mkdir(parents=True)
    (workspace / "runs" / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (workspace / ".env").write_text("DEEPSEEK_API_KEY=sk-x", encoding="utf-8")

    for path in (".git/config", "runs/events.jsonl", ".env"):
        result = await ReadFileTool(workspace).invoke({"path": path})
        assert result.is_error, f"expected denial for {path}"
        assert result.error_type == "runtime_error"
        assert "denied" in result.content


async def test_read_file_tool_denies_secret_like_filenames(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("secret.key", "id_rsa", "client.pem", "token.txt"):
        (workspace / name).write_text("payload", encoding="utf-8")

    for name in ("secret.key", "id_rsa", "client.pem", "token.txt"):
        result = await ReadFileTool(workspace).invoke({"path": name})
        assert result.is_error, f"expected denial for {name}"
        assert result.error_type == "runtime_error"
        assert "denied" in result.content


async def test_read_file_tool_keeps_ordinary_project_files_readable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("project guide", encoding="utf-8")
    (workspace / "README.md").write_text("readme", encoding="utf-8")
    (workspace / "pyproject.toml").write_text("[project]", encoding="utf-8")

    for name, expected in (
        ("AGENTS.md", "project guide"),
        ("README.md", "readme"),
        ("pyproject.toml", "[project]"),
    ):
        result = await ReadFileTool(workspace).invoke({"path": name})
        assert not result.is_error, f"unexpected denial for {name}"
        assert result.content == expected


async def test_read_file_tool_keeps_ordinary_project_logs_readable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "logs").mkdir(parents=True)
    (workspace / "logs" / "app.log").write_text("line1\n", encoding="utf-8")
    (workspace / "backend" / "logs").mkdir(parents=True)
    (workspace / "backend" / "logs" / "server.log").write_text("server ok\n", encoding="utf-8")

    for path in ("logs/app.log", "backend/logs/server.log"):
        result = await ReadFileTool(workspace).invoke({"path": path})
        assert not result.is_error, f"unexpected denial for {path}"


async def test_read_file_tool_redirects_large_files_to_ranges(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "large.js").write_bytes(("坦克大战\n" * 4000).encode("utf-8"))

    result = await ReadFileTool(workspace).invoke({"path": "large.js"})

    assert not result.is_error
    payload = json.loads(result.content)
    assert payload["complete"] is False
    assert payload["continuation_required"] is True
    assert payload["next_offset"] > 0
