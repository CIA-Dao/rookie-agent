from __future__ import annotations

from pathlib import Path

import pytest

from my_agent.core.tools.workspace import (
    DENIED_DIR_NAMES,
    SENSITIVE_FILENAME_PATTERNS,
    WorkspacePathError,
    WorkspaceSecurityError,
    is_denied_internal_path,
    resolve_workspace_path,
    workspace_root_or_cwd,
)


def test_workspace_root_defaults_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert workspace_root_or_cwd() == tmp_path.resolve()


def test_resolve_workspace_path_joins_relative_path(tmp_path: Path) -> None:
    assert resolve_workspace_path(tmp_path, "src/app.py") == (tmp_path / "src/app.py").resolve()


def test_resolve_workspace_path_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(WorkspacePathError):
        resolve_workspace_path(tmp_path, "../secret.txt")


def test_resolve_workspace_path_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(WorkspacePathError):
        resolve_workspace_path(tmp_path, str(outside))


# ── P6: workspace security boundary denylist ─────────────────────────────────


def test_denylist_includes_my_agent_internal_dirs() -> None:
    assert ".my-agent" in DENIED_DIR_NAMES
    assert ".git" in DENIED_DIR_NAMES
    assert "runs" in DENIED_DIR_NAMES
    assert "__pycache__" in DENIED_DIR_NAMES
    assert ".venv" in DENIED_DIR_NAMES
    assert "venv" in DENIED_DIR_NAMES
    assert ".pytest_cache" in DENIED_DIR_NAMES
    assert ".ruff_cache" in DENIED_DIR_NAMES
    assert ".mypy_cache" in DENIED_DIR_NAMES
    assert ".tmp-npm-cache" in DENIED_DIR_NAMES


def test_sensitive_filename_patterns_include_secret_like_extensions() -> None:
    # We assert on a representative subset; the table may grow but must include these.
    joined = " ".join(SENSITIVE_FILENAME_PATTERNS)
    assert ".env" in SENSITIVE_FILENAME_PATTERNS or ".env" in joined
    assert "id_rsa" in SENSITIVE_FILENAME_PATTERNS or "id_rsa" in joined
    assert "secret.key" in SENSITIVE_FILENAME_PATTERNS or "secret.key" in joined
    assert "client.pem" in SENSITIVE_FILENAME_PATTERNS or "client.pem" in joined


def test_is_denied_internal_path_rejects_my_agent_subtree(tmp_path: Path) -> None:
    assert is_denied_internal_path(tmp_path, tmp_path / ".my-agent" / "context.md")
    assert is_denied_internal_path(
        tmp_path, tmp_path / ".my-agent" / "skills" / "custom" / "SKILL.md"
    )


def test_is_denied_internal_path_rejects_dotgit_runs_env(tmp_path: Path) -> None:
    assert is_denied_internal_path(tmp_path, tmp_path / ".git" / "config")
    assert is_denied_internal_path(tmp_path, tmp_path / "runs" / "session" / "events.jsonl")
    assert is_denied_internal_path(tmp_path, tmp_path / ".env")
    assert is_denied_internal_path(tmp_path, tmp_path / ".env.local")


def test_is_denied_internal_path_matches_internal_dirs_case_insensitively(
    tmp_path: Path,
) -> None:
    assert is_denied_internal_path(tmp_path, tmp_path / ".GIT" / "config")
    assert is_denied_internal_path(tmp_path, tmp_path / ".MY-AGENT" / "context.md")
    assert is_denied_internal_path(tmp_path, tmp_path / "Runs" / "events.jsonl")


def test_is_denied_internal_path_rejects_secret_like_filenames(tmp_path: Path) -> None:
    assert is_denied_internal_path(tmp_path, tmp_path / "secret.key")
    assert is_denied_internal_path(tmp_path, tmp_path / "id_rsa")
    assert is_denied_internal_path(tmp_path, tmp_path / "client.pem")
    assert is_denied_internal_path(tmp_path, tmp_path / "deploy" / "token.txt")


def test_is_denied_internal_path_rejects_cache_and_venv_dirs(tmp_path: Path) -> None:
    assert is_denied_internal_path(tmp_path, tmp_path / ".venv" / "bin" / "python")
    assert is_denied_internal_path(tmp_path, tmp_path / "venv" / "lib" / "foo.pth")
    assert is_denied_internal_path(tmp_path, tmp_path / "__pycache__" / "app.pyc")
    assert is_denied_internal_path(tmp_path, tmp_path / ".pytest_cache" / "v" / "cache")
    assert is_denied_internal_path(tmp_path, tmp_path / ".mypy_cache" / "0.999" / "x.json")


def test_is_denied_internal_path_keeps_ordinary_project_files_readable(tmp_path: Path) -> None:
    assert not is_denied_internal_path(tmp_path, tmp_path / "AGENTS.md")
    assert not is_denied_internal_path(tmp_path, tmp_path / "README.md")
    assert not is_denied_internal_path(tmp_path, tmp_path / "TODO.md")
    assert not is_denied_internal_path(tmp_path, tmp_path / "pyproject.toml")
    assert not is_denied_internal_path(
        tmp_path, tmp_path / "docs" / "requirements" / "P5-2-tui-visual-polish.md"
    )
    assert not is_denied_internal_path(
        tmp_path, tmp_path / "src" / "my_agent" / "core" / "tools" / "workspace.py"
    )
    assert not is_denied_internal_path(
        tmp_path, tmp_path / "tests" / "unit" / "test_workspace_paths.py"
    )


def test_is_denied_internal_path_keeps_ordinary_project_logs_readable(tmp_path: Path) -> None:
    # P6 rule: ordinary current-project application logs are NOT denied by default.
    assert not is_denied_internal_path(tmp_path, tmp_path / "logs" / "app.log")
    assert not is_denied_internal_path(
        tmp_path, tmp_path / "backend" / "logs" / "server.log"
    )
    assert not is_denied_internal_path(tmp_path, tmp_path / "tests" / "output.log")


def test_is_denied_internal_path_only_applies_within_workspace_root(tmp_path: Path) -> None:
    # Outside the workspace_root, the denylist helper does not opine; callers still
    # gate on resolve_workspace_path() first. Passing a path outside the root with
    # the same denylist stem must not crash and must return False (the resolver
    # already rejects these).
    outside = tmp_path.parent / "other-project" / ".env"
    assert not is_denied_internal_path(tmp_path, outside)


def test_resolve_workspace_path_raises_security_error_for_denied_path(
    tmp_path: Path,
) -> None:
    # P6: resolving a denied internal path raises WorkspaceSecurityError, which is
    # a subclass of WorkspacePathError so existing tool error handling continues
    # to work.
    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, ".my-agent/context.md")

    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, ".git/config")

    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, "runs/events.jsonl")

    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, ".env")

    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, "secret.key")


def test_resolve_workspace_path_keeps_safe_files_resolvable(tmp_path: Path) -> None:
    # P6: ordinary project files still resolve cleanly.
    assert resolve_workspace_path(tmp_path, "README.md") == (tmp_path / "README.md").resolve()
    assert resolve_workspace_path(tmp_path, "src/app.py") == (tmp_path / "src/app.py").resolve()
    assert resolve_workspace_path(tmp_path, "docs/guide.md") == (
        tmp_path / "docs" / "guide.md"
    ).resolve()


def test_workspace_security_error_is_subclass_of_workspace_path_error() -> None:
    # Ensures existing tool error handlers catching WorkspacePathError still work.
    assert issubclass(WorkspaceSecurityError, WorkspacePathError)


def test_resolve_workspace_path_symlink_escape_is_rejected(
    tmp_path: Path,
) -> None:
    # P6: resolved symlink escape outside the workspace root must still be rejected
    # when the platform supports symlinks. We skip silently when symlinks cannot be
    # created (Windows unprivileged accounts, some CI sandboxes).
    outside = tmp_path.parent / "outside-target.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    with pytest.raises(WorkspacePathError):
        resolve_workspace_path(tmp_path, "link.txt")
