from __future__ import annotations

import os
from pathlib import Path

from my_agent.core.workspace.identity import canonicalize_workspace, workspaces_overlap


def test_workspace_identity_resolves_dot_and_dotdot(tmp_path: Path) -> None:
    project = tmp_path / "project"
    nested = project / "nested"

    direct = canonicalize_workspace(project)
    normalized = canonicalize_workspace(nested / "..")

    assert direct.path == normalized.path
    assert direct.key == normalized.key


def test_workspace_identity_uses_path_boundaries(tmp_path: Path) -> None:
    project = canonicalize_workspace(tmp_path / "project")
    project_copy = canonicalize_workspace(tmp_path / "project-copy")

    assert not workspaces_overlap(project, project_copy)


def test_workspace_identity_detects_parent_and_child(tmp_path: Path) -> None:
    parent = canonicalize_workspace(tmp_path / "project")
    child = canonicalize_workspace(tmp_path / "project" / "nested")

    assert workspaces_overlap(parent, child)
    assert workspaces_overlap(child, parent)


def test_workspace_identity_normalizes_windows_case(tmp_path: Path) -> None:
    project = tmp_path / "Project"
    direct = canonicalize_workspace(project)
    differently_cased = canonicalize_workspace(str(project).upper())

    if os.name == "nt":
        assert direct.key == differently_cased.key
    else:
        assert direct.key != differently_cased.key


def test_workspace_identity_allows_different_windows_drives() -> None:
    drive_c = canonicalize_workspace("C:/projects/tank")
    drive_d = canonicalize_workspace("D:/projects/tank")

    assert not workspaces_overlap(drive_c, drive_d)
