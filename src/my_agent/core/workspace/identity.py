from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class WorkspaceCanonicalizationError(ValueError):
    """Raised when a workspace cannot be assigned a safe canonical identity."""


@dataclass(frozen=True, slots=True)
class CanonicalWorkspace:
    """A normalized workspace path and its platform-aware comparison key."""

    path: Path
    key: str


def canonicalize_workspace(workspace_root: str | Path | None) -> CanonicalWorkspace:
    """Resolve a workspace path for ownership comparisons.

    ``Path.resolve(strict=False)`` resolves existing symlink/junction components
    while still allowing a new project directory to be created later. The
    normalized key is used for comparisons so Windows drive and component case
    differences do not bypass the overlap check.
    """

    requested = (
        Path.cwd()
        if workspace_root is None or not str(workspace_root).strip()
        else Path(workspace_root)
    )
    try:
        if requested.is_symlink() and not requested.exists():
            raise WorkspaceCanonicalizationError(
                f"workspace link target does not exist: {workspace_root!s}"
            )
        resolved = requested.expanduser().resolve(strict=False)
    except WorkspaceCanonicalizationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkspaceCanonicalizationError(
            f"cannot canonicalize workspace path: {workspace_root!s}"
        ) from exc

    if not resolved.is_absolute():
        raise WorkspaceCanonicalizationError(
            f"workspace path is not absolute after normalization: {workspace_root!s}"
        )

    normalized = os.path.normpath(str(resolved))
    key = os.path.normcase(normalized)
    return CanonicalWorkspace(path=Path(normalized), key=key)


def workspaces_overlap(left: CanonicalWorkspace, right: CanonicalWorkspace) -> bool:
    """Return whether two canonical paths are equal or ancestor/descendant."""

    try:
        common = os.path.commonpath([left.key, right.key])
    except ValueError:
        # Windows paths on different drives (or incompatible roots) do not
        # overlap.
        return False
    return common == left.key or common == right.key
