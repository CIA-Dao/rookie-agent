from __future__ import annotations

from pathlib import Path


def load_context_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()
