from __future__ import annotations

from pathlib import Path

from my_agent.core.memory.loader import load_context_file


def test_load_context_file_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_context_file(tmp_path / "missing.md") == ""


def test_load_context_file_returns_stripped_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "context.md"
    path.write_text("\nProject context\n中文内容\n\n", encoding="utf-8")

    assert load_context_file(path) == "Project context\n中文内容"


def test_load_context_file_returns_empty_for_whitespace_only_file(tmp_path: Path) -> None:
    path = tmp_path / "context.md"
    path.write_text("  \n\t\n", encoding="utf-8")

    assert load_context_file(path) == ""
