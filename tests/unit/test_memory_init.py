from __future__ import annotations

from pathlib import Path

import pytest

from my_agent.core.memory.init import InitResult, cmd_init


@pytest.fixture(autouse=True)
def _no_real_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """确保单元测试不调用真实 LLM provider。"""
    monkeypatch.setattr("my_agent.core.memory.init.create_init_provider", lambda _model: None)


def test_init_creates_my_agent_dir_and_context_md(tmp_path: Path) -> None:
    """空项目运行 init 创建 .my-agent/ 目录和 context.md。"""
    root = tmp_path / "testproj"
    root.mkdir()

    cmd_init(root)

    my_agent_dir = root / ".my-agent"
    context_file = my_agent_dir / "context.md"

    assert my_agent_dir.is_dir()
    assert context_file.is_file()

    content = context_file.read_text(encoding="utf-8")
    assert "# Project Context" in content
    assert "Project structure" in content or "structure" in content.lower()
    assert "Memory update" in content or "merge" in content.lower()


def test_init_creates_agents_md(tmp_path: Path) -> None:
    """空项目运行 init 创建 AGENTS.md。"""
    root = tmp_path / "testproj"
    root.mkdir()

    cmd_init(root)

    agents_file = root / "AGENTS.md"
    assert agents_file.is_file()
    content = agents_file.read_text(encoding="utf-8")
    assert ".my-agent/context.md" in content


def test_init_preserves_existing_agents_md_content(tmp_path: Path) -> None:
    """已有 AGENTS.md 时 init 保留用户内容并加入 my-agent marker 段。"""
    root = tmp_path / "testproj"
    root.mkdir()

    user_content = "# My Project\n\nThese are custom rules.\n"
    (root / "AGENTS.md").write_text(user_content, encoding="utf-8")

    cmd_init(root)

    content = (root / "AGENTS.md").read_text(encoding="utf-8")
    # 用户原有内容保留
    assert "My Project" in content
    assert "custom rules" in content
    # marker 段被追加
    assert "<!-- MY-AGENT:BEGIN -->" in content
    assert "<!-- MY-AGENT:END -->" in content


def test_init_no_duplicate_markers_on_rerun(tmp_path: Path) -> None:
    """重复运行 init 不产生重复 marker 段。"""
    root = tmp_path / "testproj"
    root.mkdir()

    # 第一次 init 创建 AGENTS.md
    cmd_init(root)
    content1 = (root / "AGENTS.md").read_text(encoding="utf-8")
    assert content1.count("<!-- MY-AGENT:BEGIN -->") == 1

    # 第二次 init 不应追加第二个 marker
    cmd_init(root)
    content2 = (root / "AGENTS.md").read_text(encoding="utf-8")
    assert content2.count("<!-- MY-AGENT:BEGIN -->") == 1
    assert content2.count("<!-- MY-AGENT:END -->") == 1


def test_context_md_contains_memory_rules(tmp_path: Path) -> None:
    """context.md 包含 memory update / merge 规则。"""
    root = tmp_path / "testproj"
    root.mkdir()

    cmd_init(root)

    content = (root / ".my-agent" / "context.md").read_text(encoding="utf-8")
    assert "Memory update" in content or "merge" in content.lower()
    assert "Preserve human-edited content" in content or "保留人工编辑" in content
    assert "tentative" in content


def test_cmd_init_returns_init_result(tmp_path: Path) -> None:
    """cmd_init 返回 InitResult 而非打印到 stdout。"""
    root = tmp_path / "testproj"
    root.mkdir()

    result = cmd_init(root)

    assert isinstance(result, InitResult)
    assert isinstance(result.messages, list)
    assert len(result.messages) > 0
    assert any("Initialized project at" in msg for msg in result.messages)
