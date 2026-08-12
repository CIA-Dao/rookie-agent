from __future__ import annotations

from pathlib import Path

from my_agent.core.agents.loader import AgentProfileLoader


def test_load_builtin_profile() -> None:
    profile = AgentProfileLoader().load("reviewer")

    assert profile is not None
    assert profile.name == "reviewer"
    assert "review" in profile.description.lower()
    assert "read_file" in profile.allowed_tools
    assert profile.system_prompt


def test_unknown_profile_returns_none(tmp_path: Path) -> None:
    assert AgentProfileLoader(tmp_path).load("missing-profile") is None


def test_project_profile_overrides_builtin(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".my-agent" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "reviewer.toml").write_text(
        """[agent]
description = "Local reviewer"
system_prompt = "Local prompt"
allowed_tools = ["list_dir"]
model = "local-model"
""",
        encoding="utf-8",
    )

    profile = AgentProfileLoader(tmp_path).load("reviewer")

    assert profile is not None
    assert profile.description == "Local reviewer"
    assert profile.system_prompt == "Local prompt"
    assert profile.allowed_tools == ["list_dir"]
    assert profile.model == "local-model"
