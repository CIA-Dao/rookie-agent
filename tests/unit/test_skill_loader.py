from __future__ import annotations

from pathlib import Path

from my_agent.core.skills.loader import Skill, SkillLoader, _parse_skill_file


def test_parse_skill_file_reads_frontmatter_and_body(tmp_path: Path) -> None:
    path = tmp_path / "custom.md"
    path.write_text(
        """---
name: custom
description: Custom skill
allowed_tools:
  - read_file
  - list_dir
---

Use this target:
$ARGUMENTS
""",
        encoding="utf-8",
    )

    skill = _parse_skill_file(path)

    assert skill.name == "custom"
    assert skill.description == "Custom skill"
    assert skill.allowed_tools == ["read_file", "list_dir"]
    assert "Use this target:" in skill.system_prompt_template


def test_parse_skill_file_without_frontmatter_uses_filename(tmp_path: Path) -> None:
    path = tmp_path / "plain.md"
    path.write_text("Plain prompt", encoding="utf-8")

    skill = _parse_skill_file(path)

    assert skill.name == "plain"
    assert skill.description == ""
    assert skill.allowed_tools == []
    assert skill.system_prompt_template == "Plain prompt"


def test_resolve_finds_builtin_skill() -> None:
    skill = SkillLoader().resolve("review")

    assert skill is not None
    assert skill.name == "review"
    assert "review" in skill.description.lower()
    assert "read_file" in skill.allowed_tools


def test_builtin_orchestrate_skill_exposes_subagent_tools() -> None:
    skill = SkillLoader().resolve("orchestrate")

    assert skill is not None
    assert skill.name == "orchestrate"
    assert "spawn_agent" in skill.allowed_tools
    assert "agent_result" in skill.allowed_tools
    assert "task_create" in skill.allowed_tools


def test_builtin_orchestrate_skill_describes_three_stage_workflow() -> None:
    skill = SkillLoader().resolve("orchestrate")

    assert skill is not None
    prompt = skill.system_prompt_template
    assert "planner" in prompt
    assert "executor" in prompt
    assert "reviewer" in prompt
    assert prompt.count("Call spawn_agent") == 3


def test_project_skill_overrides_builtin(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".my-agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "review.md").write_text(
        """---
name: review
description: Local review
---

Local prompt
""",
        encoding="utf-8",
    )

    skill = SkillLoader(tmp_path).resolve("review")

    assert skill is not None
    assert skill.description == "Local review"
    assert skill.system_prompt_template == "Local prompt"


def test_list_all_skills_returns_effective_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".my-agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "review.md").write_text(
        """---
name: review
description: Local review
---

Local prompt
""",
        encoding="utf-8",
    )

    skills = SkillLoader(tmp_path).list_all_skills()
    skills_by_name = {skill.name: skill for skill in skills}

    assert "orchestrate" in skills_by_name
    assert "review" in skills_by_name
    assert skills_by_name["review"].description == "Local review"
    assert list(skills_by_name) == sorted(skills_by_name)


def test_resolve_supports_directory_skill_file(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".my-agent" / "skills" / "custom"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: custom
description: Directory skill
---

Directory prompt
""",
        encoding="utf-8",
    )

    skill = SkillLoader(tmp_path).resolve("custom")

    assert skill is not None
    assert skill.description == "Directory skill"
    assert skill.system_prompt_template == "Directory prompt"


def test_resolve_returns_none_for_unknown_skill(tmp_path: Path) -> None:
    assert SkillLoader(tmp_path).resolve("missing-skill") is None


def test_render_prompt_replaces_arguments_placeholder() -> None:
    loader = SkillLoader()
    skill = Skill(
        name="review",
        description="Review",
        system_prompt_template="Review target:\n$ARGUMENTS",
    )

    rendered = loader.render_prompt(skill, "src/foo.py")

    assert rendered == "Review target:\nsrc/foo.py"
