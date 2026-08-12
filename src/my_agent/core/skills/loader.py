from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    system_prompt_template: str
    allowed_tools: list[str] = field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_skill_file(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    name = path.stem
    description = ""
    allowed_tools: list[str] = []
    body = text

    match = _FRONTMATTER_RE.match(text)
    if match:
        frontmatter = match.group(1)
        body = text[match.end():]

        for line in frontmatter.splitlines():
            stripped = line.strip()
            if stripped.startswith("name:"):
                name = stripped.removeprefix("name:").strip().strip('"').strip("'")
            elif stripped.startswith("description:"):
                description = (
                    stripped.removeprefix("description:")
                    .strip()
                    .strip('"')
                    .strip("'")
                )
            elif stripped.startswith("- "):
                allowed_tools.append(stripped.removeprefix("- ").strip())

    return Skill(
        name=name,
        description=description,
        system_prompt_template=body.strip(),
        allowed_tools=allowed_tools,
    )


class SkillLoader:
    _BUILTIN_DIR = Path(__file__).parent / "builtin"

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = (
            Path(workspace_root).expanduser()
            if workspace_root
            else Path.cwd()
        )

    def resolve(self, name: str) -> Skill | None:
        for path in self._search_paths(name):
            if path.exists():
                return _parse_skill_file(path)
        return None

    def list_all_skills(self) -> list[Skill]:
        skills_by_name: dict[str, Skill] = {}
        for directory in reversed(self._skill_dirs()):
            for path in self._iter_skill_files(directory):
                skill = _parse_skill_file(path)
                skills_by_name[skill.name] = skill
        return [skills_by_name[name] for name in sorted(skills_by_name)]

    def _skill_dirs(self) -> list[Path]:
        return [
            self._workspace_root / ".my-agent" / "skills",
            Path("~/.my-agent/skills").expanduser(),
            self._BUILTIN_DIR,
        ]

    def _iter_skill_files(self, directory: Path) -> list[Path]:
        try:
            if not directory.exists():
                return []
            children = list(directory.iterdir())
        except OSError:
            return []

        paths: list[Path] = []
        for child in children:
            if child.is_file() and child.suffix == ".md":
                paths.append(child)
            elif child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.exists():
                    paths.append(skill_file)
        return sorted(paths)

    def _search_paths(self, name: str) -> list[Path]:
        project_dir, global_dir, builtin_dir = self._skill_dirs()

        return [
            project_dir / f"{name}.md",
            project_dir / name / "SKILL.md",
            global_dir / f"{name}.md",
            global_dir / name / "SKILL.md",
            builtin_dir / f"{name}.md",
            builtin_dir / name / "SKILL.md",
        ]

    def render_prompt(self, skill: Skill, arguments: str) -> str:
        return skill.system_prompt_template.replace("$ARGUMENTS", arguments)
