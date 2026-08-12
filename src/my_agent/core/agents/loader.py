from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentProfile:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    model: str = ""


class AgentProfileLoader:
    _BUILTIN_DIR = Path(__file__).parent / "builtin"

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = (
            Path(workspace_root).expanduser()
            if workspace_root
            else Path.cwd()
        )

    def load(self, name: str) -> AgentProfile | None:
        for path in self._search_paths(name):
            if path.exists():
                return self._parse(path, name)

        return None

    def _search_paths(self, name: str) -> list[Path]:
        project_dir = self._workspace_root / ".my-agent" / "agents"
        global_dir = Path("~/.my-agent/agents").expanduser()
        builtin_dir = self._BUILTIN_DIR

        return [
            project_dir / f"{name}.toml",
            global_dir / f"{name}.toml",
            builtin_dir / f"{name}.toml",
        ]

    def _parse(self, path: Path, name: str) -> AgentProfile:
        with path.open("rb") as f:
            data = tomllib.load(f)

        agent = data.get("agent", {})

        return AgentProfile(
            name=name,
            description=str(agent.get("description", "")),
            system_prompt=str(agent.get("system_prompt", "")).strip(),
            allowed_tools=[str(x) for x in agent.get("allowed_tools", [])],
            model=str(agent.get("model", "")),
        )
