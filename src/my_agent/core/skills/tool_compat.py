from __future__ import annotations

from dataclasses import dataclass, field

from my_agent.core.tools.catalog import BUILTIN_TOOL_NAMES as BUILTIN_TOOL_NAMES

# P6: explicit, conservative alias table mapping third-party Skill tool names to
# My Agent built-in tool names. Only exact-string aliases live here; there is no
# fuzzy matching, no prefix normalization, and no prompt-text guessing.
#
# Examples such as `grep`, `browser.open`, `search`, or `fetch` are intentionally
# NOT in this table. They remain unresolved unless a future requirement adds an
# explicit, safe mapping.
TOOL_ALIASES: dict[str, str] = {
    "shell": "bash",
    "bash": "bash",
    "shell.exec": "bash",
    "file.read": "read_file",
    "read": "read_file",
    "file.write": "write_file",
    "write": "write_file",
    "file.list": "list_dir",
    "list": "list_dir",
}


@dataclass
class SkillToolCompatibility:
    # P6 兼容性解析结果：保留 None 与 [] 的语义差异，保留声明顺序，去重后输出
    declared_tools: list[str] | None
    available_tools: list[str]
    resolved_tools: list[str] = field(default_factory=list)
    aliases: list[dict[str, str]] = field(default_factory=list)
    unresolved_tools: list[str] = field(default_factory=list)

    @property
    def unrestricted_by_skill(self) -> bool:
        # True 表示 Skill 未声明 allowed_tools，因此 SessionManager 不施加白名单。
        # False 表示 Skill 声明了 allowed_tools（即便解析后为空），SessionManager
        # 必须按 resolved_tools 限制工具集，而不是退回到全部工具。
        return self.declared_tools is None

    @property
    def has_diagnostics(self) -> bool:
        # 仅在确实存在 alias 或 unresolved 时才需要发出诊断事件。
        return bool(self.aliases) or bool(self.unresolved_tools)


def resolve_allowed_tools(
    declared: list[str] | None,
    available: list[str],
) -> SkillToolCompatibility:
    # 纯函数：根据 Skill 声明的 allowed_tools 和当前注册表可用的工具名，给出
    # 解析后的本地工具白名单、命中的 alias 映射、未解析的声明项。
    if declared is None:
        return SkillToolCompatibility(
            declared_tools=None,
            available_tools=list(available),
        )

    available_set = set(available)
    resolved: list[str] = []
    resolved_seen: set[str] = set()
    aliases: list[dict[str, str]] = []
    aliases_seen: set[tuple[str, str]] = set()
    unresolved: list[str] = []

    for name in declared:
        # 1) Exact local name wins first.
        if name in available_set:
            if name not in resolved_seen:
                resolved.append(name)
                resolved_seen.add(name)
            continue

        # 2) Conservative alias/capability table lookup.
        alias_target = TOOL_ALIASES.get(name)
        if alias_target is not None and alias_target in available_set:
            if alias_target not in resolved_seen:
                resolved.append(alias_target)
                resolved_seen.add(alias_target)
            key = (name, alias_target)
            if key not in aliases_seen:
                aliases.append({"from": name, "to": alias_target})
                aliases_seen.add(key)
            continue

        # 3) Anything else (including grep/browser.open/search/fetch, or alias
        #    targets not present in this registry) is unresolved.
        if name not in unresolved:
            unresolved.append(name)

    return SkillToolCompatibility(
        declared_tools=list(declared),
        available_tools=list(available),
        resolved_tools=resolved,
        aliases=aliases,
        unresolved_tools=unresolved,
    )


def available_tool_names_for_runner(runner: object) -> list[str]:
    available_tool_names = getattr(runner, "available_tool_names", None)
    if callable(available_tool_names):
        return [str(name) for name in available_tool_names()]
    return list(BUILTIN_TOOL_NAMES)
