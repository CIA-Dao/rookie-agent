from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class ToolPolicy:
    default: PermissionDecision
    allow_patterns: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)


DEFAULT_POLICIES: dict[str, ToolPolicy] = {
    "bash": ToolPolicy(default=PermissionDecision.ASK),
    "write_file": ToolPolicy(default=PermissionDecision.ASK),
    "write_file_begin": ToolPolicy(default=PermissionDecision.ASK),
    "write_file_chunk": ToolPolicy(default=PermissionDecision.ASK),
    "write_file_commit": ToolPolicy(default=PermissionDecision.ASK),
    "read_file": ToolPolicy(default=PermissionDecision.ALLOW),
    "read_file_range": ToolPolicy(default=PermissionDecision.ALLOW),
    "list_dir": ToolPolicy(default=PermissionDecision.ALLOW),
    "file_metadata": ToolPolicy(default=PermissionDecision.ALLOW),
    "file_search": ToolPolicy(default=PermissionDecision.ALLOW),
    "project_build": ToolPolicy(default=PermissionDecision.ASK),
    "note_save": ToolPolicy(default=PermissionDecision.ALLOW),
}

_PREVIEW_KEY: dict[str, str] = {
    "bash": "command",
    "read_file": "path",
    "read_file_range": "path",
    "write_file": "path",
    "write_file_begin": "path",
    "write_file_chunk": "path",
    "write_file_commit": "path",
    "list_dir": "path",
    "file_metadata": "path",
    "file_search": "path",
    "project_build": "path",
    "note_save": "content",
}

_PREVIEW_MAX = 60

_UNKNOWN_TOOL_DEFAULT = PermissionDecision.ASK

OUTSIDE_CWD_HEURISTICS: list[str] = [
    r"(^|\s)/[^\s]",              # absolute path
    r"(^|\s)~",                   # home path
    r"(^|\s)\.\.(/|$|\s)",        # parent traversal
    r"\$\{?HOME\b",               # $HOME
    r"\$\{?PWD\b",                # $PWD
    r"(^|\s)[A-Za-z]:[\\/][^\s]*",  # Windows absolute path
]

_OUTSIDE_CWD_RE: list[re.Pattern[str]] = [
    re.compile(pattern) for pattern in OUTSIDE_CWD_HEURISTICS
]


def matches_outside_cwd(command: str) -> bool:
    return any(pattern.search(command) for pattern in _OUTSIDE_CWD_RE)


def evaluate(
    tool_name: str,
    params: dict[str, Any],
    policy: ToolPolicy | None = None,
) -> PermissionDecision:
    if policy is None:
        policy = DEFAULT_POLICIES.get(tool_name)

    if policy is None:
        return _UNKNOWN_TOOL_DEFAULT

    command = str(params.get("command", "")) if tool_name == "bash" else ""

    if command:
        for pattern in policy.deny_patterns:
            if re.search(pattern, command):
                return PermissionDecision.DENY

    if command and matches_outside_cwd(command):
        return PermissionDecision.ASK

    if command:
        for pattern in policy.allow_patterns:
            if re.search(pattern, command):
                return PermissionDecision.ALLOW

    return policy.default


def param_preview(tool_name: str, params: dict[str, Any]) -> str:
    key = _PREVIEW_KEY.get(tool_name)
    if key and key in params:
        value = str(params[key])
        if len(value) > _PREVIEW_MAX:
            value = value[:_PREVIEW_MAX] + "..."
        return f"{key}={value!r}"

    snippet = str(params)
    if len(snippet) > _PREVIEW_MAX:
        return snippet[:_PREVIEW_MAX] + "..."
    return snippet
