from __future__ import annotations

from my_agent.core.permissions.policy import (
    PermissionDecision,
    ToolPolicy,
    evaluate,
    matches_outside_cwd,
    param_preview,
)


def test_safe_tools_default_allow() -> None:
    assert evaluate("read_file", {"path": "README.md"}) == PermissionDecision.ALLOW
    assert evaluate("list_dir", {"path": "."}) == PermissionDecision.ALLOW
    assert evaluate("note_save", {"content": "remember this"}) == PermissionDecision.ALLOW


def test_risky_tools_default_ask() -> None:
    assert evaluate("bash", {"command": "echo hi"}) == PermissionDecision.ASK
    assert evaluate("write_file", {"path": "out.txt", "content": "hi"}) == PermissionDecision.ASK


def test_unknown_tool_default_ask() -> None:
    assert evaluate("future_tool", {}) == PermissionDecision.ASK


def test_relative_bash_command_not_outside_cwd() -> None:
    assert not matches_outside_cwd("echo hello")
    assert not matches_outside_cwd("ls src/")
    assert not matches_outside_cwd("cat README.md")
    assert not matches_outside_cwd("python -m pytest")


def test_absolute_path_forces_ask() -> None:
    assert matches_outside_cwd("cat /etc/hosts")
    assert evaluate("bash", {"command": "cat /etc/hosts"}) == PermissionDecision.ASK


def test_parent_traversal_forces_ask() -> None:
    assert matches_outside_cwd("ls ../")
    assert evaluate("bash", {"command": "ls ../"}) == PermissionDecision.ASK


def test_cd_forces_ask() -> None:
    assert matches_outside_cwd("cd /tmp && ls")
    assert evaluate("bash", {"command": "cd /tmp && ls"}) == PermissionDecision.ASK


def test_safe_relative_cd_does_not_force_permission_prompt() -> None:
    assert not matches_outside_cwd("cd . && npm run build")
    assert not matches_outside_cwd("cd src && python -m pytest")


def test_windows_absolute_cd_still_forces_permission_prompt() -> None:
    assert matches_outside_cwd(r"cd /d D:\other-project && npm run build")


def test_deny_pattern_wins() -> None:
    policy = ToolPolicy(
        default=PermissionDecision.ASK,
        deny_patterns=[r"rm\s+-rf"],
    )

    assert evaluate("bash", {"command": "rm -rf temp"}, policy) == PermissionDecision.DENY


def test_allow_pattern_grants_access() -> None:
    policy = ToolPolicy(
        default=PermissionDecision.ASK,
        allow_patterns=[r"^echo\b"],
    )

    assert evaluate("bash", {"command": "echo hello"}, policy) == PermissionDecision.ALLOW


def test_outside_cwd_forces_ask_even_when_allow_pattern_matches() -> None:
    policy = ToolPolicy(
        default=PermissionDecision.ASK,
        allow_patterns=[r".*"],
    )

    assert evaluate("bash", {"command": "cat /etc/hosts"}, policy) == PermissionDecision.ASK


def test_param_preview_known_tools() -> None:
    assert param_preview("bash", {"command": "echo hi"}) == "command='echo hi'"
    assert param_preview("read_file", {"path": "README.md"}) == "path='README.md'"
    assert param_preview("write_file", {"path": "out.txt", "content": "hello"}) == "path='out.txt'"
    assert param_preview("list_dir", {"path": "src"}) == "path='src'"
    assert param_preview("note_save", {"content": "remember this"}) == "content='remember this'"


def test_param_preview_truncates_long_value() -> None:
    preview = param_preview("bash", {"command": "x" * 80})

    assert preview == f"command='{'x' * 60}...'"


def test_param_preview_unknown_tool_falls_back_to_params_snippet() -> None:
    assert param_preview("future_tool", {"value": "hello"}) == "{'value': 'hello'}"
