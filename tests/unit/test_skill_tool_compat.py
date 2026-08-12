from __future__ import annotations

from my_agent.core.skills.tool_compat import (
    BUILTIN_TOOL_NAMES,
    TOOL_ALIASES,
    SkillToolCompatibility,
    available_tool_names_for_runner,
    resolve_allowed_tools,
)
from my_agent.core.tools.catalog import BUILTIN_TOOL_NAMES as CATALOG_BUILTIN_TOOL_NAMES


def test_builtin_tool_names_come_from_canonical_catalog() -> None:
    assert BUILTIN_TOOL_NAMES is CATALOG_BUILTIN_TOOL_NAMES
    assert "schedule_plan" in BUILTIN_TOOL_NAMES
    assert "delegation_policy" in BUILTIN_TOOL_NAMES
    assert "spawn_agent" in BUILTIN_TOOL_NAMES
    assert "agent_result" in BUILTIN_TOOL_NAMES
    assert "agent_cancel" in BUILTIN_TOOL_NAMES
    assert "orchestrate_tasks" in BUILTIN_TOOL_NAMES
    assert "orchestrate_until_idle" in BUILTIN_TOOL_NAMES
    assert "orchestration_summary" in BUILTIN_TOOL_NAMES


def test_available_tool_names_for_runner_uses_runtime_method() -> None:
    class RunnerWithRuntimeTools:
        def available_tool_names(self) -> list[object]:
            return ["read_file", "mcp.search", 123]

    assert available_tool_names_for_runner(RunnerWithRuntimeTools()) == [
        "read_file",
        "mcp.search",
        "123",
    ]


def test_available_tool_names_for_runner_falls_back_to_builtin_catalog() -> None:
    assert available_tool_names_for_runner(object()) == list(BUILTIN_TOOL_NAMES)

# ── 6.3 / 8.10: empty allowed_tools preserves existing unrestricted behavior ─

def test_resolve_returns_unrestricted_flag_when_declared_is_none() -> None:
    available = ["read_file", "write_file", "list_dir", "bash"]
    result = resolve_allowed_tools(None, available)

    assert result.declared_tools is None
    assert result.resolved_tools == []
    assert result.aliases == []
    assert result.unresolved_tools == []
    assert result.has_diagnostics is False
    # The SessionManager treats None as "do not apply a skill whitelist".
    assert result.unrestricted_by_skill is True


# ── 8.11: exact allowed_tools names still filter registry as before ──────────

def test_resolve_keeps_exact_local_tool_names() -> None:
    available = ["read_file", "write_file", "list_dir", "bash"]
    result = resolve_allowed_tools(["read_file", "bash"], available)

    assert result.resolved_tools == ["read_file", "bash"]
    assert result.unresolved_tools == []
    assert result.aliases == []
    assert result.has_diagnostics is False
    assert result.unrestricted_by_skill is False


# ── 8.12 / 8.13: alias and capability mappings ───────────────────────────────

def test_resolve_maps_shell_alias_to_bash() -> None:
    available = ["bash", "read_file", "write_file", "list_dir"]
    result = resolve_allowed_tools(["shell"], available)

    assert result.resolved_tools == ["bash"]
    assert result.aliases == [{"from": "shell", "to": "bash"}]
    assert result.has_diagnostics is True


def test_resolve_maps_file_read_alias_to_read_file() -> None:
    available = ["bash", "read_file", "write_file", "list_dir"]
    result = resolve_allowed_tools(["file.read"], available)

    assert result.resolved_tools == ["read_file"]
    assert result.aliases == [{"from": "file.read", "to": "read_file"}]


def test_resolve_maps_capability_aliases_table_driven() -> None:
    # P6 spec alias table — explicit, conservative, no fuzzy guesses.
    expected = {
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
    for src, dst in expected.items():
        assert TOOL_ALIASES.get(src) == dst, (
            f"alias {src!r} must map to {dst!r}, got {TOOL_ALIASES.get(src)!r}"
        )


# ── 8.14: duplicates deduplicate in declaration order ────────────────────────

def test_resolve_deduplicates_in_declaration_order() -> None:
    available = ["bash", "read_file", "write_file", "list_dir"]
    # `shell` and `bash` both resolve to `bash`; explicit `bash` second must not
    # produce a duplicate. `read_file` appears twice as exact.
    result = resolve_allowed_tools(
        ["shell", "bash", "read_file", "read_file"], available
    )

    assert result.resolved_tools == ["bash", "read_file"]


# ── 8.15: partial mismatch keeps resolved and reports unresolved ─────────────

def test_resolve_partial_mismatch_reports_unresolved() -> None:
    available = ["bash", "read_file", "write_file", "list_dir"]
    result = resolve_allowed_tools(
        ["shell", "file.read", "unknown_tool"], available
    )

    assert result.resolved_tools == ["bash", "read_file"]
    assert result.unresolved_tools == ["unknown_tool"]
    assert {a["from"] for a in result.aliases} == {"shell", "file.read"}
    assert result.has_diagnostics is True


# ── 8.16: full mismatch results in empty whitelist, not all tools ────────────

def test_resolve_full_mismatch_returns_empty_whitelist() -> None:
    available = ["bash", "read_file", "write_file", "list_dir"]
    result = resolve_allowed_tools(["grep", "browser.open"], available)

    assert result.resolved_tools == []
    assert result.unresolved_tools == ["grep", "browser.open"]
    assert result.aliases == []
    assert result.has_diagnostics is True
    # Critical: not unrestricted. Empty list = zero tools, not all tools.
    assert result.unrestricted_by_skill is False


# ── 8 / section 6.3: grep / browser.open must remain unresolved ──────────────

def test_resolve_does_not_special_case_grep_or_browser_open() -> None:
    available = ["bash", "read_file", "write_file", "list_dir"]
    result = resolve_allowed_tools(["grep", "browser.open", "search", "fetch"], available)

    assert result.resolved_tools == []
    assert sorted(result.unresolved_tools) == ["browser.open", "fetch", "grep", "search"]


# ── None vs [] semantic distinction ──────────────────────────────────────────

def test_resolve_distinguishes_none_from_empty_list() -> None:
    available = ["bash", "read_file"]

    none_result = resolve_allowed_tools(None, available)
    empty_result = resolve_allowed_tools([], available)

    # None: skill did not declare a whitelist → unrestricted
    assert none_result.unrestricted_by_skill is True
    assert none_result.has_diagnostics is False

    # []: skill declared a whitelist, but no tools matched → zero tools
    assert empty_result.unrestricted_by_skill is False
    assert empty_result.resolved_tools == []
    # Empty declared list does not produce noisy diagnostics.
    assert empty_result.has_diagnostics is False


# ── Exact name wins over alias ────────────────────────────────────────────────

def test_resolve_exact_name_wins_over_alias_when_both_available() -> None:
    # `bash` is both an exact local name AND an alias target of `shell`.
    # Declaring both `shell` and `bash` must yield a single `bash`, not two.
    available = ["bash"]
    result = resolve_allowed_tools(["shell", "bash"], available)

    assert result.resolved_tools == ["bash"]


# ── Declared-but-not-registered exact name is unresolved ─────────────────────

def test_resolve_reports_local_name_not_in_registry_as_unresolved() -> None:
    available = ["bash"]
    result = resolve_allowed_tools(["bash", "read_file"], available)

    # `bash` is in the registry; `read_file` is not. Exact name that is missing
    # from the registry counts as unresolved so the user sees a diagnostic.
    assert result.resolved_tools == ["bash"]
    assert result.unresolved_tools == ["read_file"]


# ── Type shape smoke test ────────────────────────────────────────────────────

def test_resolve_returns_typed_compatibility_object() -> None:
    available = ["bash", "read_file"]
    result = resolve_allowed_tools(["shell", "grep"], available)

    assert isinstance(result, SkillToolCompatibility)
    assert result.declared_tools == ["shell", "grep"]
    assert result.available_tools == available
