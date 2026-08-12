from __future__ import annotations

# Canonical local built-in tool capability names.
#
# These constants answer "can My Agent recognize/register this local tool name",
# not "will this exact run expose it". Some tools are conditional at runtime:
# note_save needs a session/store/run, while sub-agent tools need a run/provider.
CORE_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "read_file_range",
    "task_create",
    "task_update",
    "task_list",
    "task_get",
    "list_dir",
    "file_metadata",
    "file_search",
    "project_build",
    "write_file",
    "write_file_begin",
    "write_file_chunk",
    "write_file_commit",
    "bash",
    "delegation_policy",
    "schedule_plan",
    "dispatch_plan",
    "collect_dispatch_results",
    "orchestrate_tasks",
    "orchestrate_until_idle",
    "orchestration_summary",
)

SESSION_TOOL_NAMES: tuple[str, ...] = ("note_save",)

SUBAGENT_TOOL_NAMES: tuple[str, ...] = (
    "spawn_agent",
    "agent_result",
    "agent_cancel",
)

BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    *CORE_TOOL_NAMES,
    *SESSION_TOOL_NAMES,
    *SUBAGENT_TOOL_NAMES,
)
