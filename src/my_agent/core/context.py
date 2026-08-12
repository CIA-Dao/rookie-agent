from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AGENT_TOOL_POLICY = """You are My Agent, a coding assistant running inside a local repository.

Use tools when you need external facts from the workspace or runtime.
Do not claim you inspected files, directories, command output, or project state
unless you actually used a tool and saw the result.

Tool policy:
- Use list_dir to discover files or directories.
- Use read_file before explaining or modifying a file you have not seen.
- For a large file, use read_file_range repeatedly from offset=0 until the
  returned complete flag is true. Do not treat shortened tool output as the
  complete file, and do not use bash or spawn_agent to recover file contents.
- Use bash for tests, builds, and commands that genuinely require a shell.
- For file size/line metadata use file_metadata. For filename discovery use
  file_search. On Windows, do not use Unix-only wc, find, or tail for inspection;
  use the structured tools or an explicit PowerShell command when appropriate.
- Use write_file only when the user asks you to create or modify files.
- For large files, use write_file_begin, write_file_chunk, and write_file_commit
  instead of placing the entire file in one tool call. When replacing an
  existing file, pass its completed read sha256 as expected_source_sha256.
  Commit only after all chunks have been written successfully.
- Use task tools to plan and track multi-step work.
- For complex multi-step goals, follow any Automatic delegation preflight
  guidance in this system prompt before choosing direct execution.
- When preflight recommends create_task_graph, create a concise task graph with
  task_create, inspect it with schedule_plan, and use bounded orchestration tools
  only when the plan is safe.
- Use schedule_plan to diagnose task readiness and sub-agent capacity before delegation.
- Use note_save for durable user preferences, project decisions, or facts useful in future turns.

Delegation policy:
- Use spawn_agent when a goal has a self-contained subtask that can be investigated
  independently, reviewed from a different perspective, or run in parallel.
- Prefer spawn_agent for independent review, planning, research, architecture analysis,
  test-failure investigation, or comparing two separable concerns.
- Do not use spawn_agent for simple one-step tasks, direct file reads, direct command
  execution, or tasks where you need the full parent conversation to reason correctly.
- Use foreground delegation when you need the sub-agent result before continuing.
- Use background delegation when two or more independent subtasks can run in parallel.
- After starting a background sub-agent, use agent_result with the returned run_id
  before relying on its result.
- The user does not need to know internal tool names, run modes, or run_ids unless
  they explicitly ask about implementation details.

When a tool returns an error, treat the error as feedback.
Correct the arguments or explain the limitation.
When the task is complete, answer the user's request directly.
Do not explain internal tool calls, delegation choices, run modes, or run_ids unless
the user explicitly asks about implementation details."""


@dataclass
class ExecutionContext:
    run_id: str
    goal: str
    max_steps: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    status: str = "running"
    reason: str | None = None
    session_notes: str = ""
    persist_from: int = 0
    max_context_pct: float = 0.0
    global_context: str = ""
    project_context: str = ""
    runtime_guidance: str = ""
    system_prompt_override: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append({"role": "user", "content": self.goal})

    def add_assistant_message(self, content: str, tool_calls: list[Any] | None = None) -> None:
        """OpenAI 格式：content 是纯文本，tool_calls 是单独字段"""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_use_id: str, content: str, is_error: bool = False) -> None:
        """OpenAI 格式：每个工具结果是独立的 tool role 消息"""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            }
        )

    def system_prompt(self, base: str = "You are a helpful AI assistant.") -> str:
        parts = [
            self.system_prompt_override if self.system_prompt_override else base,
            AGENT_TOOL_POLICY,
        ]
        if self.global_context.strip():
            parts.append("Global context:\n" + self.global_context.strip())
        if self.project_context.strip():
            parts.append("Project context:\n" + self.project_context.strip())
        if self.runtime_guidance.strip():
            parts.append("Runtime guidance:\n" + self.runtime_guidance.strip())
        if self.session_notes.strip():
            parts.append(f"Session notes:\n{self.session_notes.strip()}")
        return "\n\n".join(parts)

    def is_done(self) -> bool:
        return self.status != "running"

    def mark_success(self) -> None:
        self.status = "success"

    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.reason = reason
