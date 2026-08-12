from __future__ import annotations

from dataclasses import dataclass

from my_agent.core.task.classifier import TaskType


@dataclass(frozen=True)
class CompactionPolicy:
    name: str
    recent_message_keep: int
    recent_message_max_chars: int
    summary_focus: str


_POLICIES: dict[TaskType, CompactionPolicy] = {
    TaskType.CHAT: CompactionPolicy(
        name="chat_v1",
        recent_message_keep=8,
        recent_message_max_chars=3_000,
        summary_focus="Preserve user preferences, explanations, decisions, and open questions.",
    ),
    TaskType.CODE_READ: CompactionPolicy(
        name="code_read_v1",
        recent_message_keep=4,
        recent_message_max_chars=4_000,
        summary_focus="Preserve file paths, symbols, module relationships, and call flow.",
    ),
    TaskType.CODE_WORK: CompactionPolicy(
        name="code_work_v1",
        recent_message_keep=6,
        recent_message_max_chars=6_000,
        summary_focus=(
            "Preserve modified files, commands, test results, errors, constraints, and TODOs."
        ),
    ),
}


def policy_for_task(task_type: TaskType) -> CompactionPolicy:
    return _POLICIES[task_type]
