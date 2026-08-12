from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_agent.core.bus.events import ContextCompactedEvent, ContextCompactionFailedEvent
from my_agent.core.compact.policy import CompactionPolicy, policy_for_task
from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.base import LLMProvider
from my_agent.core.session.store import SessionStore
from my_agent.core.task.classifier import TaskType

logger = logging.getLogger(__name__)

_COMPACT_PROMPT = """\
You are compressing an agent conversation into a handoff summary.
Another LLM instance will continue this task from your summary alone. Make it complete.

Structure your response with exactly these six sections:

## 1. Original Goal
One sentence describing what the user asked the agent to accomplish.

## 2. Completed Steps
Bullet list of what has been done. Be specific: file paths, commands run, decisions made.

## 3. Key Constraints & Discoveries
Facts learned during the run that affect future decisions.

## 4. Current File State
For each file that was created or modified: path, and a one-line description.

## 5. Remaining TODOs
Ordered list of what still needs to be done.

## 6. Critical Data
Any values the next LLM needs verbatim: IDs, exact errors, config values, paths.

Be concise. Omit reasoning steps and intermediate attempts. Keep conclusions.
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _summary_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


@dataclass
class CompactionResult:
    summary_text: str
    original_token_estimate: int
    summary_tokens: int


class Compactor:
    def __init__(
        self,
        bus: EventBus,
        session_dir: Path,
        session_id: str,
        *,
        store: SessionStore | None = None,
        task_type: TaskType = TaskType.CHAT,
    ) -> None:
        self._bus = bus
        self._session_dir = session_dir
        self._session_id = session_id
        self._store = store
        self._task_type = task_type
        self._policy = policy_for_task(task_type)

    async def compact(
        self,
        context: ExecutionContext,
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        recent_messages = _recent_plain_messages(context.messages, self._policy)
        result = await self.compact_messages(
            context.messages,
            provider,
            focus=_join_focus(self._policy.summary_focus, focus),
        )

        if result is None:
            await self._bus.publish(
                ContextCompactionFailedEvent(
                    session_id=self._session_id,
                    run_id=context.run_id,
                    reason="summary_unavailable",
                    ts=_now(),
                )
            )
            return None
        context.messages = [
            {
                "role": "user",
                "content": f"Compacted conversation summary:\n\n{result.summary_text}",
            },
            {
                "role": "assistant",
                "content": "Understood. I will continue from this compacted summary.",
            },
        ] + recent_messages

        context.persist_from = len(context.messages)

        self._write_summary(result.summary_text)

        if self._store is not None:
            self._store.compact_active_thread(
                self._session_id,
                result.summary_text,
                context.run_id,
                recent_messages=recent_messages,
            )
            self._store.append_compaction_record(
                self._session_id,
                {
                    "compact_id": context.run_id,
                    "run_id": context.run_id,
                    "task_type": self._task_type.value,
                    "policy": self._policy.name,
                    "original_tokens": result.original_token_estimate,
                    "summary_tokens": result.summary_tokens,
                    "recent_message_keep": self._policy.recent_message_keep,
                    "preserved_recent_count": len(recent_messages),
                    "created_at": _now(),
                },
            )

        await self._bus.publish(
            ContextCompactedEvent(
                session_id=self._session_id,
                run_id=context.run_id,
                original_tokens=result.original_token_estimate,
                summary_tokens=result.summary_tokens,
                ts=_now(),
            )
        )

        return result

    async def compact_messages(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        original_estimate = (
            sum(len(str(message.get("content", ""))) for message in messages) // 4
        )

        history_text = _messages_to_text(messages)
        prompt = _COMPACT_PROMPT
        if focus.strip():
            prompt += f"\n\nIMPORTANT: Pay special attention to: {focus.strip()}"

        compress_request: list[dict[str, object]] = [
            {"role": "user", "content": f"{prompt}\n\n---\n\n{history_text}"}
        ]

        try:
            response = await provider.chat(
                messages=compress_request,
                tool_schemas=[],
                bus=EventBus(),
                run_id="compact",
                step=0,
                system="You are a helpful assistant that summarizes agent conversations.",
            )
        except Exception:
            logger.exception("compactor: LLM call failed")
            return None

        summary_text = response.text.strip()
        if not summary_text:
            logger.warning("compactor: LLM returned empty summary")
            return None

        summary_tokens = response.usage.output_tokens if response.usage else len(summary_text) // 4

        return CompactionResult(
            summary_text=summary_text,
            original_token_estimate=original_estimate,
            summary_tokens=summary_tokens,
        )

    def _write_summary(self, summary_text: str) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        path = self._session_dir / f"summary_{_summary_timestamp()}.md"
        path.write_text(summary_text + "\n", encoding="utf-8")


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")

        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
        else:
            parts.append(f"[{role}]\n{content!r}")

    return "\n\n".join(parts)


def _recent_plain_messages(
    messages: list[dict[str, Any]],
    policy: CompactionPolicy,
) -> list[dict[str, Any]]:
    recent: list[dict[str, Any]] = []

    for message in reversed(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            continue
        if "tool_calls" in message:
            continue
        if not isinstance(content, str) or not content:
            continue

        recent.append(
            {
                "role": role,
                "content": _trim_recent_content(content, policy.recent_message_max_chars),
            }
        )
        if len(recent) >= policy.recent_message_keep:
            break

    return list(reversed(recent))


def _trim_recent_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    keep = max_chars // 2
    omitted = len(content) - (keep * 2)
    return content[:keep] + f"\n...[{omitted} chars omitted]...\n" + content[-keep:]


def _join_focus(policy_focus: str, user_focus: str) -> str:
    if user_focus.strip():
        return f"{policy_focus}\n{user_focus.strip()}"
    return policy_focus
