from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from my_agent.core.bus.events import StepFinishedEvent, StepStartedEvent
from my_agent.core.compact.budget import truncate_tool_results
from my_agent.core.compact.compactor import Compactor
from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.base import LLMProvider
from my_agent.core.permissions.manager import PermissionManager
from my_agent.core.tools.base import ToolResult
from my_agent.core.tools.invocation import invoke_tool
from my_agent.core.tools.registry import ToolRegistry

_MAX_CONSECUTIVE_TOOL_ERRORS = 3


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: EventBus,
        *,
        permission_manager: PermissionManager | None = None,
        session_id: str = "",
        compactor: Compactor | None = None,
        compact_token_threshold: int = 120_000,
        tool_result_limit: int = 8_000,
        tool_result_keep: int = 4_000,
        compact_context_ratio: float = 0.0,
        workspace_root: str | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        self._permission_manager = permission_manager
        self._session_id = session_id
        self._compactor = compactor
        self._compact_token_threshold = compact_token_threshold
        self._tool_result_limit = tool_result_limit
        self._tool_result_keep = tool_result_keep
        self._compact_context_ratio = compact_context_ratio
        self._workspace_root = workspace_root

    async def run(self, context: ExecutionContext) -> None:
        tool_error_streaks: dict[str, int] = {}
        while not context.is_done():
            context.step += 1

            await self._bus.publish(
                StepStartedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )

            # [plan] 调用 LLM
            try:
                context.messages = truncate_tool_results(
                    context.messages,
                    limit=self._tool_result_limit,
                    keep=self._tool_result_keep,
                )
                tool_schemas = self._registry.tool_schemas()
                system = context.system_prompt()
                prompt_tokens = self._provider.count_tokens(
                    messages=context.messages,
                    tool_schemas=tool_schemas,
                    system=system,
                )

                if (
                    self._compactor is not None
                    and prompt_tokens >= self._compact_token_threshold
                ):
                    compacted = await self._compactor.compact(context, self._provider)
                    if compacted is None:
                        context.mark_failed("context_compaction_failed")
                        break
                    tool_schemas = self._registry.tool_schemas()
                    system = context.system_prompt()
                    prompt_tokens = self._provider.count_tokens(
                        messages=context.messages,
                        tool_schemas=tool_schemas,
                        system=system,
                    )

                response = await self._provider.chat(
                    messages=context.messages,
                    tool_schemas=tool_schemas,
                    bus=self._bus,
                    run_id=context.run_id,
                    step=context.step,
                    system=system,
                )
            except asyncio.CancelledError:
                context.mark_failed("cancelled")
                raise
            except Exception as e:
                context.mark_failed(f"llm_error: {e}")
                break

            response_context_pct = (
                response.usage.context_pct
                if response.usage is not None
                else prompt_tokens / self._provider.context_window()
            )
            if response_context_pct > 0:
                context.max_context_pct = max(
                    context.max_context_pct,
                    response_context_pct,
                )

            # [observe] LLM 的回复：文本 + 工具调用（OpenAI 格式）
            openai_tool_calls = []
            for tc in response.tool_calls:
                openai_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input, ensure_ascii=False),
                        },
                    }
                )

            context.add_assistant_message(
                content=response.text,
                tool_calls=openai_tool_calls if openai_tool_calls else None,
            )

            # [act] 调用工具
            has_tool_calls = bool(response.tool_calls)
            if has_tool_calls:
                for tc in response.tool_calls:
                    protocol_error = response.tool_call_errors.get(tc.id)
                    if protocol_error is not None:
                        result = ToolResult(
                            content=(
                                f"{protocol_error}; no tool was executed. "
                                "Retry with a complete JSON object containing "
                                "all required tool fields."
                            ),
                            is_error=True,
                            error_type="protocol_error",
                        )
                    else:
                        result = await invoke_tool(
                            self._registry,
                            tc,
                            self._bus,
                            context.run_id,
                            permission_manager=self._permission_manager,
                            session_id=self._session_id,
                            workspace_root=self._workspace_root,
                            step=context.step,
                        )
                    context.add_tool_result(tc.id, result.content, is_error=result.is_error)
                    if result.is_error:
                        streak = tool_error_streaks.get(tc.name, 0) + 1
                        tool_error_streaks[tc.name] = streak
                        if streak >= _MAX_CONSECUTIVE_TOOL_ERRORS:
                            context.mark_failed(
                                f"repeated_tool_error: {tc.name} failed "
                                f"{streak} consecutive times ({result.content})"
                            )
                            break
                    else:
                        tool_error_streaks.pop(tc.name, None)
            if (
                not context.is_done()
                and has_tool_calls
                and self._compactor is not None
                and self._compact_context_ratio > 0
                and response_context_pct >= self._compact_context_ratio
            ):
                compacted = await self._compactor.compact(context, self._provider)
                if compacted is None:
                    context.mark_failed("context_compaction_failed")
                    break
                context.max_context_pct = 0.0

            # 判断是否结束
            if not has_tool_calls and response.stop_reason == "end_turn":
                context.mark_success()
            elif context.step >= context.max_steps:
                context.mark_failed("max_steps_exceeded")

            await self._bus.publish(
                StepFinishedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )


def _now() -> str:
    return datetime.now(UTC).isoformat()
