from __future__ import annotations

import asyncio
import json
from time import monotonic
from typing import Any

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.builtin.orchestrate_tasks import OrchestrateTasksTool

MAX_ROUNDS = 10
MAX_DISPATCH_PER_ROUND = 10
MAX_POLL_INTERVAL_SECONDS = 10.0
MAX_WAIT_SECONDS = 300.0


class OrchestrateUntilIdleTool(BaseTool):
    name = "orchestrate_until_idle"
    description = (
        "Run an explicit bounded orchestration loop by composing several "
        "orchestrate_tasks ticks. Defaults do not wait for background work."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional task IDs to collect or dispatch. Defaults to all relevant tasks."
                ),
            },
            "max_rounds": {
                "type": "integer",
                "description": "Maximum orchestration ticks to run. Defaults to 2.",
            },
            "max_dispatch_per_round": {
                "type": "integer",
                "description": (
                    "Maximum number of dispatchable tasks to start per tick. Defaults to 1."
                ),
            },
            "poll_interval_seconds": {
                "type": "number",
                "description": (
                    "Seconds to sleep between ticks when waiting is enabled. Defaults to 0.5."
                ),
            },
            "max_wait_seconds": {
                "type": "number",
                "description": (
                    "Total wait budget across the loop. Defaults to 0.0, which means no wait."
                ),
            },
        },
    }

    def __init__(self, tick_tool: OrchestrateTasksTool) -> None:
        self._tick_tool = tick_tool

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        task_ids = _task_ids(params)
        max_rounds = _bounded_int(params, "max_rounds", default=2, minimum=1, maximum=MAX_ROUNDS)
        max_dispatch_per_round = _bounded_int(
            params,
            "max_dispatch_per_round",
            default=1,
            minimum=0,
            maximum=MAX_DISPATCH_PER_ROUND,
        )
        poll_interval_seconds = _bounded_float(
            params,
            "poll_interval_seconds",
            default=0.5,
            minimum=0.0,
            maximum=MAX_POLL_INTERVAL_SECONDS,
        )
        max_wait_seconds = _bounded_float(
            params,
            "max_wait_seconds",
            default=0.0,
            minimum=0.0,
            maximum=MAX_WAIT_SECONDS,
        )

        started_at = monotonic()
        waited_seconds = 0.0
        rounds: list[dict[str, object]] = []
        stop_reason = "max_rounds"

        for round_number in range(1, max_rounds + 1):
            tick_result = await self._tick_tool.invoke(
                _tick_params(
                    task_ids=task_ids,
                    max_dispatch_per_round=max_dispatch_per_round,
                )
            )
            tick_payload = json.loads(tick_result.content)
            tick_payload["round"] = round_number
            rounds.append(tick_payload)

            tick_next_action = str(tick_payload.get("next_action", "idle"))
            if tick_next_action in {"idle", "replan", "human_review", "retry_or_review"}:
                stop_reason = tick_next_action
                break

            if round_number == max_rounds:
                stop_reason = "max_rounds"
                break

            if not _has_running_or_dispatched(tick_payload):
                stop_reason = tick_next_action
                break

            elapsed_seconds = monotonic() - started_at
            remaining_wait_seconds = max_wait_seconds - elapsed_seconds
            if remaining_wait_seconds <= 0:
                stop_reason = "running" if max_wait_seconds <= 0 else "max_wait_seconds"
                break

            sleep_seconds = min(poll_interval_seconds, remaining_wait_seconds)
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)
                waited_seconds += sleep_seconds

        payload = {
            "rounds": rounds,
            "rounds_run": len(rounds),
            "stop_reason": stop_reason,
            "next_action": _next_action(stop_reason),
            "waited_seconds": round(waited_seconds, 3),
            "bounds": {
                "max_rounds": max_rounds,
                "max_dispatch_per_round": max_dispatch_per_round,
                "poll_interval_seconds": poll_interval_seconds,
                "max_wait_seconds": max_wait_seconds,
            },
        }
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))


def _task_ids(params: dict[str, object]) -> list[int] | None:
    if "task_ids" not in params:
        return None
    raw_values: list[object] = list(params.get("task_ids") or [])  # type: ignore[call-overload]
    return [int(str(value)) for value in raw_values]


def _bounded_int(
    params: dict[str, object],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if key not in params:
        return default
    return min(maximum, max(minimum, int(str(params[key]))))


def _bounded_float(
    params: dict[str, object],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if key not in params:
        return default
    return min(maximum, max(minimum, float(str(params[key]))))


def _tick_params(
    *,
    task_ids: list[int] | None,
    max_dispatch_per_round: int,
) -> dict[str, object]:
    params: dict[str, object] = {"max_tasks": max_dispatch_per_round}
    if task_ids is not None:
        params["task_ids"] = list(task_ids)
    return params


def _has_running_or_dispatched(tick_payload: dict[str, object]) -> bool:
    collection = _dict_value(tick_payload, "collection")
    plan_summary = _dict_value(tick_payload, "plan_summary")
    dispatch = _dict_value(tick_payload, "dispatch")
    return bool(
        collection.get("running")
        or plan_summary.get("in_progress_task_ids")
        or dispatch.get("dispatched")
    )


def _dict_value(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _next_action(stop_reason: str) -> str:
    if stop_reason == "idle":
        return "summarize"
    if stop_reason in {"replan", "human_review", "retry_or_review"}:
        return stop_reason
    if stop_reason in {"running", "max_rounds", "max_wait_seconds"}:
        return "call_again_or_review"
    return "review"
