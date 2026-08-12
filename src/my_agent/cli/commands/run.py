from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from my_agent.cli.commands._connection_errors import print_core_connection_error
from my_agent.core.config import Config
from my_agent.core.transport.socket_client import IpcError, SocketClient


def cmd_run(goal: str, config: Config) -> None:
    try:
        exit_code = asyncio.run(_run(goal, config))
    except (ConnectionRefusedError, OSError, TimeoutError):
        print_core_connection_error(config)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(exit_code)


class StdoutPrinter:
    def __init__(self) -> None:
        self._inline = False

    def _ensure_newline(self) -> None:
        if self._inline:
            print()
            self._inline = False

    async def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "llm.token":
            print(event.get("token", ""), end="", flush=True)
            self._inline = True
            return

        self._ensure_newline()
        line = _format_event(event)
        if line is not None:
            print(line)


async def _run(goal: str, config: Config) -> int:

    client = SocketClient(config.host, config.port)
    printer = StdoutPrinter()
    finished = asyncio.Event()
    active_run_id: str | None = None
    exit_code = 0

    async def handle_event(event: dict[str, Any]) -> None:
        nonlocal exit_code

        event_type = event.get("type")
        await printer.handle(event)

        if event_type == "run.finished":
            run_id = event.get("run_id")
            if active_run_id is None or run_id == active_run_id:
                if event.get("status") != "success":
                    exit_code = 1
                finished.set()

    await client.connect()

    client.on_event(handle_event)

    loop_task = asyncio.create_task(client.run_event_loop())

    try:
        result = await client.send_command(
            "agent.run",
            {
                "type": "agent.run",
                "goal": goal,
                "workspace_root": str(Path.cwd()),
            },
        )
        active_run_id = result["run_id"]
        print(f"run_id={active_run_id}")

        try:
            await client.send_command(
                "event.subscribe",
                {
                    "type": "event.subscribe",
                    "topics": [
                        "run.*",
                        "step.*",
                        "tool.*",
                        "llm.*",
                        "task.*",
                        "scheduler.*",
                    ],
                    "scope": f"run:{active_run_id}",
                    "replay_from_run": active_run_id,
                },
            )
        except IpcError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        await finished.wait()
        return exit_code

    finally:
        loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await loop_task
        await client.close()


def _format_event(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")

    if event_type == "run.started":
        return f"Run started: {event.get('goal', '')}"

    if event_type == "run.finished":
        status = event.get("status")
        steps = event.get("steps")
        reason = event.get("reason")
        if reason:
            return f"Run finished: {status}, steps={steps}, reason={reason}"
        return f"Run finished: {status}, steps={steps}"

    if event_type == "step.started":
        return f"Step {event.get('step')} started"

    if event_type == "step.finished":
        return f"Step {event.get('step')} finished"

    if event_type == "tool.call_started":
        return f"Tool started: {event.get('tool_name')}"

    if event_type == "tool.call_finished":
        return f"Tool finished: {event.get('tool_name')} in {event.get('elapsed_ms')}ms"

    if event_type == "tool.call_failed":
        return (
            f"Tool failed: {event.get('tool_name')} "
            f"{event.get('error_type')}: {event.get('error_message')}"
        )

    if event_type == "llm.model_selected":
        return f"Model selected: {event.get('model')}"

    if event_type == "llm.usage":
        return f"LLM usage: input={event.get('input_tokens')} output={event.get('output_tokens')}"

    if isinstance(event_type, str) and event_type.startswith("task."):
        task_id = event.get("task_id")
        if task_id:
            return f"Task event: {event_type} task_id={task_id}"
        return f"Task event: {event_type}"

    if isinstance(event_type, str) and event_type.startswith("scheduler."):
        plan_id = event.get("plan_id")
        if plan_id:
            return f"Scheduler event: {event_type} plan_id={plan_id}"
        return f"Scheduler event: {event_type}"

    return None
