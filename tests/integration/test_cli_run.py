from __future__ import annotations

import asyncio
import json
from typing import Any

from pytest import CaptureFixture

from my_agent.cli.commands import run as run_command
from my_agent.core.config import Config


async def test_cli_run_subscribes_to_started_run_and_exits_on_finished(
    free_port: int,
    capsys: CaptureFixture[str],
) -> None:
    requests: list[dict[str, Any]] = []
    run_id = "run-test-123"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while line := await reader.readline():
                request = json.loads(line)
                requests.append(request)

                if request["method"] == "agent.run":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"run_id": run_id},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "event.subscribe":
                    token_event = {
                        "kind": "event",
                        "event": {
                            "type": "llm.token",
                            "run_id": run_id,
                            "token": "Hello from the model.",
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    step_event = {
                        "kind": "event",
                        "event": {
                            "type": "step.started",
                            "run_id": run_id,
                            "step": 1,
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    task_event = {
                        "kind": "event",
                        "event": {
                            "type": "task.assigned",
                            "run_id": run_id,
                            "task_id": "task-1",
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    scheduler_event = {
                        "kind": "event",
                        "event": {
                            "type": "scheduler.plan.generated",
                            "run_id": run_id,
                            "plan_id": "plan-1",
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    finished_event = {
                        "kind": "event",
                        "event": {
                            "type": "run.finished",
                            "run_id": run_id,
                            "status": "completed",
                            "reason": None,
                            "steps": 1,
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"subscription_id": "sub-test", "replayed_count": 1},
                    }
                    writer.write((json.dumps(token_event) + "\n").encode())
                    writer.write((json.dumps(step_event) + "\n").encode())
                    writer.write((json.dumps(task_event) + "\n").encode())
                    writer.write((json.dumps(scheduler_event) + "\n").encode())
                    writer.write((json.dumps(finished_event) + "\n").encode())
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    config = Config(host="127.0.0.1", port=free_port)

    async with server:
        await asyncio.wait_for(run_command._run("hello", config), timeout=5.0)

    methods = [request["method"] for request in requests]
    assert methods == ["agent.run", "event.subscribe"]

    subscribe_params = requests[1]["params"]
    assert subscribe_params["scope"] == f"run:{run_id}"
    assert subscribe_params["replay_from_run"] == run_id
    assert subscribe_params["topics"] == [
        "run.*",
        "step.*",
        "tool.*",
        "llm.*",
        "task.*",
        "scheduler.*",
    ]

    output = capsys.readouterr().out
    assert f"run_id={run_id}" in output
    assert "Hello from the model.\nStep 1 started" in output
    assert "Step 1 started" in output
    assert "Task event: task.assigned task_id=task-1" in output
    assert "Scheduler event: scheduler.plan.generated plan_id=plan-1" in output
    assert "Run finished: completed, steps=1" in output


async def test_cli_run_returns_failure_when_run_finished_status_is_failed(
    free_port: int,
) -> None:
    run_id = "run-test-failed"

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while line := await reader.readline():
                request = json.loads(line)

                if request["method"] == "agent.run":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"run_id": run_id},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "event.subscribe":
                    event = {
                        "kind": "event",
                        "event": {
                            "type": "run.finished",
                            "run_id": run_id,
                            "status": "failed",
                            "reason": "llm_error",
                            "steps": 1,
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"subscription_id": "sub-test", "replayed_count": 1},
                    }
                    writer.write((json.dumps(event) + "\n").encode())
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    config = Config(host="127.0.0.1", port=free_port)

    async with server:
        exit_code = await asyncio.wait_for(run_command._run("hello", config), timeout=5.0)

    assert exit_code == 1
