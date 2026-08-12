from __future__ import annotations

import asyncio
import json
from typing import Any

from pytest import CaptureFixture, MonkeyPatch

from my_agent.cli.commands import chat as chat_command
from my_agent.core.config import Config


async def test_chat_printer_renders_task_and_scheduler_events(
    capsys: CaptureFixture[str],
) -> None:
    printer = chat_command.ChatPrinter()

    await printer.handle({"type": "task.assigned", "task_id": "task-1"})
    await printer.handle({"type": "scheduler.plan.generated", "plan_id": "plan-1"})
    await printer.handle({"type": "engine.internal_noise"})

    output = capsys.readouterr().out
    assert "[task] task.assigned" in output
    assert "[scheduler] scheduler.plan.generated" in output
    assert "engine.internal_noise" not in output


async def test_cli_chat_creates_session_sends_message_and_closes(
    free_port: int,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    requests: list[dict[str, Any]] = []
    inputs = iter(["hello", "/exit"])

    async def fake_readline(prompt: str) -> str:
        return next(inputs)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while line := await reader.readline():
                request = json.loads(line)
                requests.append(request)

                if request["method"] == "event.subscribe":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"subscription_id": "sub-test", "replayed_count": 0},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.create":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "session_id": "sess-test",
                            "status": "active",
                            "title": "",
                        },
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.send_message":
                    token_event = {
                        "kind": "event",
                        "event": {
                            "type": "llm.token",
                            "run_id": "run-test",
                            "token": "hi",
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "session_id": "sess-test",
                            "run_id": "run-test",
                        },
                    }
                    writer.write((json.dumps(token_event) + "\n").encode())
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.close":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"session_id": "sess-test", "status": "closed"},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    monkeypatch.setattr(chat_command, "_readline", fake_readline)

    async with server:
        exit_code = await asyncio.wait_for(
            chat_command._chat(Config(port=free_port)),
            timeout=5.0,
        )

    assert exit_code == 0
    assert [request["method"] for request in requests] == [
        "event.subscribe",
        "session.create",
        "session.send_message",
        "session.close",
    ]
    assert requests[2]["params"]["session_id"] == "sess-test"
    assert requests[2]["params"]["content"] == "hello"

    output = capsys.readouterr().out
    assert "[session: sess-test]" in output
    assert "hi" in output


async def test_cli_chat_responds_to_permission_request(
    free_port: int,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    requests: list[dict[str, Any]] = []
    inputs = iter(["y", "/exit"])

    async def fake_readline(prompt: str) -> str:
        return next(inputs)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while line := await reader.readline():
                request = json.loads(line)
                requests.append(request)

                if request["method"] == "event.subscribe":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"subscription_id": "sub-test", "replayed_count": 0},
                    }
                    permission_event = {
                        "kind": "event",
                        "event": {
                            "type": "permission.requested",
                            "run_id": "run-test",
                            "tool_use_id": "tc-1",
                            "tool_name": "bash",
                            "params": {"command": "echo hi"},
                            "session_id": "sess-test",
                            "ts": "2026-01-01T00:00:00Z",
                        },
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    writer.write((json.dumps(permission_event) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.create":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "session_id": "sess-test",
                            "status": "active",
                            "title": "",
                        },
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "permission.respond":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"ok": True},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.close":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"session_id": "sess-test", "status": "closed"},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    monkeypatch.setattr(chat_command, "_readline", fake_readline)

    async with server:
        exit_code = await asyncio.wait_for(
            chat_command._chat(Config(port=free_port)),
            timeout=5.0,
        )

    assert exit_code == 0
    assert [request["method"] for request in requests] == [
        "event.subscribe",
        "session.create",
        "permission.respond",
        "session.close",
    ]
    assert requests[0]["params"]["topics"] == [
        "session.*",
        "run.*",
        "tool.*",
        "llm.token",
        "permission.*",
        "skill.*",
        "task.*",
        "scheduler.*",
    ]
    assert requests[2]["params"] == {
        "type": "permission.respond",
        "tool_use_id": "tc-1",
        "decision": "allow_once",
    }

    output = capsys.readouterr().out
    assert "[permission] bash" in output


async def test_cli_chat_sends_session_compact_command(
    free_port: int,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    requests: list[dict[str, Any]] = []
    inputs = iter(["/compact keep facts", "/exit"])

    async def fake_readline(prompt: str) -> str:
        return next(inputs)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while line := await reader.readline():
                request = json.loads(line)
                requests.append(request)

                if request["method"] == "event.subscribe":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"subscription_id": "sub-test", "replayed_count": 0},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.create":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "session_id": "sess-test",
                            "status": "active",
                            "title": "",
                        },
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.compact":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"summary_tokens": 12, "saved_tokens": 88},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                if request["method"] == "session.close":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"session_id": "sess-test", "status": "closed"},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", free_port)
    monkeypatch.setattr(chat_command, "_readline", fake_readline)

    async with server:
        exit_code = await asyncio.wait_for(
            chat_command._chat(Config(port=free_port)),
            timeout=5.0,
        )

    assert exit_code == 0
    assert [request["method"] for request in requests] == [
        "event.subscribe",
        "session.create",
        "session.compact",
        "session.close",
    ]
    assert requests[2]["params"] == {
        "type": "session.compact",
        "session_id": "sess-test",
        "focus": "keep facts",
    }

    output = capsys.readouterr().out
    assert "[compact] summary=12 tokens, saved~=88 tokens" in output
