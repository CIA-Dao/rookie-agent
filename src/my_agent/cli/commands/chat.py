from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from my_agent.cli.commands._connection_errors import print_core_connection_error
from my_agent.core.config import Config
from my_agent.core.transport.socket_client import IpcError, SocketClient

_DECISION_MAP: dict[str, str] = {
    "y": "allow_once",
    "a": "always_allow",
    "n": "deny_once",
    "d": "always_deny",
}


class ChatPrinter:
    def __init__(self) -> None:
        self._inline = False
        self.pending_permission_id: str | None = None

    def _ensure_newline(self) -> None:
        if self._inline:
            self._inline = False

    async def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "llm.token":
            print(event.get("token", ""), end="", flush=True)
            self._inline = True
            return

        if event_type == "tool.call_started":
            self._ensure_newline()
            print(f"[tool] {event.get('tool_name', '')}")
            return

        if event_type == "session.waiting_for_input":
            self._ensure_newline()
            print("[waiting for input]")
            return

        if event_type == "session.closed":
            self._ensure_newline()
            print("session closed.")

        if event_type == "permission.requested":
            self._ensure_newline()
            tool_name = str(event.get("tool_name", ""))
            tool_use_id = str(event.get("tool_use_id", ""))
            param_preview = str(event.get("param_preview", ""))
            print(f"[permission] {tool_name} {param_preview}")
            print("  y=allow once  a=always allow  n=deny once  d=always deny")
            self.pending_permission_id = tool_use_id
            return

        if event_type == "skill.invoked":
            self._ensure_newline()
            print(f"[skill] {event.get('skill_name', '')} {event.get('arguments', '')}".rstrip())
            return

        if isinstance(event_type, str) and event_type.startswith("task."):
            self._ensure_newline()
            print(f"[task] {event_type}")
            return

        if isinstance(event_type, str) and event_type.startswith("scheduler."):
            self._ensure_newline()
            print(f"[scheduler] {event_type}")
            return


async def _readline(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)


# 客户端本地处理的斜杠命令（不发往 daemon），返回 True 表示已本地处理
def _try_handle_local_slash_command(content: str, workspace_root: Path) -> bool:
    if content == "/init":
        from my_agent.core.memory.init import cmd_init as _core_init

        result = _core_init(workspace_root)
        for msg in result.messages:
            print(msg)
        return True
    return False


def cmd_chat(config: Config) -> None:
    try:
        exit_code = asyncio.run(_chat(config))
    except (ConnectionRefusedError, OSError, TimeoutError):
        print_core_connection_error(config)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(exit_code)


async def _chat(config: Config) -> int:
    client = SocketClient(config.host, config.port)
    printer = ChatPrinter()

    await client.connect()

    client.on_event(printer.handle)
    loop_task = asyncio.create_task(client.run_event_loop())

    try:
        await client.send_command(
            "event.subscribe",
            {
                "type": "event.subscribe",
                "topics": [
                    "session.*",
                    "run.*",
                    "tool.*",
                    "llm.token",
                    "permission.*",
                    "skill.*",
                    "task.*",
                    "scheduler.*",
                ],
                "scope": "global",
            },
        )

        created = await client.send_command(
            "session.create",
            {
                "type": "session.create",
                "mode": "chat",
                "title": "",
                "workspace_root": str(Path.cwd()),
            },
        )
        session_id = str(created["session_id"])
        print(f"[session: {session_id}]")

        while True:
            try:
                line = await _readline("> ")
            except (EOFError, KeyboardInterrupt):
                break

            content = line.strip()

            if printer.pending_permission_id:
                decision = _DECISION_MAP.get(content.lower())
                if decision is None:
                    print(
                        "  enter y (allow once), a (always allow), "
                        "n (deny once), d (always deny)"
                    )
                    continue
                tool_use_id = printer.pending_permission_id
                printer.pending_permission_id = None
                await client.send_command(
                    "permission.respond",
                    {
                        "type": "permission.respond",
                        "tool_use_id": tool_use_id,
                        "decision": decision,
                    },
                )
                continue

            if content in ("/exit", "/quit"):
                break

            if _try_handle_local_slash_command(content, Path.cwd()):
                continue

            if content.startswith("/compact"):
                focus = content.removeprefix("/compact").strip()
                result = await client.send_command(
                    "session.compact",
                    {
                        "type": "session.compact",
                        "session_id": session_id,
                        "focus": focus,
                    },
                )
                summary_tokens = result.get("summary_tokens", 0)
                saved_tokens = result.get("saved_tokens", 0)
                print(
                    f"[compact] summary={summary_tokens} tokens, "
                    f"saved~={saved_tokens} tokens"
                )
                continue

            if not content:
                continue

            await client.send_command(
                "session.send_message",
                {
                    "type": "session.send_message",
                    "session_id": session_id,
                    "content": content,
                },
            )

        await client.send_command(
            "session.close",
            {
                "type": "session.close",
                "session_id": session_id,
            },
        )
        return 0
    except IpcError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await loop_task
        await client.close()
