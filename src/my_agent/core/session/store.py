from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_agent.core.session.model import Session

logger = logging.getLogger(__name__)

MessageContent = str | list[dict[str, Any]]
ALLOWED_MESSAGE_ROLES = {"user", "assistant", "tool"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    def __init__(self, root: Path, *, isolate_by_workspace: bool = False) -> None:
        self._root = root.expanduser()
        self._isolate_by_workspace = isolate_by_workspace
        self._session_dirs: dict[str, Path] = {}
        self._root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, sid: str) -> Path:
        if self._isolate_by_workspace:
            return self._lookup_session_dir(sid)
        return self._root / sid

    def runs_dir(self, sid: str) -> Path:
        return self.session_dir(sid) / "runs"

    def write_run_meta(
        self,
        sid: str,
        run_id: str,
        data: dict[str, Any],
    ) -> None:
        path = self.runs_dir(sid) / run_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def append_compaction_record(self, sid: str, record: dict[str, Any]) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "compactions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_meta(self, session: Session) -> None:
        path = self._session_dir_for(session)
        self._session_dirs[session.id] = path
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if self._isolate_by_workspace:
            self._write_index_entry(session.id, path)

    def read_meta(self, sid: str) -> Session:
        path = self.session_dir(sid) / "meta.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(data)

    def append_message(
        self,
        sid: str,
        role: str,
        content: MessageContent,
        run_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "ts": _now(),
            "role": role,
            "content": content,
        }
        if extra:
            row.update(extra)

        if run_id is not None:
            row["run_id"] = run_id

        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)

        with (path / "thread.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_messages(
        self,
        sid: str,
        messages: list[dict[str, Any]],
        run_id: str,
    ) -> None:
        for msg in messages:
            row = dict(msg)
            role = str(row.pop("role"))
            content = row.pop("content", "")
            self.append_message(
                sid,
                role=role,
                content=content,
                run_id=run_id,
                extra=row,
            )

    def read_messages(self, sid: str) -> list[dict[str, Any]]:
        path = self.session_dir(sid) / "thread.jsonl"

        if not path.exists():
            return []

        messages: list[dict[str, Any]] = []

        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skip broken thread row sid=%s line=%s", sid, line_no)
                continue
            role = row.get("role")
            if role not in ALLOWED_MESSAGE_ROLES:
                logger.warning(
                    "skip unknown thread role sid=%s line=%s role=%s",
                    sid,
                    line_no,
                    role,
                )
                continue
            message = dict(row)
            message.pop("run_id", None)
            message.pop("ts", None)
            messages.append(message)

        return self._trim_orphan_tool_messages(messages)

    def _trim_orphan_tool_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending: set[str] = set()
        last_balanced = 0

        for idx, message in enumerate(messages, start=1):
            role = message.get("role")

            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id")
                            if isinstance(tool_call_id, str) and tool_call_id:
                                pending.add(tool_call_id)

            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not isinstance(tool_call_id, str) or tool_call_id not in pending:
                    logger.warning("trim orphan tool message from thread")
                    return messages[:last_balanced]
                pending.remove(tool_call_id)

            if not pending:
                last_balanced = idx

        if pending:
            logger.warning("trim unanswered tool calls from thread")
            return messages[:last_balanced]

        return messages

    def read_notes(self, sid: str) -> str:
        path = self.session_dir(sid) / "notes.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def append_note(self, sid: str, content: str, run_id: str) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)

        with (path / "notes.md").open("a", encoding="utf-8") as f:
            f.write(f"## {run_id}\n\n{content}\n\n")


    def archive_thread(self, sid: str, archive_id: str) -> Path | None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)

        thread_path = path / "thread.jsonl"
        if not thread_path.exists():
            return None

        archive_dir = path / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        archive_path = archive_dir / f"thread-before-{archive_id}.jsonl"
        archive_path.write_text(thread_path.read_text(encoding="utf-8"), encoding="utf-8")
        return archive_path

    def rewrite_thread_as_summary(
        self,
        sid: str,
        summary_text: str,
        archive_id: str,
        *,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        thread_path = path / "thread.jsonl"
        compacted_messages = [
            {
                "ts": _now(),
                "role": "user",
                "content": f"Compacted conversation summary:\n\n{summary_text}",
                "run_id": archive_id,
                "compacted": True,
            },
            {
                "ts": _now(),
                "role": "assistant",
                "content": "Understood. I will continue from this compacted summary.",
                "run_id": archive_id,
                "compacted": True,
            },
        ]
        for msg in recent_messages or []:
            compacted_messages.append(
                {
                    "ts": _now(),
                    "role": msg["role"],
                    "content": msg["content"],
                    "run_id": archive_id,
                    "compacted_recent": True,
                }
            )

        thread_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compacted_messages),
            encoding="utf-8",
        )

    def compact_active_thread(
        self,
        sid: str,
        summary_text: str,
        archive_id: str,
        *,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.archive_thread(sid, archive_id)
        self.rewrite_thread_as_summary(
            sid,
            summary_text,
            archive_id,
            recent_messages=recent_messages,
        )

    def _session_dir_for(self, session: Session) -> Path:
        if not self._isolate_by_workspace:
            return self._root / session.id
        if session.workspace_root:
            key = _workspace_key(session.workspace_root)
            return self._root / "workspaces" / key / "sessions" / session.id
        return self._root / "sessions" / session.id

    def _lookup_session_dir(self, sid: str) -> Path:
        cached = self._session_dirs.get(sid)
        if cached is not None:
            return cached

        index = self._read_index()
        rel = index.get(sid)
        if rel is not None:
            path = (self._root / rel).resolve()
            self._session_dirs[sid] = path
            return path

        for legacy in (self._root / sid, self._root / "sessions" / sid):
            if (legacy / "meta.json").exists():
                self._session_dirs[sid] = legacy
                return legacy

        return self._root / "sessions" / sid

    def _read_index(self) -> dict[str, str]:
        path = self._root / "session_index.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("session index is unreadable; falling back to path lookup")
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _write_index_entry(self, sid: str, path: Path) -> None:
        index = self._read_index()
        index[sid] = path.relative_to(self._root).as_posix()
        (self._root / "session_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _workspace_key(workspace_root: str) -> str:
    digest = hashlib.sha256(workspace_root.encode("utf-8")).hexdigest()[:12]
    return f"ws-{digest}"
