from __future__ import annotations

import json
from pathlib import Path

from my_agent.core.session.model import Session
from my_agent.core.session.store import SessionStore


def test_store_creates_root(tmp_path: Path) -> None:
    root = tmp_path / "sessions"

    SessionStore(root)

    assert root.exists()


def test_meta_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = Session(
        id="sess-1",
        mode="chat",
        status="waiting_for_input",
        title="hello",
        created_at="t1",
        updated_at="t2",
        run_ids=["run-1"],
    )

    store.write_meta(session)
    loaded = store.read_meta("sess-1")

    assert loaded == session


def test_workspace_isolated_sessions_use_separate_directories(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, isolate_by_workspace=True)
    session_a = Session(
        id="sess-a",
        mode="chat",
        status="active",
        title="a",
        created_at="t1",
        updated_at="t1",
        workspace_root="D:/projects/a",
    )
    session_b = Session(
        id="sess-b",
        mode="chat",
        status="active",
        title="b",
        created_at="t1",
        updated_at="t1",
        workspace_root="D:/projects/b",
    )

    store.write_meta(session_a)
    store.write_meta(session_b)

    assert store.session_dir("sess-a") != store.session_dir("sess-b")
    assert "workspaces" in store.session_dir("sess-a").parts
    assert store.read_meta("sess-a") == session_a
    assert store.read_meta("sess-b") == session_b


def test_workspace_isolated_store_uses_index_after_restart(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, isolate_by_workspace=True)
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="hello",
        created_at="t1",
        updated_at="t1",
        workspace_root="D:/projects/demo",
    )
    store.write_meta(session)
    original_dir = store.session_dir(session.id)

    restarted = SessionStore(tmp_path, isolate_by_workspace=True)

    assert restarted.session_dir(session.id) == original_dir
    assert restarted.read_meta(session.id) == session


def test_write_run_meta(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.write_run_meta(
        "sess-1",
        "run-1",
        {"run_id": "run-1", "task_type": "code_work"},
    )

    data = json.loads((tmp_path / "sess-1" / "runs" / "run-1" / "meta.json").read_text())
    assert data == {"run_id": "run-1", "task_type": "code_work"}


def test_append_compaction_record(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.append_compaction_record(
        "sess-1",
        {"compact_id": "compact-1", "policy": "code_work_v1"},
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "sess-1" / "compactions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows == [{"compact_id": "compact-1", "policy": "code_work_v1"}]


def test_thread_message_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.append_message("sess-1", "user", "hello")
    store.append_message("sess-1", "assistant", "hi", run_id="run-1")

    messages = store.read_messages("sess-1")

    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_append_messages_preserves_openai_tool_fields(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.append_messages(
        "sess-1",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "file content",
            },
        ],
        run_id="run-1",
    )

    assert store.read_messages("sess-1") == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "file content",
            "tool_call_id": "call-1",
        },
    ]


def test_append_message_writes_timestamp_but_read_messages_hides_it(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.append_message("sess-1", "user", "hello")

    thread_path = tmp_path / "sess-1" / "thread.jsonl"
    row = json.loads(thread_path.read_text(encoding="utf-8"))

    assert row["ts"]
    assert store.read_messages("sess-1") == [{"role": "user", "content": "hello"}]


def test_read_messages_skips_broken_rows_and_unknown_roles(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-1"
    session_dir.mkdir()
    (session_dir / "thread.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "hello"}),
                "{broken-json",
                json.dumps({"role": "debug", "content": "internal"}),
                json.dumps({"role": "tool", "content": "ok", "tool_call_id": "call-1"}),
            ]
        ),
        encoding="utf-8",
    )
    store = SessionStore(tmp_path)

    assert store.read_messages("sess-1") == [{"role": "user", "content": "hello"}]


def test_read_messages_trims_unanswered_tool_call_and_following_rows(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-1"
    session_dir.mkdir()
    (session_dir / "thread.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "hello"}),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    }
                ),
                json.dumps({"role": "user", "content": "next turn"}),
            ]
        ),
        encoding="utf-8",
    )
    store = SessionStore(tmp_path)

    assert store.read_messages("sess-1") == [{"role": "user", "content": "hello"}]


def test_read_messages_keeps_balanced_tool_call_pair(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.append_messages(
        "sess-1",
        [
            {"role": "user", "content": "read file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "file content"},
            {"role": "assistant", "content": "done"},
        ],
        run_id="run-1",
    )

    assert store.read_messages("sess-1") == [
        {"role": "user", "content": "read file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "file content", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "done"},
    ]


def test_notes_read_and_append(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    assert store.read_notes("sess-1") == ""

    store.append_note("sess-1", "Python 3.12", "run-1")
    notes = store.read_notes("sess-1")

    assert "Python 3.12" in notes
    assert "run-1" in notes


def test_archive_thread_copies_active_thread(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "hello", run_id="run-old")
    store.append_message("sess-1", "assistant", "hi", run_id="run-old")

    archive_path = store.archive_thread("sess-1", "run-new")

    assert archive_path == (
        tmp_path / "sess-1" / "archive" / "thread-before-run-new.jsonl"
    )
    archive_rows = [
        json.loads(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["content"] for row in archive_rows] == ["hello", "hi"]


def test_archive_thread_returns_none_when_no_active_thread(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    assert store.archive_thread("sess-1", "run-new") is None


def test_rewrite_thread_as_summary_replaces_active_context(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "hello", run_id="run-old")

    store.rewrite_thread_as_summary("sess-1", "summary text", "run-new")

    assert store.read_messages("sess-1") == [
        {
            "role": "user",
            "content": "Compacted conversation summary:\n\nsummary text",
            "compacted": True,
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue from this compacted summary.",
            "compacted": True,
        },
    ]


def test_rewrite_thread_as_summary_can_preserve_recent_messages(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.rewrite_thread_as_summary(
        "sess-1",
        "summary text",
        "run-new",
        recent_messages=[
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ],
    )

    assert store.read_messages("sess-1") == [
        {
            "role": "user",
            "content": "Compacted conversation summary:\n\nsummary text",
            "compacted": True,
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue from this compacted summary.",
            "compacted": True,
        },
        {"role": "user", "content": "recent question", "compacted_recent": True},
        {"role": "assistant", "content": "recent answer", "compacted_recent": True},
    ]


def test_compact_active_thread_archives_full_thread_and_rewrites_active_context(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "hello", run_id="run-old")
    store.append_message("sess-1", "assistant", "hi", run_id="run-old")

    store.compact_active_thread("sess-1", "summary text", "run-new")

    archive_path = (
        tmp_path / "sess-1" / "archive" / "thread-before-run-new.jsonl"
    )
    assert archive_path.exists()
    assert store.read_messages("sess-1")[0]["content"] == (
        "Compacted conversation summary:\n\nsummary text"
    )
