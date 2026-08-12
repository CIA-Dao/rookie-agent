from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from my_agent.core.bus.envelope import HandlerError
from my_agent.core.bus.events import (
    ContextCompactionFailedEvent,
    SkillInvokedEvent,
    SkillToolCompatibilityEvent,
)
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse, UsageStats
from my_agent.core.runner import RunOutcome
from my_agent.core.session.manager import (
    COMPACTION_FAILED,
    COMPACTION_PROVIDER_UNAVAILABLE,
    SESSION_CLOSED,
    SESSION_NOT_FOUND,
    SessionManager,
)
from my_agent.core.session.model import Session
from my_agent.core.session.store import SessionStore


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                str,
                str,
                Session,
                SessionStore,
                str | None,
                list[str] | None,
            ]
        ] = []

    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str,
        session: Session,
        store: SessionStore,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        self.calls.append(
            (goal, run_id, session, store, system_prompt_override, tool_whitelist)
        )
        store.append_message(session.id, "assistant", f"answer: {goal}", run_id=run_id)
        return RunOutcome(status="success", result=f"answer: {goal}", reason=None)


class FakeRunnerWithToolNames(FakeRunner):
    def __init__(self, tool_names: list[str]) -> None:
        super().__init__()
        self._tool_names = tool_names

    def available_tool_names(self) -> list[str]:
        return list(self._tool_names)


class FakeCompactProvider:
    def __init__(self, text: str = "summary text") -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self.calls.append(
            {
                "messages": messages,
                "tool_schemas": tool_schemas,
                "run_id": run_id,
                "step": step,
                "system": system,
            }
        )
        return LlmResponse(
            stop_reason="end_turn",
            text=self.text,
            usage=UsageStats(input_tokens=100, output_tokens=10),
        )


@pytest.fixture
def manager_parts(tmp_path: Path) -> tuple[SessionManager, FakeRunner, SessionStore]:
    store = SessionStore(tmp_path)
    runner = FakeRunner()
    manager = SessionManager(store, lambda: runner, EventBus())
    return manager, runner, store


async def test_create_writes_session_meta(
    manager_parts: tuple[SessionManager, FakeRunner, SessionStore],
) -> None:
    manager, _runner, store = manager_parts

    session = await manager.create("chat", title="learning", workspace_root="D:/project")

    assert session.id.startswith("sess-")
    assert session.mode == "chat"
    assert session.status == "active"
    assert session.title == "learning"
    assert session.workspace_root == "D:/project"
    assert store.read_meta(session.id) == session


async def test_send_message_appends_thread_and_waits_for_input(
    manager_parts: tuple[SessionManager, FakeRunner, SessionStore],
) -> None:
    manager, runner, store = manager_parts
    session = await manager.create("chat")

    run_id = await manager.send_message(session.id, "hello", run_id="run-test")

    assert run_id == "run-test"
    assert runner.calls[0][0] == "hello"
    assert store.read_messages(session.id) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer: hello"},
    ]

    saved = store.read_meta(session.id)
    assert saved.status == "waiting_for_input"
    assert saved.title == "hello"
    assert saved.run_ids == ["run-test"]


async def test_send_message_expands_slash_skill(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    manager = SessionManager(store, lambda: runner, EventBus())
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".my-agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "review.md").write_text(
        """---
name: review
description: Local review
allowed_tools:
  - read_file
  - bash
---

Review target:
$ARGUMENTS
""",
        encoding="utf-8",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/review src/foo.py", run_id="run-skill")

    assert runner.calls[0][0] == "Review target:\nsrc/foo.py"
    assert runner.calls[0][4] == "Review target:\n$ARGUMENTS"
    assert runner.calls[0][5] == ["read_file", "bash"]
    assert store.read_messages(session.id)[0] == {
        "role": "user",
        "content": "/review src/foo.py",
    }


async def test_send_message_publishes_skill_invoked_event(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".my-agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "review.md").write_text(
        """---
name: review
description: Local review
---

Review:
$ARGUMENTS
""",
        encoding="utf-8",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/review src/foo.py", run_id="run-skill")

    skill_events = [event for event in events if isinstance(event, SkillInvokedEvent)]
    assert len(skill_events) == 1
    assert skill_events[0].skill_name == "review"
    assert skill_events[0].arguments == "src/foo.py"
    assert skill_events[0].run_id == "run-skill"
    assert skill_events[0].session_id == session.id


async def test_send_message_double_slash_escapes_skill(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    manager = SessionManager(store, lambda: runner, EventBus())
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".my-agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "review.md").write_text(
        "---\nname: review\ndescription: Local review\n---\nSkill prompt",
        encoding="utf-8",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "//review src/foo.py", run_id="run-escape")

    assert runner.calls[0][0] == "/review src/foo.py"


async def test_send_message_publishes_resumed_when_waiting_for_input(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    runner = FakeRunner()
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    session = await manager.create("chat")

    await manager.send_message(session.id, "first", run_id="run-1")
    await manager.send_message(session.id, "second", run_id="run-2")

    event_types = [str(getattr(event, "type")) for event in events]
    assert event_types.count("session.resumed") == 1
    assert event_types.index("session.resumed") < event_types.index("session.message_received", 4)


async def test_one_shot_session_closes_after_message(
    manager_parts: tuple[SessionManager, FakeRunner, SessionStore],
) -> None:
    manager, _runner, store = manager_parts
    session = await manager.create("one_shot")

    await manager.send_message(session.id, "hello", run_id="run-test")

    assert store.read_meta(session.id).status == "closed"


async def test_send_message_rejects_unknown_session(
    manager_parts: tuple[SessionManager, FakeRunner, SessionStore],
) -> None:
    manager, _runner, _store = manager_parts

    with pytest.raises(HandlerError) as exc:
        await manager.send_message("missing", "hello")

    assert exc.value.code == SESSION_NOT_FOUND


async def test_closed_session_rejects_new_message(
    manager_parts: tuple[SessionManager, FakeRunner, SessionStore],
) -> None:
    manager, _runner, _store = manager_parts
    session = await manager.create("chat")
    await manager.close(session.id)

    with pytest.raises(HandlerError) as exc:
        await manager.send_message(session.id, "hello")

    assert exc.value.code == SESSION_CLOSED


async def test_compact_rewrites_thread_and_returns_saved_tokens(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    runner = FakeRunner()
    provider = FakeCompactProvider()
    manager = SessionManager(store, lambda: runner, EventBus(), provider=provider)
    session = await manager.create("chat")
    store.append_message(session.id, "user", "old message " * 100)
    store.append_message(session.id, "assistant", "old answer " * 100)

    result = await manager.compact(session.id, focus="keep decisions")

    assert result.summary_tokens == 10
    assert result.saved_tokens > 0
    assert provider.calls[0]["run_id"] == "compact"
    request = provider.calls[0]["messages"][0]["content"]
    assert isinstance(request, str)
    assert "Preserve user preferences" in request
    assert "keep decisions" in request
    assert store.read_messages(session.id) == [
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
        {
            "role": "user",
            "content": "old message " * 100,
            "compacted_recent": True,
        },
        {
            "role": "assistant",
            "content": "old answer " * 100,
            "compacted_recent": True,
        },
    ]
    archives = list((tmp_path / session.id / "archive").glob("thread-before-compact-*.jsonl"))
    assert len(archives) == 1
    compactions = [
        line
        for line in (tmp_path / session.id / "compactions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(compactions) == 1
    assert '"policy": "chat_v1"' in compactions[0]


async def test_compact_requires_provider(
    manager_parts: tuple[SessionManager, FakeRunner, SessionStore],
) -> None:
    manager, _runner, _store = manager_parts
    session = await manager.create("chat")

    with pytest.raises(HandlerError) as exc:
        await manager.compact(session.id)

    assert exc.value.code == COMPACTION_PROVIDER_UNAVAILABLE


async def test_compact_publishes_failed_event_when_provider_returns_empty(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    runner = FakeRunner()
    provider = FakeCompactProvider(text="   ")
    bus = EventBus()
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus, provider=provider)
    session = await manager.create("chat")
    store.append_message(session.id, "user", "old message")

    with pytest.raises(HandlerError) as exc:
        await manager.compact(session.id)

    assert exc.value.code == COMPACTION_FAILED
    failed_events = [event for event in events if isinstance(event, ContextCompactionFailedEvent)]
    assert len(failed_events) == 1
    assert failed_events[0].session_id == session.id
    assert failed_events[0].run_id.startswith("compact-")


# ── P6: slash skill tool compatibility resolution ────────────────────────────


def _write_skill(workspace: Path, name: str, body: str) -> None:
    skill_dir = workspace / ".my-agent" / "skills"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / f"{name}.md").write_text(body, encoding="utf-8")


async def test_send_message_resolves_alias_tools_for_slash_skill(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    manager = SessionManager(store, lambda: runner, EventBus())
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "deploy",
        """---
name: deploy
description: alias-driven skill
allowed_tools:
  - shell
  - file.read
---

Deploy $ARGUMENTS
""",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/deploy web", run_id="run-alias")

    # shell -> bash, file.read -> read_file, in declaration order
    assert runner.calls[0][5] == ["bash", "read_file"]


async def test_send_message_emits_tool_compatibility_event_on_partial_mismatch(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "review",
        """---
name: review
description: partial mismatch
allowed_tools:
  - shell
  - file.read
  - unknown_tool
---

Review $ARGUMENTS
""",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/review src/x.py", run_id="run-partial")

    compat_events = [
        event for event in events if isinstance(event, SkillToolCompatibilityEvent)
    ]
    assert len(compat_events) == 1
    evt = compat_events[0]
    assert evt.skill_name == "review"
    assert evt.run_id == "run-partial"
    assert evt.session_id == session.id
    assert evt.resolved_tools == ["bash", "read_file"]
    assert evt.unresolved_tools == ["unknown_tool"]
    alias_pairs = {(a["from"], a["to"]) for a in evt.aliases}
    assert alias_pairs == {("shell", "bash"), ("file.read", "read_file")}
    # Runner still got the resolved whitelist.
    assert runner.calls[0][5] == ["bash", "read_file"]


async def test_send_message_emits_tool_compatibility_event_on_full_mismatch(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "exotic",
        """---
name: exotic
description: only unknown tools
allowed_tools:
  - grep
  - browser.open
---

Exotic $ARGUMENTS
""",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/exotic now", run_id="run-empty")

    compat_events = [
        event for event in events if isinstance(event, SkillToolCompatibilityEvent)
    ]
    assert len(compat_events) == 1
    evt = compat_events[0]
    assert evt.resolved_tools == []
    assert sorted(evt.unresolved_tools) == ["browser.open", "grep"]
    # Critical: runner is given an empty whitelist (zero tools), NOT None (all tools).
    assert runner.calls[0][5] == []


async def test_send_message_does_not_emit_compatibility_event_when_no_diagnostics(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "review",
        """---
name: review
description: clean skill
allowed_tools:
  - read_file
  - bash
---

Review $ARGUMENTS
""",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/review x", run_id="run-clean")

    compat_events = [
        event for event in events if isinstance(event, SkillToolCompatibilityEvent)
    ]
    assert compat_events == []
    assert runner.calls[0][5] == ["read_file", "bash"]


async def test_send_message_resolves_runtime_mcp_tool_name_from_runner(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunnerWithToolNames(["bash", "docs__search"])
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "docs",
        """---
name: docs
description: runtime MCP skill
allowed_tools:
  - docs__search
---

Search $ARGUMENTS
""",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/docs api", run_id="run-mcp")

    assert runner.calls[0][5] == ["docs__search"]
    compat_events = [
        event for event in events if isinstance(event, SkillToolCompatibilityEvent)
    ]
    assert compat_events == []


async def test_send_message_skill_without_allowed_tools_keeps_unrestricted(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "review",
        """---
name: review
description: no whitelist
---

Review $ARGUMENTS
""",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/review x", run_id="run-none")

    # No whitelist (None) — runner is unrestricted by the skill.
    assert runner.calls[0][5] is None
    compat_events = [
        event for event in events if isinstance(event, SkillToolCompatibilityEvent)
    ]
    assert compat_events == []


async def test_send_message_skill_with_empty_allowed_tools_keeps_unrestricted(
    tmp_path: Path,
) -> None:
    # P6 section 6.3: "Empty allowed_tools still means no Skill-specific
    # restriction, preserving existing behavior for skills without a whitelist."
    # The None-vs-[] distinction lives at the resolver API (see
    # test_skill_tool_compat.py); SessionManager preserves existing parsing.
    store = SessionStore(tmp_path / "store")
    runner = FakeRunner()
    bus = EventBus()
    events: list[BaseModel] = []

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    manager = SessionManager(store, lambda: runner, bus)
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "empty",
        """---
name: empty
description: declared but empty
allowed_tools: []
---

Empty $ARGUMENTS
""",
    )
    session = await manager.create("chat", workspace_root=str(workspace))

    await manager.send_message(session.id, "/empty x", run_id="run-empty-list")

    # Empty declared list → SessionManager treats it as unrestricted (None),
    # matching existing behavior prior to P6.
    assert runner.calls[0][5] is None
    compat_events = [
        event for event in events if isinstance(event, SkillToolCompatibilityEvent)
    ]
    assert compat_events == []
