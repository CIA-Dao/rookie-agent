from __future__ import annotations

import asyncio
from pathlib import Path

from my_agent.core.config import CompactConfig, Config
from my_agent.core.events.bus import EventBus
from my_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from my_agent.core.runner import AgentRunner, _workspace_manifest, _write_artifact_manifest
from my_agent.core.session.model import Session
from my_agent.core.session.store import SessionStore


def test_artifact_manifest_is_workspace_relative(tmp_path: Path) -> None:
    before_root = tmp_path / "workspace"
    before_root.mkdir()
    (before_root / "old.js").write_text("old", encoding="utf-8")
    before = _workspace_manifest(before_root)
    (before_root / "old.js").write_text("new", encoding="utf-8")
    (before_root / "new.js").write_text("created", encoding="utf-8")
    after = _workspace_manifest(before_root)

    run_path = tmp_path / "run"
    run_path.mkdir()
    _write_artifact_manifest(run_path, before, after)

    manifest = (run_path / "artifact-manifest.json").read_text(encoding="utf-8")
    assert '"new.js"' in manifest
    assert '"old.js"' in manifest
    assert '"outside_workspace": []' in manifest


class _FinalAnswerProvider:
    def __init__(self) -> None:
        self.seen_messages: list[dict[str, object]] | None = None
        self.seen_system: str | None = None

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return 0

    def context_window(self) -> int:
        return 64_000

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
        self.seen_messages = list(messages)
        self.seen_system = system
        return LlmResponse(
            stop_reason="end_turn",
            text="done",
            tool_calls=[],
            usage=None,
        )


class _CompactThenAnswerProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages) // 4

    def context_window(self) -> int:
        return 64_000

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
                "messages": list(messages),
                "tool_schemas": tool_schemas,
                "run_id": run_id,
                "step": step,
                "system": system,
            }
        )
        if run_id == "compact":
            return LlmResponse(
                stop_reason="end_turn",
                text="summary after compact",
                tool_calls=[],
                usage=None,
            )
        return LlmResponse(
            stop_reason="end_turn",
            text="answer after compact",
            tool_calls=[],
            usage=None,
        )


class _SpawnAgentProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return 0

    def context_window(self) -> int:
        return 64_000

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
                "run_id": run_id,
                "messages": list(messages),
                "tool_schemas": tool_schemas,
            }
        )

        if run_id == "parent-run" and step == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="spawn-1",
                        name="spawn_agent",
                        input={
                            "description": "analyze child task",
                            "prompt": "Analyze this sub task.",
                        },
                    )
                ],
            )

        if run_id != "parent-run":
            return LlmResponse(stop_reason="end_turn", text="child analysis")

        return LlmResponse(
            stop_reason="end_turn",
            text="parent used child analysis",
        )


class _BackgroundSpawnProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.tool_results: list[str] = []
        self.child_run_id = ""

    def count_tokens(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None = None,
    ) -> int:
        return 0

    def context_window(self) -> int:
        return 64_000

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
                "run_id": run_id,
                "step": step,
                "messages": list(messages),
                "tool_schemas": tool_schemas,
            }
        )

        if run_id != "parent-run":
            return LlmResponse(stop_reason="end_turn", text="background child analysis")

        tool_messages = [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "tool"
        ]
        self.tool_results = tool_messages

        if step == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="spawn-bg-1",
                        name="spawn_agent",
                        input={
                            "description": "analyze in background",
                            "prompt": "Analyze this background sub task.",
                            "run_in_background": True,
                        },
                    )
                ],
            )

        if not self.child_run_id:
            started = tool_messages[-1]
            self.child_run_id = started.rsplit("=", maxsplit=1)[1]

        if step == 2 or (
            tool_messages and "still running" in tool_messages[-1]
        ):
            await asyncio.sleep(0)
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id=f"result-{step}",
                        name="agent_result",
                        input={"run_id": self.child_run_id},
                    )
                ],
            )

        return LlmResponse(
            stop_reason="end_turn",
            text="parent received background child analysis",
        )


async def test_runner_run_and_capture_returns_outcome(tmp_path: Path) -> None:
    runner = AgentRunner(Config(), provider=_FinalAnswerProvider(), runs_dir=tmp_path)

    outcome = await runner.run_and_capture("say hi", run_id="run-test")

    assert outcome.status == "success"
    assert outcome.result == "done"
    assert outcome.reason is None
    assert (tmp_path / "run-test" / "events.jsonl").exists()


async def test_runner_reports_failed_delivery_instead_of_model_completion(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "App.vue").write_text(
        "<script setup>\nimport TankGame from './components/TankGame.vue'\n</script>\n",
        encoding="utf-8",
    )
    session = Session(
        id="sess-delivery",
        mode="chat",
        status="active",
        title="delivery",
        created_at="t1",
        updated_at="t1",
        workspace_root=str(workspace),
    )
    provider = _FinalAnswerProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path / "runs")

    outcome = await runner.run_and_capture(
        "创建一个 Vue 项目",
        run_id="run-delivery",
        session=session,
    )

    assert outcome.status == "failed"
    assert "未确认交付完成" in outcome.result
    assert "delivery_verification_failed" in (outcome.reason or "")
    assert outcome.delivery is not None
    assert outcome.delivery["final_status"] == "not_accepted"
    delivery_file = (
        tmp_path / "runs" / "run-delivery" / "delivery-result.json"
    ).read_text(encoding="utf-8")
    assert '"final_status": "not_accepted"' in delivery_file


async def test_runner_injects_delegation_preflight_for_complex_goal(
    tmp_path: Path,
) -> None:
    provider = _FinalAnswerProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path)
    goal = (
        "First inspect the architecture, then split the work, and after that "
        "implement the scheduler changes."
    )

    await runner.run_and_capture(goal, run_id="run-complex")

    assert provider.seen_system is not None
    assert "Automatic delegation preflight:" in provider.seen_system
    assert '"decision": "create_task_graph"' in provider.seen_system
    assert "task_create" in provider.seen_system
    assert provider.seen_messages == [{"role": "user", "content": goal}]


async def test_runner_skips_delegation_preflight_for_simple_goal(tmp_path: Path) -> None:
    provider = _FinalAnswerProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path)

    await runner.run_and_capture("fix typo", run_id="run-simple")

    assert provider.seen_system is not None
    assert "Automatic delegation preflight:" not in provider.seen_system


async def test_runner_skips_delegation_preflight_when_tool_is_not_allowed(
    tmp_path: Path,
) -> None:
    provider = _FinalAnswerProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path)

    await runner.run_and_capture(
        (
            "First inspect the architecture, then split the work, and after that "
            "implement the scheduler changes."
        ),
        run_id="run-whitelist",
        tool_whitelist=["read_file"],
    )

    assert provider.seen_system is not None
    assert "Automatic delegation preflight:" not in provider.seen_system


async def test_runner_can_use_spawn_agent_tool_foreground(tmp_path: Path) -> None:
    provider = _SpawnAgentProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path)

    outcome = await runner.run_and_capture("delegate this", run_id="parent-run")

    assert outcome.status == "success"
    assert outcome.result == "parent used child analysis"

    run_ids = [str(call["run_id"]) for call in provider.calls]
    assert run_ids[0] == "parent-run"
    assert any(run_id != "parent-run" for run_id in run_ids)
    assert run_ids[-1] == "parent-run"

    final_parent_messages = provider.calls[-1]["messages"]
    assert isinstance(final_parent_messages, list)
    assert final_parent_messages[-1]["role"] == "tool"
    assert final_parent_messages[-1]["content"] == "child analysis"


async def test_runner_can_use_spawn_agent_tool_background(tmp_path: Path) -> None:
    provider = _BackgroundSpawnProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path)

    outcome = await runner.run_and_capture("delegate in background", run_id="parent-run")

    assert outcome.status == "success"
    assert outcome.result == "parent received background child analysis"
    assert provider.child_run_id

    assert provider.tool_results[0].startswith(
        "Subagent started in background: run_id="
    )
    assert provider.tool_results[-1] == "background child analysis"

    run_ids = [str(call["run_id"]) for call in provider.calls]
    assert "parent-run" in run_ids
    assert provider.child_run_id in run_ids


async def test_runner_records_code_work_task_type(tmp_path: Path) -> None:
    runner = AgentRunner(Config(), provider=_FinalAnswerProvider(), runs_dir=tmp_path)

    await runner.run_and_capture("修复 pytest 失败", run_id="run-code-work")

    meta = (tmp_path / "run-code-work" / "meta.json").read_text(encoding="utf-8")
    assert '"task_type": "code_work"' in meta


async def test_runner_uses_session_history_and_writes_final_answer(tmp_path: Path) -> None:
    provider = _FinalAnswerProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path / "runs")
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="learning",
        created_at="t1",
        updated_at="t1",
    )
    store.write_meta(session)
    store.append_message(session.id, "user", "remember python")
    store.append_note(session.id, "user prefers Chinese explanations", "run-prev")

    outcome = await runner.run_and_capture(
        "remember python",
        run_id="run-test",
        session=session,
        store=store,
    )

    assert outcome.status == "success"
    assert provider.seen_messages == [{"role": "user", "content": "remember python"}]
    assert provider.seen_system is not None
    assert "Session notes:" in provider.seen_system
    assert "user prefers Chinese explanations" in provider.seen_system
    assert store.read_messages(session.id) == [
        {"role": "user", "content": "remember python"},
        {"role": "assistant", "content": "done"},
    ]
    assert (tmp_path / "sessions" / session.id / "runs" / "run-test" / "events.jsonl").exists()
    session_run_meta = (
        tmp_path / "sessions" / session.id / "runs" / "run-test" / "meta.json"
    ).read_text(encoding="utf-8")
    assert '"task_type": "chat"' in session_run_meta
    assert not (tmp_path / "runs" / "run-test" / "events.jsonl").exists()


async def test_delegation_preflight_is_not_persisted_to_session_history(
    tmp_path: Path,
) -> None:
    provider = _FinalAnswerProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path / "runs")
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-delegation",
        mode="chat",
        status="active",
        title="delegation",
        created_at="t1",
        updated_at="t1",
    )
    store.write_meta(session)
    goal = (
        "First inspect the architecture, then split the work, and after that "
        "implement the scheduler changes."
    )
    store.append_message(session.id, "user", goal)

    await runner.run_and_capture(
        goal,
        run_id="run-delegation",
        session=session,
        store=store,
    )

    assert provider.seen_system is not None
    assert "Automatic delegation preflight:" in provider.seen_system
    stored = store.read_messages(session.id)
    assert stored == [
        {"role": "user", "content": goal},
        {"role": "assistant", "content": "done"},
    ]
    assert "Automatic delegation preflight:" not in str(stored)


async def test_runner_appends_only_messages_after_compaction_cursor(
    tmp_path: Path,
) -> None:
    provider = _CompactThenAnswerProvider()
    runner = AgentRunner(Config(), provider=provider, runs_dir=tmp_path / "runs")
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="learning",
        created_at="t1",
        updated_at="t1",
    )
    store.write_meta(session)
    store.append_message(session.id, "user", "x" * 480_001, run_id="run-old")

    outcome = await runner.run_and_capture(
        "continue",
        run_id="run-compact",
        session=session,
        store=store,
    )

    assert outcome.status == "success"
    assert outcome.result == "answer after compact"
    assert [call["run_id"] for call in provider.calls] == ["compact", "run-compact"]
    second_call_messages = provider.calls[1]["messages"]
    assert second_call_messages[:2] == [
        {
            "role": "user",
            "content": "Compacted conversation summary:\n\nsummary after compact",
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue from this compacted summary.",
        },
    ]
    assert len(second_call_messages) == 3
    assert second_call_messages[2]["role"] == "user"
    assert isinstance(second_call_messages[2]["content"], str)
    assert "chars omitted" in second_call_messages[2]["content"]
    assert store.read_messages(session.id) == [
        {
            "role": "user",
            "content": "Compacted conversation summary:\n\nsummary after compact",
            "compacted": True,
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue from this compacted summary.",
            "compacted": True,
        },
        {
            "role": "user",
            "content": second_call_messages[2]["content"],
            "compacted_recent": True,
        },
        {"role": "assistant", "content": "answer after compact"},
    ]
    assert (
        tmp_path
        / "sessions"
        / session.id
        / "archive"
        / "thread-before-run-compact.jsonl"
    ).exists()


async def test_runner_does_not_compact_when_compact_is_disabled(
    tmp_path: Path,
) -> None:
    provider = _CompactThenAnswerProvider()
    config = Config(compact=CompactConfig(enabled=False, token_threshold=1))
    runner = AgentRunner(config, provider=provider, runs_dir=tmp_path / "runs")
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="learning",
        created_at="t1",
        updated_at="t1",
    )
    store.write_meta(session)
    store.append_message(session.id, "user", "x" * 100, run_id="run-old")

    outcome = await runner.run_and_capture(
        "continue",
        run_id="run-no-compact",
        session=session,
        store=store,
    )

    assert outcome.status == "success"
    assert [call["run_id"] for call in provider.calls] == ["run-no-compact"]
    assert not (tmp_path / "sessions" / session.id / "archive").exists()


async def test_runner_compacts_after_successful_chat_when_context_ratio_is_high(
    tmp_path: Path,
) -> None:
    provider = _CompactThenAnswerProvider()
    config = Config(compact=CompactConfig(context_ratio=0.80))
    runner = AgentRunner(config, provider=provider, runs_dir=tmp_path / "runs")
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="learning",
        created_at="t1",
        updated_at="t1",
    )
    store.write_meta(session)
    store.append_message(session.id, "user", "old context", run_id="run-old")

    class _HighContextProvider(_CompactThenAnswerProvider):
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
                    "messages": list(messages),
                    "tool_schemas": tool_schemas,
                    "run_id": run_id,
                    "step": step,
                    "system": system,
                }
            )
            if run_id == "compact":
                return LlmResponse(stop_reason="end_turn", text="summary after run")
            return LlmResponse(
                stop_reason="end_turn",
                text="final answer",
                usage=UsageStats(input_tokens=90, output_tokens=10, context_pct=0.90),
            )

    high_context_provider = _HighContextProvider()
    runner = AgentRunner(config, provider=high_context_provider, runs_dir=tmp_path / "runs")

    outcome = await runner.run_and_capture(
        "continue",
        run_id="run-high-context",
        session=session,
        store=store,
    )

    assert outcome.status == "success"
    assert outcome.result == "final answer"
    assert [call["run_id"] for call in high_context_provider.calls] == [
        "run-high-context",
        "compact",
    ]
    assert store.read_messages(session.id) == [
        {
            "role": "user",
            "content": "Compacted conversation summary:\n\nsummary after run",
            "compacted": True,
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue from this compacted summary.",
            "compacted": True,
        },
        {"role": "user", "content": "old context", "compacted_recent": True},
        {"role": "assistant", "content": "final answer", "compacted_recent": True},
    ]
