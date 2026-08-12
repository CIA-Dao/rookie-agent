from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from my_agent.core.app import CoreApp
from my_agent.core.config import Config
from my_agent.core.session.model import Session, SessionMode


class _FakeSessionManager:
    def __init__(self) -> None:
        self.created: list[tuple[str, str, str]] = []
        self.sent: list[tuple[str, str, str | None]] = []

    async def create(
        self,
        mode: SessionMode,
        title: str = "",
        *,
        workspace_root: str = "",
    ) -> Session:
        self.created.append((mode, title, workspace_root))
        return Session(
            id=f"sess-{len(self.created)}",
            mode=mode,
            status="active",
            title=title,
            created_at="t1",
            updated_at="t1",
            run_ids=[],
            workspace_root=workspace_root,
        )

    async def send_message(
        self,
        sid: str,
        content: str,
        *,
        run_id: str | None = None,
    ) -> str:
        self.sent.append((sid, content, run_id))
        await asyncio.Event().wait()
        return run_id or "run-generated"


async def test_agent_run_handler_allows_multiple_running_tasks() -> None:
    app = CoreApp()
    app._config = Config()
    app._session_manager = _FakeSessionManager()  # type: ignore[assignment]

    first = await app._agent_run_handler({"type": "agent.run", "goal": "first"})
    second = await app._agent_run_handler({"type": "agent.run", "goal": "second"})

    assert first.run_id != second.run_id

    tasks: set[asyncio.Task[None]] = getattr(app, "_running_runs", set())
    for task in list(tasks):
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_agent_run_handler_uses_one_shot_session() -> None:
    sessions = _FakeSessionManager()

    app = CoreApp()
    app._config = Config()
    app._session_manager = sessions  # type: ignore[assignment]

    result = await app._agent_run_handler({"type": "agent.run", "goal": "hello"})
    await asyncio.sleep(0)

    assert sessions.created == [("one_shot", "hello", "")]
    assert sessions.sent == [("sess-1", "hello", result.run_id)]

    tasks: set[asyncio.Task[None]] = getattr(app, "_running_runs", set())
    for task in list(tasks):
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_runner_loads_global_and_project_context(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """AgentRunner 把全局 context 和项目 context 注入 ExecutionContext。"""
    from my_agent.core.context import ExecutionContext
    from my_agent.core.runner import AgentRunner

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # 用假的 load_context_file 返回受控值
    def fake_load(path: Path) -> str:
        s = str(path)
        if ".my-agent" in s and s.endswith("context.md"):
            if "workspace" in s:
                return "project: specific rules"
            # 返回全局 context（通过 home path 匹配）
            return "global: rules here"
        return ""

    monkeypatch.setattr("my_agent.core.runner.load_context_file", fake_load)

    config = Config()

    captured_contexts: list[ExecutionContext] = []

    class FakeLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, context: ExecutionContext) -> None:
            captured_contexts.append(context)

    monkeypatch.setattr("my_agent.core.runner.AgentLoop", FakeLoop)
    monkeypatch.setattr("my_agent.core.runner.DeepSeekProvider", MagicMock())

    runner = AgentRunner(config, runs_dir=tmp_path / "runs")
    runner._provider = MagicMock()  # type: ignore[assignment]

    session = Session(
        id="sess-test",
        mode="one_shot",
        status="active",
        title="test",
        created_at="t1",
        updated_at="t1",
        run_ids=[],
        workspace_root=str(workspace),
    )

    outcome = await runner.run_and_capture("test goal", session=session)
    assert outcome.status == "running"

    assert len(captured_contexts) >= 1, "AgentLoop.run was never called with a context"
    ctx = captured_contexts[0]
    assert ctx.global_context == "global: rules here"
    assert ctx.project_context == "project: specific rules"
