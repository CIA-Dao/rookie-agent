from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

from my_agent.core.app import CoreApp
from my_agent.core.config import Config


class _FakeWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        pass


async def test_replay_events_reads_utf8_event_logs_from_configured_runs_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-test"
    run_dir.mkdir()
    event = {
        "type": "run.started",
        "run_id": "run-test",
        "goal": "introduce yourself",
        "ts": "2026-01-01T00:00:00Z",
    }
    (run_dir / "events.jsonl").write_text(
        json.dumps(event, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    app = CoreApp()
    app._config = Config(runs_dir=str(tmp_path))
    writer = _FakeWriter()

    replayed_count = await app._replay_events(
        "run-test",
        cast(asyncio.StreamWriter, writer),
        ["run.*"],
    )

    assert replayed_count == 1
    assert len(writer.chunks) == 1


async def test_replay_events_falls_back_to_session_run_logs(tmp_path: Path) -> None:
    run_dir = tmp_path / "sessions" / "sess-1" / "runs" / "run-test"
    run_dir.mkdir(parents=True)
    event = {
        "type": "run.started",
        "run_id": "run-test",
        "goal": "from session",
        "ts": "2026-01-01T00:00:00Z",
    }
    (run_dir / "events.jsonl").write_text(
        json.dumps(event, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    app = CoreApp()
    app._config = Config(runs_dir=str(tmp_path))
    writer = _FakeWriter()

    replayed_count = await app._replay_events(
        "run-test",
        cast(asyncio.StreamWriter, writer),
        ["run.*"],
    )

    assert replayed_count == 1
    assert len(writer.chunks) == 1
