from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

RUNS_DIR = Path("runs")


def new_run_id() -> str:
    """生成格式为 YYYYMMDD-HHMMSS-xxxxxx 的唯一 run ID"""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}-{suffix}"


def run_dir(run_id: str, runs_dir: Path | None = None ) -> Path:
    root = runs_dir or RUNS_DIR
    return root / run_id


def events_file(run_id: str, runs_dir: Path | None = None) -> Path:
    return run_dir(run_id, runs_dir) / "events.jsonl"
