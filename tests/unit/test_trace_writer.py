import asyncio
from pathlib import Path

from my_agent.core.trace.record import TraceRecord
from my_agent.core.trace.writer import TraceWriter


def _record() -> TraceRecord:
    return TraceRecord(
        ts="2026-08-10T00:00:00+00:00",
        direction="CORE->LLM",
        layer="llm",
        kind="test",
        run_id="run",
        step=1,
        data={},
    )


async def test_trace_writer_records_without_changing_payload(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "trace.jsonl")
    await writer.start()
    writer.emit(_record())
    await writer.stop()

    assert '"kind":"test"' in (tmp_path / "trace.jsonl").read_text(encoding="utf-8")


async def test_trace_writer_disables_unwritable_path(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "directory" / "trace.jsonl")
    await writer.start()
    writer.emit(_record())
    await asyncio.sleep(0)
    await writer.stop()

    assert writer.dropped_records == 0
