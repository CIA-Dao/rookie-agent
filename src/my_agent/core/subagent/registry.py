from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from my_agent.core.context import ExecutionContext

SubagentStatus = Literal["running", "success", "failed", "cancelled", "timed_out"]


@dataclass
class BackgroundSubagentRecord:
    run_id: str
    parent_run_id: str
    root_run_id: str
    session_id: str
    depth: int
    description: str
    subagent_type: str
    run_in_background: bool
    created_at: str
    task: asyncio.Task[None] | None
    context: ExecutionContext
    status: SubagentStatus = "running"
    completed_at: str = ""
    reason: str = ""


@dataclass(frozen=True)
class SubagentLimits:
    max_depth: int = 2
    max_children_per_root: int = 4
    max_grandchildren_per_child: int = 2
    max_total_descendants_per_root: int = 8
    max_concurrent_background_subagents_per_session: int = 4
    background_timeout_seconds: float | None = 600.0
    max_completed_records: int = 100


class BackgroundTaskRegistry:
    def __init__(self) -> None:
        self._records: dict[str, BackgroundSubagentRecord] = {}

    def register(
        self,
        run_id: str,
        task: asyncio.Task[None] | None,
        context: ExecutionContext,
        *,
        parent_run_id: str = "",
        root_run_id: str = "",
        session_id: str = "",
        depth: int = 0,
        description: str = "",
        subagent_type: str = "",
        run_in_background: bool = True,
        created_at: str = "",
        status: SubagentStatus = "running",
        completed_at: str = "",
        reason: str = "",
    ) -> None:
        self._records[run_id] = BackgroundSubagentRecord(
            run_id=run_id,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id or parent_run_id,
            session_id=session_id,
            depth=depth,
            description=description,
            subagent_type=subagent_type,
            run_in_background=run_in_background,
            created_at=created_at,
            task=task,
            context=context,
            status=status,
            completed_at=completed_at,
            reason=reason,
        )

    def register_record(self, record: BackgroundSubagentRecord) -> None:
        self._records[record.run_id] = record

    def get(self, run_id: str) -> tuple[asyncio.Task[None], ExecutionContext] | None:
        record = self._records.get(run_id)
        if record is None or record.task is None:
            return None
        return (record.task, record.context)

    def get_record(self, run_id: str) -> BackgroundSubagentRecord | None:
        return self._records.get(run_id)

    def all(self) -> list[tuple[asyncio.Task[None], ExecutionContext]]:
        return [
            (record.task, record.context)
            for record in self._records.values()
            if record.task is not None
        ]

    def records(self) -> list[BackgroundSubagentRecord]:
        return list(self._records.values())

    def count_descendants(self, root_run_id: str) -> int:
        return sum(1 for record in self._records.values() if record.root_run_id == root_run_id)

    def count_direct_children(self, parent_run_id: str) -> int:
        return sum(1 for record in self._records.values() if record.parent_run_id == parent_run_id)

    def count_running_background(self, session_id: str) -> int:
        return sum(
            1
            for record in self._records.values()
            if record.session_id == session_id
            and record.run_in_background
            and record.task is not None
            and record.status == "running"
            and not record.task.done()
        )

    def mark(
        self,
        run_id: str,
        status: SubagentStatus,
        *,
        completed_at: str = "",
        reason: str = "",
    ) -> None:
        record = self._records.get(run_id)
        if record is None:
            return
        record.status = status
        if completed_at:
            record.completed_at = completed_at
        if reason:
            record.reason = reason

    def cancel(self, run_id: str, *, reason: str = "cancelled") -> bool:
        record = self._records.get(run_id)
        if record is None or record.task is None:
            return False
        if record.status != "running" or record.task.done():
            return False
        record.reason = reason
        return record.task.cancel()

    def descendants_of(self, run_id: str) -> list[BackgroundSubagentRecord]:
        descendants: list[BackgroundSubagentRecord] = []
        pending = [run_id]
        while pending:
            parent = pending.pop()
            children = [
                record
                for record in self._records.values()
                if record.parent_run_id == parent
            ]
            descendants.extend(children)
            pending.extend(record.run_id for record in children)
        return descendants

    def cancel_tree(self, run_id: str, *, reason: str = "cancelled") -> int:
        cancelled = 0
        for record in [*self.descendants_of(run_id), self._records.get(run_id)]:
            if record is None or record.task is None:
                continue
            if record.status == "running" and not record.task.done():
                record.reason = reason
                if record.task.cancel():
                    cancelled += 1
        return cancelled

    def prune_completed(self, max_records: int) -> int:
        if max_records < 0:
            max_records = 0
        terminal = [
            record
            for record in self._records.values()
            if record.status != "running"
        ]
        if len(terminal) <= max_records:
            return 0

        terminal.sort(key=lambda record: record.completed_at or record.created_at)
        remove = terminal[: len(terminal) - max_records]
        for record in remove:
            self._records.pop(record.run_id, None)
        return len(remove)
