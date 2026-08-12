from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from my_agent.core.task.model import Task, TaskStatus

VALID_STATUSES: tuple[TaskStatus, ...] = ("pending", "in_progress", "completed", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskManager:
    def __init__(self, tasks_dir: Path) -> None:
        self._dir = tasks_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [
            int(path.stem.split("_")[1])
            for path in self._dir.glob("task_*.json")
            if path.stem.split("_")[1].isdigit()
        ]

        return max(ids) if ids else 0

    def _save(self, task: Task) -> None:
        path = self._dir / f"task_{task.id}.json"
        path.write_text(json.dumps(task.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self, task_id: int) -> Task:
        path = self._dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"task {task_id} not found")
        return Task.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def get(self, task_id: int) -> Task:
        return self._load(task_id)

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[int] | None = None,
        *,
        task_type: str = "general",
        priority: int = 0,
        risk: str = "medium",
        suggested_agent_level: str = "",
        required_capabilities: list[str] | None = None,
        expected_outputs: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        can_parallelize: bool = False,
        requires_human_review: bool = False,
        estimated_complexity: str = "",
    ) -> Task:

        for dep_id in blocked_by or []:
            if not (self._dir / f"task_{dep_id}.json").exists():
                raise ValueError(f"blocked_by task {dep_id} not found")
        now = _now()
        task = Task(
            id=self._next_id,
            subject=subject,
            description=description,
            status="pending",
            blocked_by=list(blocked_by or []),
            created_at=now,
            updated_at=now,
            task_type=task_type,
            priority=priority,
            risk=risk,
            suggested_agent_level=suggested_agent_level,
            required_capabilities=list(required_capabilities or []),
            expected_outputs=list(expected_outputs or []),
            acceptance_criteria=list(acceptance_criteria or []),
            can_parallelize=can_parallelize,
            requires_human_review=requires_human_review,
            estimated_complexity=estimated_complexity,
        )

        self._save(task)
        self._next_id += 1

        return task

    def update(
        self,
        task_id: int,
        *,
        status: TaskStatus | None = None,
        add_blocked_by: list[int] | None = None,
        remove_blocked_by: list[int] | None = None,
        assigned_run_id: str | None = None,
        completed_by_run_id: str | None = None,
        failed_by_run_id: str | None = None,
        failure_reason: str | None = None,
        task_type: str | None = None,
        priority: int | None = None,
        risk: str | None = None,
        suggested_agent_level: str | None = None,
        required_capabilities: list[str] | None = None,
        expected_outputs: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        can_parallelize: bool | None = None,
        requires_human_review: bool | None = None,
        estimated_complexity: str | None = None,
    ) -> Task:
        task = self._load(task_id)

        if status is not None:
            if status not in VALID_STATUSES:
                raise ValueError(f"invalid status: {status!r}")

            task.status = status
            if status == "completed":
                self._clear_dependency(task_id)

        if add_blocked_by:
            task.blocked_by = list(set(task.blocked_by + add_blocked_by))
        if remove_blocked_by:
            task.blocked_by = [x for x in task.blocked_by if x not in remove_blocked_by]

        if assigned_run_id is not None:
            task.assigned_run_id = assigned_run_id
        if completed_by_run_id is not None:
            task.completed_by_run_id = completed_by_run_id
        if failed_by_run_id is not None:
            task.failed_by_run_id = failed_by_run_id
        if failure_reason is not None:
            task.failure_reason = failure_reason
        if task_type is not None:
            task.task_type = task_type
        if priority is not None:
            task.priority = priority
        if risk is not None:
            task.risk = risk
        if suggested_agent_level is not None:
            task.suggested_agent_level = suggested_agent_level
        if required_capabilities is not None:
            task.required_capabilities = list(required_capabilities)
        if expected_outputs is not None:
            task.expected_outputs = list(expected_outputs)
        if acceptance_criteria is not None:
            task.acceptance_criteria = list(acceptance_criteria)
        if can_parallelize is not None:
            task.can_parallelize = can_parallelize
        if requires_human_review is not None:
            task.requires_human_review = requires_human_review
        if estimated_complexity is not None:
            task.estimated_complexity = estimated_complexity

        task.updated_at = _now()
        self._save(task)
        return task

    def ready_tasks(self) -> list[Task]:
        return [
            task for task in self.list_all() if task.status == "pending" and not task.blocked_by
        ]

    def blocked_tasks(self) -> list[Task]:
        return [task for task in self.list_all() if task.status == "pending" and task.blocked_by]

    def in_progress_tasks(self) -> list[Task]:
        return [task for task in self.list_all() if task.status == "in_progress"]

    def failed_tasks(self) -> list[Task]:
        return [task for task in self.list_all() if task.status == "failed"]

    def assign_run(self, task_id: int, run_id: str) -> Task:
        return self.update(
            task_id,
            status="in_progress",
            assigned_run_id=run_id,
            completed_by_run_id="",
            failed_by_run_id="",
            failure_reason="",
        )

    def complete_with_run(self, task_id: int, run_id: str) -> Task:
        task = self._load(task_id)
        assigned_run_id = task.assigned_run_id or run_id
        return self.update(
            task_id,
            status="completed",
            assigned_run_id=assigned_run_id,
            completed_by_run_id=run_id,
            failed_by_run_id="",
            failure_reason="",
        )

    def fail_with_run(self, task_id: int, run_id: str, reason: str) -> Task:
        task = self._load(task_id)
        assigned_run_id = task.assigned_run_id or run_id
        return self.update(
            task_id,
            status="failed",
            assigned_run_id=assigned_run_id,
            failed_by_run_id=run_id,
            failure_reason=reason,
        )

    def detect_cycles(self) -> list[list[int]]:
        tasks = {task.id: task for task in self.list_all()}
        visited: set[int] = set()
        visiting: set[int] = set()
        stack: list[int] = []
        cycles: dict[tuple[int, ...], list[int]] = {}

        def normalize(cycle: list[int]) -> tuple[int, ...]:
            nodes = cycle[:-1]
            min_index = min(range(len(nodes)), key=nodes.__getitem__)
            rotated = nodes[min_index:] + nodes[:min_index]
            return tuple(rotated)

        def dfs(task_id: int) -> None:
            if task_id in visiting:
                start = stack.index(task_id)
                cycle = stack[start:] + [task_id]
                key = normalize(cycle)
                cycles.setdefault(key, list(key) + [key[0]])
                return
            if task_id in visited:
                return

            visiting.add(task_id)
            stack.append(task_id)
            for dep_id in tasks[task_id].blocked_by:
                if dep_id in tasks:
                    dfs(dep_id)
            stack.pop()
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in sorted(tasks):
            if task_id not in visited:
                dfs(task_id)

        return [cycles[key] for key in sorted(cycles)]

    def diagnose_deadlock(self) -> str | None:
        pending = [task for task in self.list_all() if task.status == "pending"]
        if not pending or self.ready_tasks():
            return None

        cycles = self.detect_cycles()
        if cycles:
            rendered = ", ".join(" -> ".join(str(task_id) for task_id in cycle) for cycle in cycles)
            return f"cycle detected: {rendered}"

        blocked_ids = [task.id for task in self.blocked_tasks()]
        return f"deadlock: pending tasks are blocked: {blocked_ids}"

    def _clear_dependency(self, completed_id: int) -> None:
        for path in self._dir.glob("task_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, json.JSONDecodeError):
                continue

            blocked = [int(x) for x in data.get("blocked_by", [])]

            if completed_id in blocked:
                data["blocked_by"] = [x for x in blocked if x != completed_id]
                data["updated_at"] = _now()
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_all(self) -> list[Task]:
        tasks: list[Task] = []
        for path in sorted(
            self._dir.glob("task_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
        ):
            try:
                tasks.append(Task.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
        return tasks

    def format_list(self) -> str:
        tasks = self.list_all()
        if not tasks:
            return "No tasks."

        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "failed": "[!]"}
        lines = []
        for task in tasks:
            blocked = f" (blocked by: {task.blocked_by})" if task.blocked_by else ""
            run = f" (run: {task.assigned_run_id})" if task.assigned_run_id else ""
            failed = f" (failed: {task.failure_reason})" if task.failure_reason else ""
            lines.append(
                f"{marker.get(task.status, '[?]')} #{task.id}: {task.subject}{blocked}{run}{failed}"
            )

        return "\n".join(lines)
