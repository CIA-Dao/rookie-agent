from __future__ import annotations

import json
from pathlib import Path

from my_agent.core.subagent.registry import BackgroundTaskRegistry
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.builtin.orchestration_summary import OrchestrationSummaryTool


def _tool(
    manager: TaskManager,
    registry: BackgroundTaskRegistry,
) -> OrchestrationSummaryTool:
    return OrchestrationSummaryTool(
        manager,
        registry,
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )


async def test_orchestration_summary_reports_empty_graph(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / "tasks")
    result = await _tool(manager, BackgroundTaskRegistry()).invoke({})
    data = json.loads(result.content)

    assert not result.is_error
    assert data["counts"]["total"] == 0
    assert data["recovery"]["next_actions"] == ["create_tasks"]
    assert "completed task count (0/0)" in data["final_synthesis_prompt"]


async def test_orchestration_summary_reports_failure_recovery(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("fix failing test")
    manager.fail_with_run(task.id, "child-1", "pytest failed")

    before = manager.get(task.id).to_dict()
    result = await _tool(manager, BackgroundTaskRegistry()).invoke({})
    after = manager.get(task.id).to_dict()
    data = json.loads(result.content)

    assert before == after
    assert data["counts"]["failed"] == 1
    assert data["scheduler"]["requires_human_review"] is True
    assert data["recovery"]["failed_tasks"][0]["task_id"] == task.id
    assert data["recovery"]["failed_tasks"][0]["failure_reason"] == "pytest failed"
    assert "review_failed_tasks" in data["recovery"]["next_actions"]


async def test_orchestration_summary_reports_missing_background_record(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("running elsewhere")
    manager.assign_run(task.id, "missing-child-run")

    result = await _tool(manager, BackgroundTaskRegistry()).invoke({})
    data = json.loads(result.content)

    assert data["counts"]["in_progress"] == 1
    assert data["recovery"]["missing_background_records"] == [
        {
            "task_id": task.id,
            "subject": "running elsewhere",
            "assigned_run_id": "missing-child-run",
            "guidance": (
                "The task is in_progress but the assigned background run is not "
                "present in the current in-memory registry. Review manually or "
                "reset/retry explicitly."
            ),
        }
    ]
    assert "collect_or_wait_for_running_tasks" in data["recovery"]["next_actions"]


async def test_orchestration_summary_can_omit_task_details(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / "tasks")
    manager.create("first")

    result = await _tool(manager, BackgroundTaskRegistry()).invoke({"include_tasks": False})
    data = json.loads(result.content)

    assert "tasks" not in data
    assert data["counts"]["ready"] == 1
