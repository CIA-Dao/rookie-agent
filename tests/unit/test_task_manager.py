from __future__ import annotations

import json
from pathlib import Path

import pytest

from my_agent.core.task.manager import TaskManager


def test_create_writes_file(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    task = manager.create("do something")
    assert task.id == 1
    assert task.subject == "do something"
    assert task.description == ""
    assert task.status == "pending"
    assert task.blocked_by == []
    assert (tmp_path / "task_1.json").exists()


def test_create_stores_scheduler_metadata(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    task = manager.create(
        "implement parser",
        task_type="implementation",
        priority=5,
        risk="high",
        suggested_agent_level="child",
        required_capabilities=["read_code", "edit_code"],
        expected_outputs=["code patch", "unit tests"],
        acceptance_criteria=["pytest passes"],
        can_parallelize=True,
        requires_human_review=True,
        estimated_complexity="medium",
    )

    assert task.task_type == "implementation"
    assert task.priority == 5
    assert task.risk == "high"
    assert task.suggested_agent_level == "child"
    assert task.required_capabilities == ["read_code", "edit_code"]
    assert task.expected_outputs == ["code patch", "unit tests"]
    assert task.acceptance_criteria == ["pytest passes"]
    assert task.can_parallelize is True
    assert task.requires_human_review is True
    assert task.estimated_complexity == "medium"


def test_create_increments_id(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    first = manager.create("first")
    second = manager.create("second")

    assert first.id == 1
    assert second.id == 2
    assert (tmp_path / "task_1.json").exists()
    assert (tmp_path / "task_2.json").exists()


def test_get_returns_task(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    manager.create("hello")
    task = manager.get(1)

    assert task.subject == "hello"


def test_get_nonexistent_raises(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        manager.get(999)


def test_create_invalid_blocked_by_raises(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        manager.create("dependent", blocked_by=[999])


def test_update_status(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    manager.create("work")
    manager.update(1, status="in_progress")

    assert manager.get(1).status == "in_progress"


def test_update_failed_status(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    manager.create("work")
    task = manager.update(1, status="failed", failure_reason="tool unavailable")

    assert task.status == "failed"
    assert task.failure_reason == "tool unavailable"


def test_update_metadata_does_not_change_status_blockers_or_run_fields(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("a")
    manager.create("b", blocked_by=[1])
    manager.assign_run(2, "run-child-1")

    task = manager.update(
        2,
        task_type="test",
        priority=7,
        risk="low",
        suggested_agent_level="grandchild",
        required_capabilities=["run_tests"],
        expected_outputs=["test report"],
        acceptance_criteria=["failure reproduced"],
        can_parallelize=True,
        requires_human_review=False,
        estimated_complexity="small",
    )

    assert task.status == "in_progress"
    assert task.blocked_by == [1]
    assert task.assigned_run_id == "run-child-1"
    assert task.completed_by_run_id == ""
    assert task.failed_by_run_id == ""
    assert task.failure_reason == ""
    assert task.task_type == "test"
    assert task.priority == 7
    assert task.risk == "low"
    assert task.suggested_agent_level == "grandchild"
    assert task.required_capabilities == ["run_tests"]
    assert task.expected_outputs == ["test report"]
    assert task.acceptance_criteria == ["failure reproduced"]
    assert task.can_parallelize is True
    assert task.requires_human_review is False
    assert task.estimated_complexity == "small"


def test_update_completed_clears_dependency(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    manager.create("step 1")
    manager.create("step 2", blocked_by=[1])

    manager.update(1, status="completed")

    assert manager.get(2).blocked_by == []
    assert manager.get(2).status == "pending"


def test_run_lifecycle_helpers_track_owner_and_clear_dependencies(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("step 1")
    manager.create("step 2", blocked_by=[1])

    assigned = manager.assign_run(1, "run-child-1")
    completed = manager.complete_with_run(1, "run-child-1")

    assert assigned.status == "in_progress"
    assert assigned.assigned_run_id == "run-child-1"
    assert completed.status == "completed"
    assert completed.assigned_run_id == "run-child-1"
    assert completed.completed_by_run_id == "run-child-1"
    assert completed.failed_by_run_id == ""
    assert completed.failure_reason == ""
    assert manager.get(2).blocked_by == []


def test_fail_with_run_records_reason_without_clearing_dependencies(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("step 1")
    manager.create("step 2", blocked_by=[1])

    failed = manager.fail_with_run(1, "run-child-1", "tests failed")

    assert failed.status == "failed"
    assert failed.assigned_run_id == "run-child-1"
    assert failed.failed_by_run_id == "run-child-1"
    assert failed.failure_reason == "tests failed"
    assert manager.get(2).blocked_by == [1]


def test_update_add_blocked_by(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("a")
    manager.create("b")

    manager.update(2, add_blocked_by=[1])

    assert 1 in manager.get(2).blocked_by


def test_update_remove_blocked_by(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("a")
    manager.create("b", blocked_by=[1])

    manager.update(2, remove_blocked_by=[1])

    assert manager.get(2).blocked_by == []


def test_list_all_ordered(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    manager.create("a")
    manager.create("b")
    manager.create("c")

    tasks = manager.list_all()

    assert len(tasks) == 3
    assert [task.id for task in tasks] == [1, 2, 3]


def test_task_graph_status_queries(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("ready")
    manager.create("blocked", blocked_by=[1])
    manager.create("started")
    manager.create("broken")

    manager.assign_run(3, "run-started")
    manager.fail_with_run(4, "run-failed", "no compatible tool")

    assert [task.id for task in manager.ready_tasks()] == [1]
    assert [task.id for task in manager.blocked_tasks()] == [2]
    assert [task.id for task in manager.in_progress_tasks()] == [3]
    assert [task.id for task in manager.failed_tasks()] == [4]


def test_detect_cycles_reports_direct_cycle(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("a")
    manager.create("b", blocked_by=[1])
    manager.update(1, add_blocked_by=[2])

    assert manager.detect_cycles() == [[1, 2, 1]]


def test_detect_cycles_reports_longer_cycle(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("a")
    manager.create("b")
    manager.create("c")
    manager.update(1, add_blocked_by=[3])
    manager.update(2, add_blocked_by=[1])
    manager.update(3, add_blocked_by=[2])

    assert manager.detect_cycles() == [[1, 3, 2, 1]]


def test_diagnose_deadlock_ignores_graph_with_ready_work(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("ready")
    manager.create("blocked", blocked_by=[1])

    assert manager.diagnose_deadlock() is None


def test_diagnose_deadlock_reports_blocked_only_graph(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("missing external blocker")
    manager.update(1, add_blocked_by=[99])

    assert manager.diagnose_deadlock() == "deadlock: pending tasks are blocked: [1]"


def test_diagnose_deadlock_reports_cycle(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("a")
    manager.create("b", blocked_by=[1])
    manager.update(1, add_blocked_by=[2])

    assert manager.diagnose_deadlock() == "cycle detected: 1 -> 2 -> 1"


def test_old_task_json_loads_with_empty_run_fields(tmp_path: Path) -> None:
    (tmp_path / "task_1.json").write_text(
        json.dumps(
            {
                "id": 1,
                "subject": "old task",
                "description": "",
                "status": "pending",
                "blocked_by": [],
                "created_at": "t1",
                "updated_at": "t2",
            }
        ),
        encoding="utf-8",
    )
    manager = TaskManager(tmp_path)

    task = manager.get(1)

    assert task.assigned_run_id == ""
    assert task.completed_by_run_id == ""
    assert task.failed_by_run_id == ""
    assert task.failure_reason == ""
    assert task.task_type == "general"
    assert task.priority == 0
    assert task.risk == "medium"
    assert task.suggested_agent_level == ""
    assert task.required_capabilities == []
    assert task.expected_outputs == []
    assert task.acceptance_criteria == []
    assert task.can_parallelize is False
    assert task.requires_human_review is False
    assert task.estimated_complexity == ""


def test_format_list_content(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    manager.create("alpha")
    manager.create("beta")

    manager.update(1, status="completed")

    result = manager.format_list()

    assert "[x]" in result
    assert "alpha" in result
    assert "beta" in result


def test_format_list_shows_failed_tasks(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("alpha")
    manager.fail_with_run(1, "run-1", "no match")

    result = manager.format_list()

    assert "[!] #1: alpha" in result
    assert "run-1" in result
    assert "no match" in result


def test_manager_resumes_id_from_existing_files(tmp_path: Path) -> None:
    first_manager = TaskManager(tmp_path)

    first_manager.create("first")
    first_manager.create("second")

    second_manager = TaskManager(tmp_path)
    task = second_manager.create("third")

    assert task.id == 3
    assert (tmp_path / "task_3.json").exists()
