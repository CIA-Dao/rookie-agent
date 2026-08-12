from __future__ import annotations

from my_agent.core.task.model import Task


def test_task_to_dict_keys() -> None:
    task = Task(
        id=1,
        subject="test",
        description="desc",
        status="pending",
        blocked_by=[],
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )

    data = task.to_dict()

    assert set(data) == {
        "id",
        "subject",
        "description",
        "status",
        "blocked_by",
        "created_at",
        "updated_at",
        "assigned_run_id",
        "completed_by_run_id",
        "failed_by_run_id",
        "failure_reason",
        "task_type",
        "priority",
        "risk",
        "suggested_agent_level",
        "required_capabilities",
        "expected_outputs",
        "acceptance_criteria",
        "can_parallelize",
        "requires_human_review",
        "estimated_complexity",
    }


def test_task_roundtrip() -> None:
    task = Task(
        id=3,
        subject="write tests",
        description="cover all tools",
        status="in_progress",
        blocked_by=[1, 2],
        created_at="t1",
        updated_at="t2",
        assigned_run_id="run-a",
        completed_by_run_id="",
        failed_by_run_id="",
        failure_reason="",
        task_type="implementation",
        priority=3,
        risk="low",
        suggested_agent_level="child",
        required_capabilities=["read_code", "edit_code"],
        expected_outputs=["patch"],
        acceptance_criteria=["tests pass"],
        can_parallelize=True,
        requires_human_review=False,
        estimated_complexity="medium",
    )

    restored = Task.from_dict(task.to_dict())

    assert task == restored


def test_task_from_dict_defaults_and_casts() -> None:
    task = Task.from_dict(
        {
            "id": "4",
            "subject": "loose input",
            "blocked_by": ["1", 2],
        }
    )

    assert task.id == 4
    assert task.description == ""
    assert task.status == "pending"
    assert task.blocked_by == [1, 2]
    assert task.created_at == ""
    assert task.updated_at == ""
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


def test_task_blocked_by_not_shared() -> None:
    first = Task(
        id=1,
        subject="a",
        description="",
        status="pending",
        blocked_by=[],
        created_at="",
        updated_at="",
    )

    second = Task(
        id=2,
        subject="b",
        description="",
        status="pending",
        blocked_by=[],
        created_at="",
        updated_at="",
    )

    first.blocked_by.append(99)

    assert second.blocked_by == []


def test_task_metadata_lists_not_shared() -> None:
    first = Task(
        id=1,
        subject="a",
        description="",
        status="pending",
        blocked_by=[],
        created_at="",
        updated_at="",
    )

    second = Task(
        id=2,
        subject="b",
        description="",
        status="pending",
        blocked_by=[],
        created_at="",
        updated_at="",
    )

    first.required_capabilities.append("edit_code")
    first.expected_outputs.append("patch")
    first.acceptance_criteria.append("tests pass")

    assert second.required_capabilities == []
    assert second.expected_outputs == []
    assert second.acceptance_criteria == []
