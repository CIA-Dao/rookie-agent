from __future__ import annotations

from my_agent.tui.event_policy import (
    context_compacted_entry,
    context_compaction_failed_entry,
    run_finished_entry,
    scheduler_event_entry,
    skill_invoked_entry,
    skill_tool_compatibility_entry,
    subagent_finished_entry,
    subagent_started_entry,
    task_event_entry,
    unknown_event_entry,
)


def test_run_finished_entry_distinguishes_build_verification_failure() -> None:
    entry = run_finished_entry(
        {
            "type": "run.finished",
            "status": "failed",
            "delivery": {
                "build_status": "failed",
                "final_status": "not_accepted",
            },
        }
    )

    assert entry.category == "verification"
    assert entry.content == "completed: build verification failed"
    assert entry.visible is True


def test_run_finished_entry_keeps_success_low_noise() -> None:
    entry = run_finished_entry({"type": "run.finished", "status": "success"})

    assert entry.category == "run"
    assert entry.content == "completed"
    assert entry.visible is False


def test_skill_invoked_entry_is_hidden_by_default() -> None:
    entry = skill_invoked_entry({"skill_name": "review", "arguments": "README.md"})

    assert entry.category == "skill"
    assert entry.content == "review README.md"
    assert entry.visible is False


def test_skill_tool_compatibility_entry_renders_aliases_and_missing_tools() -> None:
    entry = skill_tool_compatibility_entry(
        {
            "skill_name": "third-party",
            "aliases": [{"from": "shell", "to": "bash"}],
            "unresolved_tools": ["browser.open"],
        }
    )

    assert entry.category == "skill"
    assert entry.visible is True
    assert entry.content == "third-party // mapped shell -> bash; missing browser.open"


def test_subagent_entries_are_visible_and_compact() -> None:
    started = subagent_started_entry(
        {"run_id": "child-run", "description": "review README.md"}
    )
    finished = subagent_finished_entry({"run_id": "child-run", "status": "success"})

    assert started.category == "subagent"
    assert started.content == "started: review README.md child=child-run"
    assert started.visible is True
    assert finished.category == "subagent"
    assert finished.content == "finished: child-run status=success"
    assert finished.visible is True


def test_context_entries_are_visible() -> None:
    compacted = context_compacted_entry({"original_tokens": 120, "summary_tokens": 40})
    failed = context_compaction_failed_entry({"reason": "budget exhausted"})

    assert compacted.category == "system"
    assert compacted.content == "context compacted original=120 summary=40 saved~=80"
    assert compacted.visible is True
    assert failed.category == "system"
    assert failed.content == "context compaction failed: budget exhausted"
    assert failed.visible is True


def test_unknown_task_and_scheduler_events_are_visible_but_other_unknowns_are_hidden() -> None:
    task = unknown_event_entry("task.assigned")
    scheduler = unknown_event_entry("scheduler.plan.generated")
    internal = unknown_event_entry("engine.internal_noise")

    assert task.category == "task"
    assert task.content == "task.assigned"
    assert task.visible is True
    assert scheduler.category == "scheduler"
    assert scheduler.content == "scheduler.plan.generated"
    assert scheduler.visible is True
    assert internal.category == "system"
    assert internal.content == "engine.internal_noise"
    assert internal.visible is False


def test_task_event_entries_include_operational_fields() -> None:
    assigned = task_event_entry(
        {
            "type": "task.assigned",
            "task_id": 2,
            "subject": "write tests",
            "assigned_run_id": "child-1",
        }
    )
    completed = task_event_entry(
        {
            "type": "task.completed",
            "task_id": 2,
            "subject": "write tests",
            "completed_by_run_id": "child-1",
        }
    )
    failed = task_event_entry(
        {
            "type": "task.failed",
            "task_id": 3,
            "subject": "lint",
            "failed_by_run_id": "child-2",
            "failure_reason": "ruff failed",
        }
    )

    assert assigned.content == "assigned #2 write tests -> child-1"
    assert assigned.visible is True
    assert completed.content == "completed #2 write tests by child-1"
    assert failed.content == "failed #3 lint by child-2 reason=ruff failed"


def test_scheduler_event_entries_include_counts_and_flags() -> None:
    plan = scheduler_event_entry(
        {
            "type": "scheduler.plan.generated",
            "plan_id": "plan-1",
            "ready_task_ids": [1, 2],
            "dispatchable_task_ids": [1],
            "skipped_task_ids": [2],
            "should_replan": True,
            "requires_human_review": False,
            "diagnostics_count": 1,
        }
    )
    diagnosis = scheduler_event_entry(
        {
            "type": "scheduler.diagnosis.reported",
            "plan_id": "plan-1",
            "diagnostics": ["cycle detected: 1 -> 2 -> 1"],
        }
    )
    skipped = scheduler_event_entry(
        {
            "type": "scheduler.dispatch.skipped",
            "plan_id": "plan-1",
            "skipped": [{"task_id": 2, "reason": "high_risk_root_review"}],
        }
    )

    assert plan.content == (
        "plan plan=plan-1 ready=2 dispatchable=1 skipped=1 "
        "replan=True review=False diagnostics=1"
    )
    assert diagnosis.content == "diagnosis plan=plan-1: cycle detected: 1 -> 2 -> 1"
    assert skipped.content == "dispatch skipped plan=plan-1 count=1"
    assert plan.visible is True
