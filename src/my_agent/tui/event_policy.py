from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventLogEntry:
    category: str
    content: str
    visible: bool = False


def run_finished_entry(event: dict[str, Any]) -> EventLogEntry:
    """Render run completion separately from delivery verification outcome."""
    status = str(event.get("status", "unknown"))
    if status == "success":
        return EventLogEntry("run", "completed")

    delivery = event.get("delivery", {})
    if isinstance(delivery, dict):
        build_status = str(delivery.get("build_status", ""))
        import_status = str(delivery.get("import_status", ""))
        final_status = str(delivery.get("final_status", ""))
        if build_status == "failed":
            return EventLogEntry("verification", "completed: build verification failed", True)
        if import_status == "failed":
            return EventLogEntry("verification", "completed: import verification failed", True)
        if final_status == "not_accepted":
            return EventLogEntry("delivery", "completed: delivery not accepted", True)

    reason = str(event.get("reason", "")).strip()
    suffix = f": {reason}" if reason else ""
    return EventLogEntry("run", f"completed with failure{suffix}", True)


def skill_invoked_entry(event: dict[str, Any]) -> EventLogEntry:
    skill_name = str(event.get("skill_name", "unknown"))
    arguments = str(event.get("arguments", "")).strip()
    content = f"{skill_name} {arguments}" if arguments else skill_name
    return EventLogEntry("skill", content)


def skill_tool_compatibility_entry(event: dict[str, Any]) -> EventLogEntry:
    skill_name = str(event.get("skill_name", "unknown"))
    aliases = event.get("aliases", []) or []
    unresolved = event.get("unresolved_tools", []) or []
    parts: list[str] = []
    for alias in aliases:
        if isinstance(alias, dict):
            src = str(alias.get("from", ""))
            dst = str(alias.get("to", ""))
            if src and dst:
                parts.append(f"mapped {src} -> {dst}")
    if unresolved:
        missing = ", ".join(str(name) for name in unresolved)
        parts.append(f"missing {missing}")
    content = f"{skill_name} // " + "; ".join(parts) if parts else skill_name
    return EventLogEntry("skill", content, visible=True)


def subagent_started_entry(event: dict[str, Any]) -> EventLogEntry:
    child_run_id = str(event.get("run_id", ""))
    description = str(event.get("description", "")).strip()
    content = (
        f"started: {description} child={child_run_id}"
        if description
        else f"started: child={child_run_id}"
    )
    return EventLogEntry("subagent", content, visible=True)


def subagent_finished_entry(event: dict[str, Any]) -> EventLogEntry:
    child_run_id = str(event.get("run_id", ""))
    finish_status = str(event.get("status", "unknown"))
    return EventLogEntry(
        "subagent",
        f"finished: {child_run_id} status={finish_status}",
        visible=True,
    )


def context_compacted_entry(event: dict[str, Any]) -> EventLogEntry:
    original = int(event.get("original_tokens", 0))
    summary = int(event.get("summary_tokens", 0))
    saved = max(0, original - summary)
    return EventLogEntry(
        "system",
        f"context compacted original={original} summary={summary} saved~={saved}",
        visible=True,
    )


def context_compaction_failed_entry(event: dict[str, Any]) -> EventLogEntry:
    reason = str(event.get("reason", "unknown"))
    return EventLogEntry(
        "system",
        f"context compaction failed: {reason}",
        visible=True,
    )


def task_event_entry(event: dict[str, Any]) -> EventLogEntry:
    event_type = str(event.get("type", "task.unknown"))
    task_id = str(event.get("task_id", "?"))
    subject = str(event.get("subject", "")).strip()
    status = str(event.get("status", "")).strip()
    label = f"#{task_id}"
    if subject:
        label = f"{label} {subject}"

    if event_type == "task.assigned":
        assigned = str(event.get("assigned_run_id", "")).strip()
        return EventLogEntry(
            "task",
            f"assigned {label} -> {assigned}" if assigned else f"assigned {label}",
            visible=True,
        )
    if event_type == "task.completed":
        completed_by = str(event.get("completed_by_run_id", "")).strip()
        return EventLogEntry(
            "task",
            (
                f"completed {label} by {completed_by}"
                if completed_by
                else f"completed {label}"
            ),
            visible=True,
        )
    if event_type == "task.failed":
        failed_by = str(event.get("failed_by_run_id", "")).strip()
        reason = str(event.get("failure_reason", "")).strip()
        parts = [f"failed {label}"]
        if failed_by:
            parts.append(f"by {failed_by}")
        if reason:
            parts.append(f"reason={reason}")
        return EventLogEntry("task", " ".join(parts), visible=True)
    if event_type in {"task.created", "task.updated", "task.status_changed"}:
        status_part = f" status={status}" if status else ""
        verb = event_type.removeprefix("task.").replace("_", " ")
        return EventLogEntry("task", f"{verb} {label}{status_part}", visible=True)
    return unknown_event_entry(event_type)


def scheduler_event_entry(event: dict[str, Any]) -> EventLogEntry:
    event_type = str(event.get("type", "scheduler.unknown"))
    plan_id = str(event.get("plan_id", "")).strip()
    plan = f" plan={plan_id}" if plan_id else ""

    if event_type == "scheduler.plan.generated":
        ready = _len_value(event.get("ready_task_ids", []))
        dispatchable = _len_value(event.get("dispatchable_task_ids", []))
        skipped_count = _len_value(event.get("skipped_task_ids", []))
        replan = bool(event.get("should_replan", False))
        review = bool(event.get("requires_human_review", False))
        diagnostics_count = int(event.get("diagnostics_count", 0))
        return EventLogEntry(
            "scheduler",
            (
                f"plan{plan} ready={ready} dispatchable={dispatchable} skipped={skipped_count} "
                f"replan={replan} review={review} diagnostics={diagnostics_count}"
            ),
            visible=True,
        )
    if event_type == "scheduler.diagnosis.reported":
        diagnostics = event.get("diagnostics", []) or []
        text = "; ".join(str(item) for item in diagnostics) or "no diagnostics"
        return EventLogEntry("scheduler", f"diagnosis{plan}: {text}", visible=True)
    if event_type == "scheduler.dispatch.skipped":
        skipped_items = event.get("skipped", []) or []
        return EventLogEntry(
            "scheduler",
            f"dispatch skipped{plan} count={_len_value(skipped_items)}",
            visible=True,
        )
    return unknown_event_entry(event_type)


def unknown_event_entry(event_type: object) -> EventLogEntry:
    if isinstance(event_type, str) and event_type.startswith("task."):
        return EventLogEntry("task", event_type, visible=True)
    if isinstance(event_type, str) and event_type.startswith("scheduler."):
        return EventLogEntry("scheduler", event_type, visible=True)
    return EventLogEntry("system", str(event_type))


def _len_value(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    return 0
