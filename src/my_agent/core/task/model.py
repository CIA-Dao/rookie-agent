from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

TaskStatus = Literal["pending", "in_progress", "completed", "failed"]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


@dataclass
class Task:
    id: int
    subject: str
    description: str
    status: TaskStatus
    blocked_by: list[int]
    created_at: str
    updated_at: str
    assigned_run_id: str = ""
    completed_by_run_id: str = ""
    failed_by_run_id: str = ""
    failure_reason: str = ""
    task_type: str = "general"
    priority: int = 0
    risk: str = "medium"
    suggested_agent_level: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    can_parallelize: bool = False
    requires_human_review: bool = False
    estimated_complexity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "blocked_by": self.blocked_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "assigned_run_id": self.assigned_run_id,
            "completed_by_run_id": self.completed_by_run_id,
            "failed_by_run_id": self.failed_by_run_id,
            "failure_reason": self.failure_reason,
            "task_type": self.task_type,
            "priority": self.priority,
            "risk": self.risk,
            "suggested_agent_level": self.suggested_agent_level,
            "required_capabilities": self.required_capabilities,
            "expected_outputs": self.expected_outputs,
            "acceptance_criteria": self.acceptance_criteria,
            "can_parallelize": self.can_parallelize,
            "requires_human_review": self.requires_human_review,
            "estimated_complexity": self.estimated_complexity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(
            id=int(data["id"]),
            subject=str(data["subject"]),
            description=str(data.get("description", "")),
            status=cast(TaskStatus, data.get("status", "pending")),
            blocked_by=[int(x) for x in data.get("blocked_by", [])],
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            assigned_run_id=str(data.get("assigned_run_id", "")),
            completed_by_run_id=str(data.get("completed_by_run_id", "")),
            failed_by_run_id=str(data.get("failed_by_run_id", "")),
            failure_reason=str(data.get("failure_reason", "")),
            task_type=str(data.get("task_type", "general")),
            priority=int(data.get("priority", 0)),
            risk=str(data.get("risk", "medium")),
            suggested_agent_level=str(data.get("suggested_agent_level", "")),
            required_capabilities=_string_list(data.get("required_capabilities", [])),
            expected_outputs=_string_list(data.get("expected_outputs", [])),
            acceptance_criteria=_string_list(data.get("acceptance_criteria", [])),
            can_parallelize=bool(data.get("can_parallelize", False)),
            requires_human_review=bool(data.get("requires_human_review", False)),
            estimated_complexity=str(data.get("estimated_complexity", "")),
        )
