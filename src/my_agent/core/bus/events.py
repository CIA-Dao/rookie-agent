from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator

from my_agent.core.command.evidence import CommandEvidence


# S0 保留
class CoreStartedEvent(BaseModel):
    type: Literal["core.started"] = "core.started"
    listen_addr: str
    version: str


# S1 新增：run 级事件
class RunStartedEvent(BaseModel):
    type: Literal["run.started"] = "run.started"
    run_id: str
    goal: str
    ts: str


class RunFinishedEvent(BaseModel):
    type: Literal["run.finished"] = "run.finished"
    run_id: str
    status: str
    reason: str | None = None
    steps: int
    delivery: dict[str, Any] | None = None
    ts: str


# S1 新增：step 级事件
class StepStartedEvent(BaseModel):
    type: Literal["step.started"] = "step.started"
    run_id: str
    step: int
    ts: str


class StepFinishedEvent(BaseModel):
    type: Literal["step.finished"] = "step.finished"
    run_id: str
    step: int
    ts: str


# S1 新增：工具调用事件
class ToolCallStartedEvent(BaseModel):
    type: Literal["tool.call_started"] = "tool.call_started"
    run_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]
    evidence: CommandEvidence | None = None
    ts: str


class ToolCallFinishedEvent(BaseModel):
    type: Literal["tool.call_finished"] = "tool.call_finished"
    run_id: str
    tool_use_id: str
    tool_name: str
    elapsed_ms: int
    evidence: CommandEvidence | None = None
    ts: str


class ToolCallFailedEvent(BaseModel):
    type: Literal["tool.call_failed"] = "tool.call_failed"
    run_id: str
    tool_use_id: str
    tool_name: str
    error_type: str
    error_message: str
    elapsed_ms: int
    evidence: CommandEvidence | None = None
    ts: str


# S1 新增：LLM 事件
class LlmModelSelectedEvent(BaseModel):
    type: Literal["llm.model_selected"] = "llm.model_selected"
    run_id: str
    model: str
    strategy: str
    ts: str


class LlmTokenEvent(BaseModel):
    type: Literal["llm.token"] = "llm.token"
    run_id: str
    token: str
    ts: str


class LlmUsageEvent(BaseModel):
    type: Literal["llm.usage"] = "llm.usage"
    run_id: str
    input_tokens: int
    output_tokens: int
    context_pct: float = 0.0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    ts: str


class SessionCreatedEvent(BaseModel):
    type: Literal["session.created"] = "session.created"
    session_id: str
    mode: str
    ts: str


class SessionMessageReceivedEvent(BaseModel):
    type: Literal["session.message_received"] = "session.message_received"
    session_id: str
    content: str
    ts: str


class SessionWaitingForInputEvent(BaseModel):
    type: Literal["session.waiting_for_input"] = "session.waiting_for_input"
    session_id: str
    last_run_id: str
    ts: str


class SessionResumedEvent(BaseModel):
    type: Literal["session.resumed"] = "session.resumed"
    session_id: str
    ts: str


class SessionClosedEvent(BaseModel):
    type: Literal["session.closed"] = "session.closed"
    session_id: str
    ts: str


class ContextCompactedEvent(BaseModel):
    type: Literal["context.compacted"] = "context.compacted"
    session_id: str
    run_id: str
    original_tokens: int
    summary_tokens: int
    ts: str


class ContextCompactionFailedEvent(BaseModel):
    type: Literal["context.compaction_failed"] = "context.compaction_failed"
    session_id: str
    run_id: str
    reason: str
    ts: str


class PermissionRequestedEvent(BaseModel):
    type: Literal["permission.requested"] = "permission.requested"
    run_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]
    session_id: str
    ts: str


class PermissionGrantedEvent(BaseModel):
    type: Literal["permission.granted"] = "permission.granted"
    run_id: str
    tool_use_id: str
    decision: str
    ts: str


class PermissionDeniedEvent(BaseModel):
    type: Literal["permission.denied"] = "permission.denied"
    run_id: str
    tool_use_id: str
    decision: str
    ts: str


class SkillInvokedEvent(BaseModel):
    type: Literal["skill.invoked"] = "skill.invoked"
    skill_name: str
    arguments: str
    run_id: str
    session_id: str
    ts: str


# P6: structured diagnostics for third-party Skill allowed_tools compatibility.
# Emitted only when there is something to report (alias hits or unresolved tools).
class SkillToolCompatibilityEvent(BaseModel):
    type: Literal["skill.tool_compatibility"] = "skill.tool_compatibility"
    skill_name: str
    run_id: str
    session_id: str
    resolved_tools: list[str]
    aliases: list[dict[str, str]]
    unresolved_tools: list[str]
    ts: str


class SubagentStartedEvent(BaseModel):
    type: Literal["subagent.started"] = "subagent.started"
    run_id: str
    parent_run_id: str
    description: str
    root_run_id: str | None = None
    depth: int | None = None
    subagent_type: str = ""
    ts: str


class SubagentFinishedEvent(BaseModel):
    type: Literal["subagent.finished"] = "subagent.finished"
    run_id: str
    parent_run_id: str
    status: str
    root_run_id: str | None = None
    depth: int | None = None
    subagent_type: str = ""
    ts: str


class TaskCreatedEvent(BaseModel):
    type: Literal["task.created"] = "task.created"
    run_id: str
    session_id: str = ""
    task_id: int
    subject: str
    status: str
    ts: str


class TaskUpdatedEvent(BaseModel):
    type: Literal["task.updated"] = "task.updated"
    run_id: str
    session_id: str = ""
    task_id: int
    subject: str
    status: str
    ts: str


class TaskStatusChangedEvent(BaseModel):
    type: Literal["task.status_changed"] = "task.status_changed"
    run_id: str
    session_id: str = ""
    task_id: int
    subject: str
    previous_status: str
    status: str
    ts: str


class TaskAssignedEvent(BaseModel):
    type: Literal["task.assigned"] = "task.assigned"
    run_id: str
    session_id: str = ""
    task_id: int
    subject: str
    status: str
    assigned_run_id: str
    ts: str


class TaskCompletedEvent(BaseModel):
    type: Literal["task.completed"] = "task.completed"
    run_id: str
    session_id: str = ""
    task_id: int
    subject: str
    status: str
    completed_by_run_id: str
    assigned_run_id: str = ""
    ts: str


class TaskFailedEvent(BaseModel):
    type: Literal["task.failed"] = "task.failed"
    run_id: str
    session_id: str = ""
    task_id: int
    subject: str
    status: str
    failed_by_run_id: str
    assigned_run_id: str = ""
    failure_reason: str = ""
    ts: str


class SchedulerPlanGeneratedEvent(BaseModel):
    type: Literal["scheduler.plan.generated"] = "scheduler.plan.generated"
    run_id: str
    session_id: str = ""
    plan_id: str
    parent_run_id: str
    root_run_id: str
    ready_task_ids: list[int]
    dispatchable_task_ids: list[int]
    skipped_task_ids: list[int]
    should_replan: bool
    requires_human_review: bool
    diagnostics_count: int
    ts: str


class SchedulerDiagnosisReportedEvent(BaseModel):
    type: Literal["scheduler.diagnosis.reported"] = "scheduler.diagnosis.reported"
    run_id: str
    session_id: str = ""
    plan_id: str
    diagnostics: list[str]
    should_replan: bool
    requires_human_review: bool
    ts: str


class SchedulerDispatchSkippedEvent(BaseModel):
    type: Literal["scheduler.dispatch.skipped"] = "scheduler.dispatch.skipped"
    run_id: str
    session_id: str = ""
    plan_id: str
    skipped: list[dict[str, object]]
    ts: str


Event = Annotated[
    CoreStartedEvent
    | RunStartedEvent
    | RunFinishedEvent
    | StepStartedEvent
    | StepFinishedEvent
    | ToolCallStartedEvent
    | ToolCallFinishedEvent
    | ToolCallFailedEvent
    | LlmModelSelectedEvent
    | LlmTokenEvent
    | LlmUsageEvent
    | SessionCreatedEvent
    | SessionMessageReceivedEvent
    | SessionWaitingForInputEvent
    | SessionResumedEvent
    | SessionClosedEvent
    | PermissionRequestedEvent
    | PermissionGrantedEvent
    | PermissionDeniedEvent
    | ContextCompactedEvent
    | ContextCompactionFailedEvent
    | SkillInvokedEvent
    | SkillToolCompatibilityEvent
    | SubagentStartedEvent
    | SubagentFinishedEvent
    | TaskCreatedEvent
    | TaskUpdatedEvent
    | TaskStatusChangedEvent
    | TaskAssignedEvent
    | TaskCompletedEvent
    | TaskFailedEvent
    | SchedulerPlanGeneratedEvent
    | SchedulerDiagnosisReportedEvent
    | SchedulerDispatchSkippedEvent,
    Discriminator("type"),
]
