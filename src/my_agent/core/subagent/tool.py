from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from my_agent.core.agents.loader import AgentProfile, AgentProfileLoader
from my_agent.core.bus.events import SubagentFinishedEvent, SubagentStartedEvent
from my_agent.core.context import ExecutionContext
from my_agent.core.events.bus import EventBus
from my_agent.core.events.writer import EventWriter
from my_agent.core.loop import AgentLoop
from my_agent.core.runs import new_run_id
from my_agent.core.subagent.registry import BackgroundTaskRegistry, SubagentLimits, SubagentStatus
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.builtin import (
    BashTool,
    FileMetadataTool,
    FileSearchTool,
    ListDirTool,
    ProjectBuildTool,
    ReadFileTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileTool,
)
from my_agent.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from my_agent.core.llm.base import LLMProvider
    from my_agent.core.permissions.manager import PermissionManager


def _final_answer(context: ExecutionContext) -> str:
    for message in reversed(context.messages):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            return str(content)
    return ""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SpawnAgentParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str
    prompt: str
    run_in_background: bool = False
    subagent_type: str = ""


class AgentResultParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str


class AgentCancelParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    cascade: bool = True


class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        "Delegate a self-contained subtask to an isolated sub-agent. "
        "Use this when independent analysis, review, planning, research, or parallel "
        "investigation would improve the answer. Do not use it for simple direct tasks "
        "that can be solved with one ordinary tool call."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short task description shown in progress displays.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Complete task prompt. Include all context the sub-agent needs."
                ),
            },
            "run_in_background": {
                "type": "boolean",
                "description": (
                    "Use false when the parent agent needs the result before continuing. "
                    "Use true when launching independent subtasks that can run in parallel; "
            "then call agent_result with the returned run_id."
                ),
            },
            "subagent_type": {
                "type": "string",
                "description": (
                    "Optional agent profile name, such as planner, executor, or reviewer."
                ),
            },
        },
        "required": ["description", "prompt"],
    }
    params_model = SpawnAgentParams

    def __init__(
        self,
        provider: LLMProvider,
        parent_bus: EventBus,
        parent_run_id: str,
        permission_manager: PermissionManager | None,
        max_steps: int,
        task_registry: BackgroundTaskRegistry,
        runs_dir: Path,
        session_id: str,
        workspace_root: str = "",
        depth: int = 0,
        root_run_id: str | None = None,
        limits: SubagentLimits | None = None,
    ) -> None:
        self._provider = provider
        self._parent_bus = parent_bus
        self._parent_run_id = parent_run_id
        self._root_run_id = root_run_id or parent_run_id
        self._permission_manager = permission_manager
        self._max_steps = max_steps
        self._task_registry = task_registry
        self._runs_dir = runs_dir
        self._session_id = session_id
        self._workspace_root = workspace_root
        self._depth = depth
        self._limits = limits or SubagentLimits()

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = SpawnAgentParams.model_validate(params)

        limit_error = self._check_limits(p.run_in_background)
        if limit_error is not None:
            return limit_error

        child_depth = self._depth + 1
        created_at = _now()
        child_run_id = new_run_id()
        profile: AgentProfile | None = None
        if p.subagent_type:
            profile = AgentProfileLoader(self._workspace_root).load(p.subagent_type)
        child_context = ExecutionContext(
            run_id=child_run_id,
            goal=p.prompt,
            max_steps=self._max_steps,
            system_prompt_override=profile.system_prompt if profile else None,
        )

        child_bus = EventBus()

        async def bridge(event: BaseModel) -> None:
            await self._parent_bus.publish(event)

        child_bus.subscribe(bridge)

        child_registry = self._build_child_registry(child_bus, child_run_id, profile)

        child_loop = AgentLoop(
            self._provider,
            child_registry,
            child_bus,
            permission_manager=self._permission_manager,
            session_id=self._session_id,
            workspace_root=str(self._workspace_root),
        )

        await self._parent_bus.publish(
            SubagentStartedEvent(
                run_id=child_run_id,
                parent_run_id=self._parent_run_id,
                description=p.description,
                root_run_id=self._root_run_id,
                depth=child_depth,
                subagent_type=p.subagent_type,
                ts=_now(),
            )
        )

        if p.run_in_background:
            task = asyncio.create_task(
                self._run_child(
                    child_context=child_context,
                    child_loop=child_loop,
                    child_bus=child_bus,
                    depth=child_depth,
                    subagent_type=p.subagent_type,
                    timeout_seconds=self._limits.background_timeout_seconds,
                )
            )
            self._task_registry.register(
                child_run_id,
                task,
                child_context,
                parent_run_id=self._parent_run_id,
                root_run_id=self._root_run_id,
                session_id=self._session_id,
                depth=child_depth,
                description=p.description,
                subagent_type=p.subagent_type,
                run_in_background=True,
                created_at=created_at,
            )
            self._task_registry.prune_completed(self._limits.max_completed_records)
            return ToolResult(
                content=f"Subagent started in background: run_id={child_run_id}"
            )

        self._task_registry.register(
            child_run_id,
            task=None,
            context=child_context,
            parent_run_id=self._parent_run_id,
            root_run_id=self._root_run_id,
            session_id=self._session_id,
            depth=child_depth,
            description=p.description,
            subagent_type=p.subagent_type,
            run_in_background=False,
            created_at=created_at,
        )
        await self._run_child(
            child_context=child_context,
            child_loop=child_loop,
            child_bus=child_bus,
            depth=child_depth,
            subagent_type=p.subagent_type,
            timeout_seconds=None,
        )

        answer = _final_answer(child_context)
        if child_context.status == "success":
            return ToolResult(content=answer or "Subagent completed with no text output.")

        return ToolResult(
            content=answer
            or f"Subagent failed: {child_context.reason or child_context.status}",
            is_error=True,
            error_type="runtime_error",
        )

    def _build_child_registry(
        self,
        child_bus: EventBus,
        child_run_id: str,
        profile: AgentProfile | None,
    ) -> ToolRegistry:
        allowed = set(profile.allowed_tools) if profile and profile.allowed_tools else None

        def ok(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        child_task_manager = TaskManager(self._runs_dir / child_run_id / ".tasks")

        for tool in [
            ReadFileTool(self._workspace_root),
            ListDirTool(self._workspace_root),
            FileMetadataTool(self._workspace_root),
            FileSearchTool(self._workspace_root),
            ProjectBuildTool(self._workspace_root),
            WriteFileTool(self._workspace_root),
            BashTool(self._workspace_root),
            TaskCreateTool(
                child_task_manager,
                bus=child_bus,
                run_id=child_run_id,
                session_id=self._session_id,
            ),
            TaskUpdateTool(
                child_task_manager,
                bus=child_bus,
                run_id=child_run_id,
                session_id=self._session_id,
            ),
            TaskListTool(child_task_manager),
            TaskGetTool(child_task_manager),
        ]:
            if ok(tool.name):
                registry.register(tool)

        if self._depth < 1:
            nested = SpawnAgentTool(
                provider=self._provider,
                parent_bus=child_bus,
                parent_run_id=child_run_id,
                permission_manager=self._permission_manager,
                max_steps=self._max_steps,
                task_registry=self._task_registry,
                runs_dir=self._runs_dir,
                session_id=self._session_id,
                workspace_root=self._workspace_root,
                depth=self._depth + 1,
                root_run_id=self._root_run_id,
                limits=self._limits,
            )
            if ok("spawn_agent"):
                registry.register(nested)
            if ok("agent_result"):
                registry.register(AgentResultTool(self._task_registry))
            if ok("agent_cancel"):
                registry.register(AgentCancelTool(self._task_registry))

        return registry

    def _check_limits(self, run_in_background: bool) -> ToolResult | None:
        limits = self._limits
        if self._depth >= limits.max_depth:
            return ToolResult(
                content=(
                    f"Subagent limit max_depth reached ({limits.max_depth}); "
                    "cannot spawn further subagents."
                ),
                is_error=True,
                error_type="runtime_error",
            )

        child_depth = self._depth + 1

        if child_depth == 1:
            child_count = self._task_registry.count_direct_children(self._root_run_id)
            if child_count >= limits.max_children_per_root:
                return ToolResult(
                    content=(
                        "Subagent limit max_children_per_root reached "
                        f"({limits.max_children_per_root}); cannot spawn more child agents."
                    ),
                    is_error=True,
                    error_type="runtime_error",
                )

        if child_depth == 2:
            grandchild_count = self._task_registry.count_direct_children(self._parent_run_id)
            if grandchild_count >= limits.max_grandchildren_per_child:
                return ToolResult(
                    content=(
                        "Subagent limit max_grandchildren_per_child reached "
                        f"({limits.max_grandchildren_per_child}); "
                        "cannot spawn more grandchild agents."
                    ),
                    is_error=True,
                    error_type="runtime_error",
                )

        descendant_count = self._task_registry.count_descendants(self._root_run_id)
        if descendant_count >= limits.max_total_descendants_per_root:
            return ToolResult(
                content=(
                    "Subagent limit max_total_descendants_per_root reached "
                    f"({limits.max_total_descendants_per_root}); "
                    "cannot spawn more descendants."
                ),
                is_error=True,
                error_type="runtime_error",
            )

        if run_in_background:
            running = self._task_registry.count_running_background(self._session_id)
            if running >= limits.max_concurrent_background_subagents_per_session:
                return ToolResult(
                    content=(
                        "Subagent limit max_concurrent_background_subagents_per_session "
                        f"reached ({limits.max_concurrent_background_subagents_per_session}); "
                        "cannot start more background subagents."
                    ),
                    is_error=True,
                    error_type="runtime_error",
                )

        return None

    async def _run_child(
        self,
        *,
        child_context: ExecutionContext,
        child_loop: AgentLoop,
        child_bus: EventBus,
        depth: int,
        subagent_type: str,
        timeout_seconds: float | None,
    ) -> None:
        child_run_path = self._runs_dir / child_context.run_id
        child_run_path.mkdir(parents=True, exist_ok=True)

        try:
            async with EventWriter(child_run_path / "events.jsonl") as writer:
                writer.subscribe(child_bus)
                if timeout_seconds is None:
                    await child_loop.run(child_context)
                else:
                    try:
                        await asyncio.wait_for(
                            child_loop.run(child_context),
                            timeout=timeout_seconds,
                        )
                    except TimeoutError:
                        child_context.mark_failed("timed_out")
        except asyncio.CancelledError:
            if not child_context.is_done():
                child_context.mark_failed("cancelled")
            raise
        except Exception as exc:
            if not child_context.is_done():
                child_context.mark_failed(f"subagent_crashed: {exc}")
            raise
        finally:
            await self._parent_bus.publish(
                SubagentFinishedEvent(
                    run_id=child_context.run_id,
                    parent_run_id=self._parent_run_id,
                    status=child_context.status,
                    root_run_id=self._root_run_id,
                    depth=depth,
                    subagent_type=subagent_type,
                    ts=_now(),
                )
            )
            self._task_registry.mark(
                child_context.run_id,
                _record_status(child_context),
                completed_at=_now(),
                reason=child_context.reason or "",
            )


class AgentResultTool(BaseTool):
    name = "agent_result"
    description = (
        "Get the status and final answer of a background sub-agent using the run_id "
        "returned by spawn_agent. Use this before relying on background work."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The child run_id returned by spawn_agent.",
            }
        },
        "required": ["run_id"],
    }
    params_model = AgentResultParams

    def __init__(self, task_registry: BackgroundTaskRegistry) -> None:
        self._task_registry = task_registry

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = AgentResultParams.model_validate(params)
        record = self._task_registry.get_record(p.run_id)
        if record is None or record.task is None:
            return ToolResult(
                content=f"No background subagent found for run_id={p.run_id}",
                is_error=True,
                error_type="not_found",
            )

        task = record.task
        context = record.context

        if record.status == "running" and not task.done():
            return ToolResult(
                content=f"Subagent is still running: run_id={p.run_id}"
            )

        if record.status == "cancelled" or task.cancelled():
            return ToolResult(
                content=f"Subagent was cancelled: run_id={p.run_id}",
                is_error=True,
                error_type="cancelled",
            )

        if record.status == "timed_out":
            return ToolResult(
                content=f"Subagent timed out: run_id={p.run_id}",
                is_error=True,
                error_type="timeout",
            )

        error = None if task.cancelled() else task.exception()
        if error is not None:
            return ToolResult(
                content=f"Subagent crashed: {error}",
                is_error=True,
                error_type="runtime_error",
            )

        answer = _final_answer(context)
        if record.status == "success" or context.status == "success":
            return ToolResult(content=answer or "Subagent completed with no text output.")

        return ToolResult(
            content=answer or f"Subagent failed: {context.reason or context.status}",
            is_error=True,
            error_type="runtime_error",
        )


class AgentCancelTool(BaseTool):
    name = "agent_cancel"
    description = (
        "Cancel a running background sub-agent by run_id. By default this also "
        "cancels background descendants of that run."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The background sub-agent run_id to cancel.",
            },
            "cascade": {
                "type": "boolean",
                "description": "Whether to also cancel descendant background sub-agents.",
            },
        },
        "required": ["run_id"],
    }
    params_model = AgentCancelParams

    def __init__(self, task_registry: BackgroundTaskRegistry) -> None:
        self._task_registry = task_registry

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = AgentCancelParams.model_validate(params)
        record = self._task_registry.get_record(p.run_id)
        if record is None or record.task is None:
            return ToolResult(
                content=f"No background subagent found for run_id={p.run_id}",
                is_error=True,
                error_type="not_found",
            )

        if record.status != "running" or record.task.done():
            return ToolResult(
                content=f"Subagent is not running: run_id={p.run_id} status={record.status}",
                is_error=True,
                error_type="runtime_error",
            )

        cancelled = (
            self._task_registry.cancel_tree(p.run_id)
            if p.cascade
            else int(self._task_registry.cancel(p.run_id))
        )
        if cancelled <= 0:
            return ToolResult(
                content=f"Subagent could not be cancelled: run_id={p.run_id}",
                is_error=True,
                error_type="runtime_error",
            )

        return ToolResult(content=f"Cancelled subagent(s): count={cancelled}")


def _record_status(context: ExecutionContext) -> SubagentStatus:
    if context.status == "success":
        return "success"
    if context.reason == "timed_out":
        return "timed_out"
    if context.reason == "cancelled":
        return "cancelled"
    return "failed"
