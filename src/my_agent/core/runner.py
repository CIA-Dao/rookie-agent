from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_agent.core.bus.events import RunFinishedEvent, RunStartedEvent, TaskFailedEvent
from my_agent.core.compact.compactor import Compactor
from my_agent.core.config import Config
from my_agent.core.context import ExecutionContext
from my_agent.core.delegation.policy import looks_complex_goal
from my_agent.core.delivery import (
    DeliveryVerification,
    aggregate_delivery_result,
    verify_project_delivery,
)
from my_agent.core.events.bus import EventBus
from my_agent.core.events.writer import EventWriter
from my_agent.core.llm.base import LLMProvider
from my_agent.core.llm.provider import DeepSeekProvider
from my_agent.core.loop import AgentLoop
from my_agent.core.mcp.server import McpServerManager
from my_agent.core.memory.loader import load_context_file
from my_agent.core.permissions.manager import PermissionManager
from my_agent.core.runs import new_run_id
from my_agent.core.session.model import Session
from my_agent.core.session.store import SessionStore
from my_agent.core.subagent.registry import BackgroundTaskRegistry
from my_agent.core.subagent.tool import AgentCancelTool, AgentResultTool, SpawnAgentTool
from my_agent.core.task.classifier import classify_task
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.base import BaseTool
from my_agent.core.tools.builtin import (
    BashTool,
    ChunkedWriteStore,
    CollectDispatchResultsTool,
    DelegationPolicyTool,
    DispatchPlanTool,
    FileMetadataTool,
    FileSearchTool,
    ListDirTool,
    NoteSaveTool,
    OrchestrateTasksTool,
    OrchestrateUntilIdleTool,
    OrchestrationSummaryTool,
    ProjectBuildTool,
    ReadFileRangeTool,
    ReadFileTool,
    SchedulePlanTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileBeginTool,
    WriteFileChunkTool,
    WriteFileCommitTool,
    WriteFileTool,
)
from my_agent.core.tools.catalog import BUILTIN_TOOL_NAMES
from my_agent.core.tools.registry import ToolRegistry
from my_agent.core.tools.workspace import workspace_root_or_cwd
from my_agent.core.trace.provider import TracingProvider
from my_agent.core.trace.writer import TraceWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _get_final_answer(context: ExecutionContext) -> str | None:
    """从消息历史里取出 LLM 的最后一条文本回复"""
    for msg in reversed(context.messages):
        if msg["role"] == "assistant" and msg["content"]:
            content = msg["content"]
            if isinstance(content, str):
                return content
    return None


def _workspace_manifest(root: Path) -> dict[str, dict[str, Any]]:
    excluded = {".git", ".my-agent", "node_modules", "runs"}
    manifest: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return manifest
    for path in root.rglob("*"):
        if not path.is_file() or excluded.intersection(path.parts):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        manifest[path.relative_to(root).as_posix()] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
    return manifest


def _write_artifact_manifest(
    run_path: Path,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> None:
    before_paths = set(before)
    after_paths = set(after)
    created = sorted(after_paths - before_paths)
    deleted = sorted(before_paths - after_paths)
    modified = sorted(
        path for path in before_paths & after_paths if before[path] != after[path]
    )
    payload: dict[str, Any] = {
        "created": created,
        "modified": modified,
        "deleted": deleted,
        "outside_workspace": [],
    }
    (run_path / "artifact-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None
    delivery: dict[str, object] | None = None


class AgentRunner:
    def __init__(
        self,
        config: Config,
        *,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        runs_dir: Path | None = None,
        trace: TraceWriter | None = None,
        permission_manager: PermissionManager | None = None,
        mcp_manager: McpServerManager | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._provider = provider
        self._runs_dir = runs_dir or Path(config.runs_dir).expanduser()
        self._trace = trace
        self._permission_manager = permission_manager
        self._subagent_tasks = BackgroundTaskRegistry()
        self._mcp_manager = mcp_manager

    def available_tool_names(self) -> list[str]:
        names = list(BUILTIN_TOOL_NAMES)
        if self._mcp_manager is not None:
            seen = set(names)
            for tool in self._mcp_manager.get_tools():
                if tool.name not in seen:
                    names.append(tool.name)
                    seen.add(tool.name)
        return names

    def _build_registry(
        self,
        task_manager: TaskManager,
        *,
        session: Session | None = None,
        store: SessionStore | None = None,
        run_id: str | None = None,
        workspace_root: str | Path | None = None,
        tool_whitelist: list[str] | None = None,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
    ) -> ToolRegistry:
        allowed = set(tool_whitelist) if tool_whitelist is not None else None
        run_bus = bus or self._bus or EventBus()

        registry = ToolRegistry()
        self._register_tools(
            registry,
            self._base_tools(
                task_manager,
                session=session,
                run_id=run_id,
                workspace_root=workspace_root,
                bus=run_bus,
            ),
            allowed,
        )
        self._register_note_tool(
            registry,
            session=session,
            store=store,
            run_id=run_id,
            allowed=allowed,
        )

        run_provider = provider or self._provider
        if run_id is not None and run_provider is not None:
            self._register_dispatch_tools(
                registry,
                task_manager,
                provider=run_provider,
                session=session,
                run_id=run_id,
                workspace_root=workspace_root,
                bus=run_bus,
                allowed=allowed,
            )
            self._register_tools(
                registry,
                self._subagent_tools(
                    provider=run_provider,
                    session=session,
                    run_id=run_id,
                    workspace_root=workspace_root,
                    bus=run_bus,
                ),
                allowed,
            )

        self._register_mcp_tools(registry, allowed)

        return registry

    @staticmethod
    def _tool_allowed(name: str, allowed: set[str] | None) -> bool:
        return allowed is None or name in allowed

    def _register_tools(
        self,
        registry: ToolRegistry,
        tools: list[BaseTool],
        allowed: set[str] | None,
    ) -> None:
        for tool in tools:
            if self._tool_allowed(tool.name, allowed):
                registry.register(tool)

    def _base_tools(
        self,
        task_manager: TaskManager,
        *,
        session: Session | None,
        run_id: str | None,
        workspace_root: str | Path | None,
        bus: EventBus,
    ) -> list[BaseTool]:
        session_id = session.id if session is not None else ""
        effective_run_id = run_id or ""
        chunked_store = ChunkedWriteStore(workspace_root)
        return [
            ReadFileTool(workspace_root),
            ReadFileRangeTool(workspace_root),
            SchedulePlanTool(
                task_manager,
                self._subagent_tasks,
                parent_run_id=effective_run_id,
                root_run_id=effective_run_id,
                session_id=session_id,
                bus=bus,
            ),
            TaskCreateTool(
                task_manager,
                bus=bus,
                run_id=effective_run_id,
                session_id=session_id,
            ),
            TaskUpdateTool(
                task_manager,
                bus=bus,
                run_id=effective_run_id,
                session_id=session_id,
            ),
            TaskListTool(task_manager),
            TaskGetTool(task_manager),
            ListDirTool(workspace_root),
            FileMetadataTool(workspace_root),
            FileSearchTool(workspace_root),
            ProjectBuildTool(workspace_root),
            WriteFileTool(workspace_root),
            WriteFileBeginTool(chunked_store),
            WriteFileChunkTool(chunked_store),
            WriteFileCommitTool(chunked_store),
            BashTool(workspace_root),
            DelegationPolicyTool(
                task_manager,
                self._subagent_tasks,
                parent_run_id=effective_run_id,
                root_run_id=effective_run_id,
                session_id=session_id,
            ),
        ]

    def _register_note_tool(
        self,
        registry: ToolRegistry,
        *,
        session: Session | None,
        store: SessionStore | None,
        run_id: str | None,
        allowed: set[str] | None,
    ) -> None:
        if session is not None and store is not None and run_id is not None:
            note_tool = NoteSaveTool(store, session.id, run_id)
            if self._tool_allowed(note_tool.name, allowed):
                registry.register(note_tool)

    def _spawn_agent_tool(
        self,
        *,
        provider: LLMProvider,
        session: Session | None,
        run_id: str,
        workspace_root: str | Path | None,
        bus: EventBus,
    ) -> SpawnAgentTool:
        return SpawnAgentTool(
            provider=provider,
            parent_bus=bus,
            parent_run_id=run_id,
            permission_manager=self._permission_manager,
            max_steps=self._config.agent.max_steps,
            task_registry=self._subagent_tasks,
            runs_dir=self._runs_dir,
            session_id=session.id if session is not None else "",
            workspace_root=str(workspace_root or ""),
        )

    def _subagent_tools(
        self,
        *,
        provider: LLMProvider,
        session: Session | None,
        run_id: str,
        workspace_root: str | Path | None,
        bus: EventBus,
    ) -> list[BaseTool]:
        return [
            self._spawn_agent_tool(
                provider=provider,
                session=session,
                run_id=run_id,
                workspace_root=workspace_root,
                bus=bus,
            ),
            AgentResultTool(self._subagent_tasks),
            AgentCancelTool(self._subagent_tasks),
        ]

    def _register_dispatch_tools(
        self,
        registry: ToolRegistry,
        task_manager: TaskManager,
        *,
        provider: LLMProvider,
        session: Session | None,
        run_id: str,
        workspace_root: str | Path | None,
        bus: EventBus,
        allowed: set[str] | None,
    ) -> None:
        session_id = session.id if session is not None else ""
        if self._tool_allowed("dispatch_plan", allowed):
            registry.register(
                DispatchPlanTool(
                    task_manager,
                    self._subagent_tasks,
                    self._spawn_agent_tool(
                        provider=provider,
                        session=session,
                        run_id=run_id,
                        workspace_root=workspace_root,
                        bus=bus,
                    ),
                    parent_run_id=run_id,
                    root_run_id=run_id,
                    session_id=session_id,
                    bus=bus,
                )
            )
        if self._tool_allowed("collect_dispatch_results", allowed):
            registry.register(
                CollectDispatchResultsTool(
                    task_manager,
                    self._subagent_tasks,
                    bus=bus,
                    run_id=run_id,
                    session_id=session_id,
                )
            )
        if self._tool_allowed("orchestrate_tasks", allowed):
            registry.register(
                OrchestrateTasksTool(
                    task_manager,
                    self._subagent_tasks,
                    DispatchPlanTool(
                        task_manager,
                        self._subagent_tasks,
                        self._spawn_agent_tool(
                            provider=provider,
                            session=session,
                            run_id=run_id,
                            workspace_root=workspace_root,
                            bus=bus,
                        ),
                        parent_run_id=run_id,
                        root_run_id=run_id,
                        session_id=session_id,
                        bus=bus,
                    ),
                    parent_run_id=run_id,
                    root_run_id=run_id,
                    session_id=session_id,
                    bus=bus,
                )
            )
        if self._tool_allowed("orchestrate_until_idle", allowed):
            registry.register(
                OrchestrateUntilIdleTool(
                    OrchestrateTasksTool(
                        task_manager,
                        self._subagent_tasks,
                        DispatchPlanTool(
                            task_manager,
                            self._subagent_tasks,
                            self._spawn_agent_tool(
                                provider=provider,
                                session=session,
                                run_id=run_id,
                                workspace_root=workspace_root,
                                bus=bus,
                            ),
                            parent_run_id=run_id,
                            root_run_id=run_id,
                            session_id=session_id,
                            bus=bus,
                        ),
                        parent_run_id=run_id,
                        root_run_id=run_id,
                        session_id=session_id,
                        bus=bus,
                    )
                )
            )
        if self._tool_allowed("orchestration_summary", allowed):
            registry.register(
                OrchestrationSummaryTool(
                    task_manager,
                    self._subagent_tasks,
                    parent_run_id=run_id,
                    root_run_id=run_id,
                    session_id=session_id,
                )
            )

    def _register_mcp_tools(self, registry: ToolRegistry, allowed: set[str] | None) -> None:
        if self._mcp_manager is not None:
            for tool in self._mcp_manager.get_tools():
                if self._tool_allowed(tool.name, allowed):
                    registry.register(tool)

    async def run(self, goal: str, *, run_id: str | None = None) -> None:
        await self.run_and_capture(goal, run_id=run_id)

    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        tool_whitelist: list[str] | None = None,
        system_prompt_override: str | None = None,
    ) -> RunOutcome:
        run_id = run_id or new_run_id()
        workspace_root = session.workspace_root if session is not None else ""

        if session is not None and store is not None:
            run_path = store.runs_dir(session.id) / run_id
        else:
            run_path = self._runs_dir / run_id

        run_path.mkdir(parents=True, exist_ok=True)
        workspace_path = workspace_root_or_cwd(workspace_root)
        manifest_before = _workspace_manifest(workspace_path)
        task_manager = TaskManager(run_path / ".tasks")
        task_type = classify_task(goal)
        run_meta = {
            "run_id": run_id,
            "goal": goal,
            "task_type": task_type.value,
            "workspace_root": workspace_root,
            "created_at": _now(),
        }
        if session is not None and store is not None:
            store.write_run_meta(session.id, run_id, run_meta)
        else:
            (run_path / "meta.json").write_text(
                json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        bus = self._bus if self._bus is not None else EventBus()

        provider = self._provider or DeepSeekProvider(
            self._config.llm.default_model,
            trace=self._trace,
        )

        prefill_messages = (
            store.read_messages(session.id)
            if session is not None and store is not None
            else []
        )

        prefill_len = len(prefill_messages)

        session_notes = (
            store.read_notes(session.id)
            if session is not None and store is not None
            else ""
        )
        global_context = load_context_file(Path("~/.my-agent/context.md").expanduser())
        project_context = (
            load_context_file(Path(workspace_root) / ".my-agent" / "context.md")
            if workspace_root
            else ""
        )

        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=self._config.agent.max_steps,
            messages=prefill_messages,
            session_notes=session_notes,
            persist_from=prefill_len,
            global_context=global_context,
            project_context=project_context,
            system_prompt_override=system_prompt_override,
        )

        if self._trace is not None:
            provider = TracingProvider(
                provider, self._trace, include_payload=self._config.trace.include_llm_payload
            )

        registry = self._build_registry(
            task_manager,
            session=session,
            store=store,
            run_id=run_id,
            workspace_root=workspace_root,
            tool_whitelist=tool_whitelist,
            bus=bus,
            provider=provider,
        )
        context.runtime_guidance = await self._delegation_runtime_guidance(
            goal,
            registry,
        )

        compactor = (
            Compactor(
                bus,
                store.session_dir(session.id),
                session.id,
                store=store,
                task_type=task_type,
            )
            if self._config.compact.enabled and session is not None and store is not None
            else None
        )

        loop = AgentLoop(
            provider,
            registry,
            bus,
            permission_manager=self._permission_manager,
            session_id=session.id if session is not None else "",
            compactor=compactor,
            compact_token_threshold=self._config.compact.token_threshold,
            tool_result_limit=self._config.compact.tool_result_limit,
            tool_result_keep=self._config.compact.tool_result_keep,
            compact_context_ratio=self._config.compact.context_ratio,
            workspace_root=str(workspace_path),
        )

        async with EventWriter(run_path / "events.jsonl") as writer:
            writer.subscribe(bus)

            await bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))

            try:
                await loop.run(context)
            except asyncio.CancelledError:
                if not context.is_done():
                    context.mark_failed("cancelled")
                raise

            verification = DeliveryVerification(checked=False, passed=True)
            if context.status == "success":
                verification = await verify_project_delivery(workspace_path, goal)
                if verification.checked and not verification.passed:
                    context.mark_failed(
                        f"delivery_verification_failed: {verification.summary}"
                    )

            manifest_after = _workspace_manifest(workspace_path)
            _write_artifact_manifest(run_path, manifest_before, manifest_after)
            manifest: dict[str, object] = {
                "created": sorted(set(manifest_after) - set(manifest_before)),
                "modified": sorted(
                    path
                    for path in set(manifest_before) & set(manifest_after)
                    if manifest_before[path] != manifest_after[path]
                ),
            }
            delivery = aggregate_delivery_result(
                context_status=context.status,
                context_reason=context.reason,
                verification=verification,
                manifest=manifest,
            )
            (run_path / "delivery-result.json").write_text(
                json.dumps(delivery.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if context.status == "failed" and context.reason and task_manager.list_all():
                for task in task_manager.list_all():
                    if task.status in {"completed", "in_progress"}:
                        failed = task_manager.fail_with_run(
                            task.id,
                            run_id,
                            f"run delivery failed: {context.reason}",
                        )
                        await bus.publish(
                            TaskFailedEvent(
                                run_id=run_id,
                                session_id=session.id if session is not None else "",
                                task_id=failed.id,
                                subject=failed.subject,
                                status=failed.status,
                                failed_by_run_id=run_id,
                                assigned_run_id=failed.assigned_run_id,
                                failure_reason=failed.failure_reason,
                                ts=_now(),
                            )
                        )

            await bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status=context.status,
                    reason=context.reason,
                    steps=context.step,
                    delivery=delivery.as_dict(),
                    ts=_now(),
                )
            )
        final_msg = _get_final_answer(context)
        if context.status == "failed":
            final_msg = (
                "本次运行失败，未确认交付完成。"
                + (f" 原因：{context.reason}" if context.reason else "")
            )
            final_msg += (
                f" Delivery: files={delivery.write_status}, "
                f"build={delivery.build_status}, final={delivery.final_status}."
            )

        if (
            context.status == "success"
            and compactor is not None
            and self._config.compact.context_ratio > 0
            and context.max_context_pct >= self._config.compact.context_ratio
        ):
            await compactor.compact(context, provider)

        if session is not None and store is not None:
            store.append_messages(
                session.id,
                context.messages[context.persist_from:],
                run_id=run_id,
            )

        return RunOutcome(
            status=context.status,
            result=final_msg or "",
            reason=context.reason,
            delivery=delivery.as_dict(),
        )

    async def _delegation_runtime_guidance(
        self,
        goal: str,
        registry: ToolRegistry,
    ) -> str:
        if not looks_complex_goal(goal):
            return ""

        tool = registry.get("delegation_policy")
        if tool is None:
            return ""

        result = await tool.invoke({"goal": goal, "allow_auto_dispatch": True})
        if result.is_error:
            return ""

        return (
            "Automatic delegation preflight:\n"
            f"{result.content}\n\n"
            "Follow this preflight before choosing an execution path. If the decision is "
            "`create_task_graph`, create a concise task graph with task_create before "
            "broad implementation, inspect it with schedule_plan, then use "
            "orchestrate_tasks or orchestrate_until_idle only when the plan is safe. "
            "If the decision is `explicit_orchestration`, prefer bounded orchestration "
            "tools. If the decision is `manual_review`, do not dispatch sub-agents. "
            "If the decision is `direct`, proceed directly."
        )
