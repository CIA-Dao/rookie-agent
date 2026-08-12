from __future__ import annotations

import json
from pathlib import Path

from my_agent.core.subagent.registry import BackgroundTaskRegistry
from my_agent.core.task.manager import TaskManager
from my_agent.core.tools.builtin.delegation_policy import DelegationPolicyTool


def _tool(manager: TaskManager) -> DelegationPolicyTool:
    return DelegationPolicyTool(
        manager,
        BackgroundTaskRegistry(),
        parent_run_id="root-run",
        root_run_id="root-run",
        session_id="sess-1",
    )


async def test_delegation_policy_recommends_direct_for_simple_empty_goal(
    tmp_path: Path,
) -> None:
    result = await _tool(TaskManager(tmp_path / "tasks")).invoke({"goal": "fix typo"})
    data = json.loads(result.content)

    assert data["decision"] == "direct"
    assert data["auto_delegation_enabled"] is False
    assert data["safe_to_auto_dispatch"] is False
    assert data["recommended_tools"] == []


async def test_delegation_policy_recommends_task_graph_for_complex_empty_goal(
    tmp_path: Path,
) -> None:
    result = await _tool(TaskManager(tmp_path / "tasks")).invoke(
        {
            "goal": (
                "First inspect the architecture, then split the work, and after "
                "that implement the scheduler changes."
            )
        }
    )
    data = json.loads(result.content)

    assert data["decision"] == "create_task_graph"
    assert "task_create" in data["recommended_tools"]
    assert data["safe_to_auto_dispatch"] is False


async def test_delegation_policy_requires_explicit_orchestration_for_dispatchable_graph(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create("parallel work", can_parallelize=True)

    result = await _tool(manager).invoke({})
    data = json.loads(result.content)

    assert data["decision"] == "explicit_orchestration"
    assert data["scheduler"]["dispatchable_task_ids"] == [task.id]
    assert data["safe_to_auto_dispatch"] is False
    assert "orchestrate_until_idle" in data["recommended_tools"]


async def test_delegation_policy_opt_in_can_mark_clean_dispatch_safe(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / "tasks")
    manager.create("parallel work", can_parallelize=True)

    result = await _tool(manager).invoke({"allow_auto_dispatch": True})
    data = json.loads(result.content)

    assert data["decision"] == "explicit_orchestration"
    assert data["auto_delegation_enabled"] is True
    assert data["safe_to_auto_dispatch"] is True


async def test_delegation_policy_stops_for_manual_review(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / "tasks")
    high_risk = manager.create("dangerous change", risk="high")

    result = await _tool(manager).invoke({"allow_auto_dispatch": True})
    data = json.loads(result.content)

    assert data["decision"] == "manual_review"
    assert data["scheduler"]["ready_task_ids"] == [high_risk.id]
    assert data["reasons"] == ["scheduler_skipped_manual_review_task"]
    assert data["safe_to_auto_dispatch"] is False
    assert "orchestration_summary" in data["recommended_tools"]
