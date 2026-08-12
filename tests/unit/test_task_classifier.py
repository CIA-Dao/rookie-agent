from __future__ import annotations

from my_agent.core.compact.policy import policy_for_task
from my_agent.core.task.classifier import TaskType, classify_task


def test_classify_code_work_from_fix_or_test_terms() -> None:
    assert classify_task("修复 pytest 失败") == TaskType.CODE_WORK
    assert classify_task("implement the new handler") == TaskType.CODE_WORK


def test_classify_code_read_from_explain_or_read_terms() -> None:
    assert classify_task("解释一下 runner.py 的流程") == TaskType.CODE_READ
    assert classify_task("read the project structure") == TaskType.CODE_READ


def test_classify_unknown_goal_falls_back_to_chat() -> None:
    assert classify_task("今天状态怎么样") == TaskType.CHAT


def test_policy_for_task_returns_stable_policy_names() -> None:
    assert policy_for_task(TaskType.CHAT).name == "chat_v1"
    assert policy_for_task(TaskType.CODE_READ).name == "code_read_v1"
    assert policy_for_task(TaskType.CODE_WORK).name == "code_work_v1"
