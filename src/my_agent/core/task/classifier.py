from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    CHAT = "chat"
    CODE_READ = "code_read"
    CODE_WORK = "code_work"


_CODE_WORK_HINTS = (
    "bug",
    "error",
    "fail",
    "fix",
    "implement",
    "pytest",
    "test",
    "修改",
    "修复",
    "实现",
    "报错",
    "失败",
    "测试",
)

_CODE_READ_HINTS = (
    "architecture",
    "explain",
    "flow",
    "read",
    "structure",
    "分析",
    "解释",
    "架构",
    "流程",
    "读一下",
    "看一下",
)


def classify_task(goal: str) -> TaskType:
    normalized = goal.lower()
    if any(hint in normalized for hint in _CODE_WORK_HINTS):
        return TaskType.CODE_WORK
    if any(hint in normalized for hint in _CODE_READ_HINTS):
        return TaskType.CODE_READ
    return TaskType.CHAT
