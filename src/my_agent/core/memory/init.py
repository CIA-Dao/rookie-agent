from __future__ import annotations

import asyncio
from pathlib import Path

from my_agent.core.config import get_config
from my_agent.core.memory.project_init import (
    InitResult,
    ProjectInitService,
    create_init_provider,
)

# 向后兼容：原有测试从本模块导入 cmd_init 和 InitResult
__all__ = ["cmd_init", "InitResult"]


def cmd_init(root: Path | None = None) -> InitResult:
    """在当前项目根目录执行项目初始化，生成 context 和指导文件。"""
    target = root or Path.cwd()
    config = get_config()
    provider = create_init_provider(config.llm.default_model)
    service = ProjectInitService(target, provider=provider)
    return asyncio.run(service.run())
