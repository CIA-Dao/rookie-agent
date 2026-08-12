from __future__ import annotations

from pathlib import Path

from my_agent.core.memory.init import cmd_init as _core_cmd_init


# CLI 入口：在当前工作目录执行 init，打印返回的消息
def cmd_init() -> None:
    result = _core_cmd_init(Path.cwd())
    for msg in result.messages:
        print(msg)
