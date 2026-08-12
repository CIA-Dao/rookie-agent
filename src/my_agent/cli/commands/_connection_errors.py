from __future__ import annotations

import sys

from my_agent.core.config import Config


# 打印 CLI/Core 连接失败的统一提示到 stderr；不 sys.exit，退出策略由调用方控制
def print_core_connection_error(config: Config) -> None:
    print(
        f"error: cannot connect to my-agent core at {config.host}:{config.port}",
        file=sys.stderr,
    )
    print("hint: run `my-agent core status` to inspect the daemon.", file=sys.stderr)
    print("hint: run `my-agent core start` to start it.", file=sys.stderr)
    print(
        "hint: if you changed the address, check MY_AGENT_HOST and MY_AGENT_PORT.",
        file=sys.stderr,
    )
