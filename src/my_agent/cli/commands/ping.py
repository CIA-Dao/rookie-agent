from __future__ import annotations

import asyncio
import json
import sys
import time

from my_agent.cli.commands._connection_errors import print_core_connection_error
from my_agent.core.config import Config


def cmd_ping(config: Config) -> None:
    try:
        asyncio.run(_ping(config))
    except (ConnectionRefusedError, OSError, TimeoutError):
        print_core_connection_error(config)
        sys.exit(1)


async def _ping(config: Config) -> None:
    t0 = time.monotonic()
    reader, writer = await asyncio.open_connection(config.host, config.port)

    req = {
        "jsonrpc": "2.0",
        "id": "cli-1",
        "method": "core.ping",
        "params": {"client": "cli/0.0.1"},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=10.0)
    latency_ms = int((time.monotonic() - t0) * 1000)

    writer.close()
    await writer.wait_closed()

    raw = json.loads(line)
    if "error" in raw:
        print(f"error: {raw['error']['code']} {raw['error']['message']}", file=sys.stderr)
        sys.exit(1)

    result = raw["result"]
    msg = (
        f"pong server={result['server_version']} "
        f"uptime={result['uptime_ms']}ms latency={latency_ms}ms"
    )
    print(msg)
