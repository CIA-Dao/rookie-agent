# 每次跑集成测试时，自动启动一个临时 daemon，测完杀掉
import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator

import pytest


@pytest.fixture
def free_port() -> int:
    """找个没人用的端口，返回后释放"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
async def running_daemon(free_port: int) -> AsyncIterator[subprocess.Popen[bytes]]:
    """用 free_port 启动真 daemon，轮询等待就绪，测试结束杀掉"""
    env = os.environ.copy()
    env["MY_AGENT_PORT"] = str(free_port)
    env["MY_AGENT_LOG_FILE"] = ""
    env["MY_AGENT_LOG_LEVEL"] = "WARNING"

    proc = subprocess.Popen(
        [sys.executable, "-m", "my_agent.core"],
        env=env,
    )

    # 轮询：等 daemon 启动完毕
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
            writer.close()
            await writer.wait_closed()
            break
        except (ConnectionRefusedError, OSError):
            pass

    else:
        proc.terminate()
        proc.wait()
        pytest.fail("daemon 启动失败，3 秒内无法连接")

    yield proc

    proc.terminate()

    try:
        proc.wait(timeout=5)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
