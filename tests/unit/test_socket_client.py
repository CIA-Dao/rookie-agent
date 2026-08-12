from __future__ import annotations

import pytest

from my_agent.core.transport.socket_client import SocketClient


class _ResetOnCloseWriter:
    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        raise ConnectionResetError("peer already closed")


@pytest.mark.asyncio
async def test_socket_client_close_ignores_reset_from_already_closed_peer() -> None:
    client = SocketClient("127.0.0.1", 7437)
    client._writer = _ResetOnCloseWriter()  # type: ignore[assignment]

    await client.close()
