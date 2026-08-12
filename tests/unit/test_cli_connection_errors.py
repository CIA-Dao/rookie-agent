from __future__ import annotations

import pytest

from my_agent.cli.commands._connection_errors import print_core_connection_error
from my_agent.core.config import Config


def test_print_core_connection_error_includes_address_and_next_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_core_connection_error(Config(host="127.0.0.9", port=8765))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: cannot connect to my-agent core at 127.0.0.9:8765" in captured.err
    assert "my-agent core status" in captured.err
    assert "my-agent core start" in captured.err
    assert "MY_AGENT_HOST" in captured.err
    assert "MY_AGENT_PORT" in captured.err
