from __future__ import annotations

from pytest import MonkeyPatch

from my_agent.cli import main as cli_main


class _FakeStream:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


def test_configure_stdio_sets_utf8_for_all_standard_streams(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_in = _FakeStream()
    fake_out = _FakeStream()
    fake_err = _FakeStream()

    monkeypatch.setattr(cli_main.sys, "stdin", fake_in)
    monkeypatch.setattr(cli_main.sys, "stdout", fake_out)
    monkeypatch.setattr(cli_main.sys, "stderr", fake_err)

    cli_main._configure_stdio()

    assert fake_in.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_err.calls == [{"encoding": "utf-8", "errors": "replace"}]
