from __future__ import annotations

from my_agent.tui.app import _display_width, _soft_wrap_stream_text, _stream_preview_width


def test_display_width_counts_cjk_as_two() -> None:
    assert _display_width("abc") == 3
    assert _display_width("你好") == 4
    assert _display_width("a你b") == 4


def test_soft_wrap_stream_text_wraps_cjk_by_display_width() -> None:
    text = "前些天傍晚我路过一条老街"

    wrapped = _soft_wrap_stream_text(text, 10)

    lines = wrapped.splitlines()
    assert len(lines) > 1
    assert all(_display_width(line) <= 10 for line in lines)


def test_soft_wrap_stream_text_preserves_existing_newlines() -> None:
    text = "第一段\n第二段很长很长"

    wrapped = _soft_wrap_stream_text(text, 8)

    assert wrapped.splitlines()[0] == "第一段"


def test_stream_preview_width_uses_stable_floor_for_tiny_layout_width() -> None:
    assert _stream_preview_width(0) == 80
    assert _stream_preview_width(12) == 72
    assert _stream_preview_width(40) == 72
    assert _stream_preview_width(120) == 104


def test_stream_preview_floor_does_not_create_narrow_cjk_column() -> None:
    text = "那时候窄的发糕，早晨推开门，冷风把人吹醒。" * 3

    wrapped = _soft_wrap_stream_text(text, _stream_preview_width(20))

    lines = [line for line in wrapped.splitlines() if line]
    assert lines
    assert all(_display_width(line) >= 20 for line in lines[:-1])
