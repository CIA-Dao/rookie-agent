from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from rich.markup import escape
from textual.widgets import RichLog

from my_agent.tui.theme import (
    _ELECTRIC_CYAN,
    _HOT_PINK,
    _soft_wrap_stream_text,
    _stream_preview_width,
)


async def schedule_llm_render(app: Any) -> None:
    """Schedule a batched RichLog update; skip if one is already pending."""
    if app._llm_render_timer is not None:
        return
    app._llm_render_timer = asyncio.create_task(flush_after_delay(app))


async def flush_after_delay(app: Any) -> None:
    await asyncio.sleep(0.05)
    flush_llm_render(app)


def flush_llm_render(app: Any) -> None:
    app._llm_render_timer = None
    if app._current_llm is None:
        return
    text = app._current_llm.text
    if text == app._llm_last_render_text:
        return
    app._llm_last_render_text = text
    render_llm_log_line(app, _soft_wrap_stream_text(text, stream_wrap_width(app)))


async def cancel_and_flush_llm_render(app: Any) -> None:
    if app._llm_render_timer is not None:
        app._llm_render_timer.cancel()
        app._llm_render_timer = None
    await asyncio.sleep(0.06)
    flush_llm_render(app)


async def cancel_llm_render(app: Any) -> None:
    if app._llm_render_timer is not None:
        app._llm_render_timer.cancel()
        app._llm_render_timer = None
    await asyncio.sleep(0)


async def ensure_llm_block(app: Any) -> Any:
    if app._current_llm is not None:
        return app._current_llm

    from my_agent.tui.app import LLMStreamBlock

    block = LLMStreamBlock()
    block.display = False
    app._current_llm = block
    app._current_llm_log_range = None
    app._llm_last_render_text = ""
    if app._llm_render_timer is not None:
        app._llm_render_timer.cancel()
        app._llm_render_timer = None
    log = app.query_one("#log", RichLog)
    await log.mount(block)
    return block


async def add_tool_block(app: Any, tool_use_id: str, tool_name: str, params: dict[str, Any]) -> Any:
    from my_agent.tui.app import ToolCallBlock

    block = ToolCallBlock(tool_name, params)
    app._tool_blocks[tool_use_id] = block
    log = app.query_one("#log", RichLog)
    await log.mount(block)
    render_tool_log_block(app, tool_use_id)
    return block


def render_tool_log_block(app: Any, tool_use_id: str) -> None:
    block = app._tool_blocks.get(tool_use_id)
    if block is None:
        return

    log = app.query_one("#log", RichLog)
    old_range = app._tool_log_ranges.pop(tool_use_id, None)
    if old_range is not None:
        start, end = old_range
        del log.lines[start:end]
        shift_log_ranges_after_deleted_range(app, start, end)
        log.refresh()

    start = len(log.lines)
    log.write(block.render_markup())
    app._tool_log_ranges[tool_use_id] = (start, len(log.lines))


def shift_log_ranges_after_deleted_range(app: Any, start: int, end: int) -> None:
    removed = end - start
    if removed <= 0:
        return
    for tool_use_id, (range_start, range_end) in list(app._tool_log_ranges.items()):
        if range_start >= end:
            app._tool_log_ranges[tool_use_id] = (
                range_start - removed,
                range_end - removed,
            )
    if app._current_llm_log_range is not None:
        range_start, range_end = app._current_llm_log_range
        if range_start >= end:
            app._current_llm_log_range = (
                range_start - removed,
                range_end - removed,
            )


async def add_chat(app: Any, role: str, content: str) -> Any:
    from my_agent.tui.app import ChatMessageBlock

    app._leave_welcome_state()
    block = ChatMessageBlock(role, content)
    log = app.query_one("#log", RichLog)
    if role == "user":
        log.write(f"[bold {_ELECTRIC_CYAN}]YOU //[/bold {_ELECTRIC_CYAN}]  {escape(content)}")
    else:
        log.write(f"[bold {_HOT_PINK}]AGENT //[/bold {_HOT_PINK}]  {escape(content)}")
    await log.mount(block)
    return block


async def add_event(app: Any, kind: str, content: str, *, visible: bool = False) -> Any:
    from my_agent.tui.app import EventLineBlock

    if visible:
        app._leave_welcome_state()
    block = EventLineBlock(kind, content)
    log = app.query_one("#log", RichLog)
    if visible:
        log.write(
            f"[bold {_HOT_PINK}]{escape(kind.upper())} //[/bold {_HOT_PINK}] "
            f"{escape(content)}"
        )
    await log.mount(block)
    return block


def stream_wrap_width(app: Any) -> int:
    widths: list[int] = []
    with suppress(Exception):
        log = app.query_one("#log", RichLog)
        widths.append(int(getattr(log.size, "width", 0) or 0))
    with suppress(Exception):
        widths.append(int(getattr(app.size, "width", 0) or 0))
    with suppress(Exception):
        widths.append(int(getattr(app.console.size, "width", 0) or 0))
    return _stream_preview_width(max(widths, default=0))


def render_llm_log_line(app: Any, content: str) -> None:
    log = app.query_one("#log", RichLog)
    if app._current_llm_log_range is not None:
        start, end = app._current_llm_log_range
        del log.lines[start:end]
        log.refresh()
    start = len(log.lines)
    log.write(f"[bold {_HOT_PINK}]AGENT //[/bold {_HOT_PINK}]  {escape(content)}")
    app._current_llm_log_range = (start, len(log.lines))
