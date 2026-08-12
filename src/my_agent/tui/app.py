from __future__ import annotations

import asyncio
import json
import platform
import shutil
import subprocess
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from unicodedata import east_asian_width

from rich.markdown import Markdown
from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.events import Key
from textual.widgets import Button, Input, RichLog, Static, TextArea

from my_agent.core.config import DEEPSEEK_MODEL_OPTIONS, normalize_deepseek_model
from my_agent.core.lifecycle import ensure_core_started
from my_agent.core.skills.loader import SkillLoader
from my_agent.core.transport.socket_client import IpcError, SocketClient
from my_agent.core.user_config import save_deepseek_api_key, save_deepseek_model
from my_agent.tui import rendering
from my_agent.tui.event_policy import (
    context_compacted_entry,
    context_compaction_failed_entry,
    run_finished_entry,
    scheduler_event_entry,
    skill_invoked_entry,
    skill_tool_compatibility_entry,
    subagent_finished_entry,
    subagent_started_entry,
    task_event_entry,
    unknown_event_entry,
)
from my_agent.tui.messages import CoreConnected, CoreDisconnected, CoreEvent
from my_agent.tui.overlays import (
    ModelSelect,
    PendingPermission,
    PermissionSelect,
    PromptTextArea,
    SettingsDialog,
    SettingsSelection,
    SlashCommand,
    SlashPalette,
)
from my_agent.tui.terminal_title import (
    STARTUP_STATE_CONNECTING,
    STARTUP_STATE_CREATING_SESSION,
    STARTUP_STATE_DISCONNECTED,
    STARTUP_STATE_READY,
    STARTUP_STATE_RUNNING,
    STARTUP_STATE_SETUP_ERROR,
    STARTUP_STATE_STARTING,
    STARTUP_STATE_WAITING_PERMISSION,
    STARTUP_STATE_WORKSPACE_IN_USE,
    connection_label_for_state,
    format_terminal_title,
    prompt_border_title_for_state,
    run_state_label_for_state,
    sanitize_project_name,
)
from my_agent.tui.theme import (
    _BRAND_TEXT,
    _ELECTRIC_CYAN,
    _ERROR,
    _HOT_PINK,
    _SUCCESS,
    _WARNING,
    _display_width,  # noqa: F401 - compatibility export for existing TUI tests
    _soft_wrap_stream_text,
    _stream_preview_width,
)

_DECISION_MAP: dict[str, str] = {
    "y": "allow_once",
    "a": "always_allow",
    "n": "deny_once",
    "d": "always_deny",
}

# ---------------------------------------------------------------------------
# P5-2: shared visual theme tokens.
# ---------------------------------------------------------------------------

def _legacy_soft_wrap_stream_text(text: str, width: int) -> str:
    """Stable soft-wrap for streaming preview. Does NOT insert Rich markup.

    Wraps by display width so CJK text gets consistent line breaks during
    incremental streaming — RichLog's own auto-wrap can produce unstable
    narrow columns when re-measuring long CJK text repeatedly.
    """
    if width <= 8:
        return text

    lines: list[str] = []
    current: list[str] = []
    current_width = 0

    for char in text:
        if char == "\n":
            lines.append("".join(current))
            current = []
            current_width = 0
            continue

        char_width = 2 if east_asian_width(char) in ("F", "W") else 1
        if current and current_width + char_width > width:
            lines.append("".join(current))
            current = [char]
            current_width = char_width
        else:
            current.append(char)
            current_width += char_width

    if current or not lines:
        lines.append("".join(current))

    return "\n".join(lines)


def _legacy_stream_preview_width(measured_width: int) -> int:
    """Return a stable wrap width for assistant streaming preview.

    RichLog.size.width can be transiently tiny while Textual is laying out
    docked widgets or while the log is being rewritten. If we trust that tiny
    value, CJK streaming text gets pre-wrapped into a narrow column, which looks
    like garbled vertical text. Keep a generous floor and reserve a little room
    for the assistant label, padding, and scrollbar.
    """
    if measured_width <= 0:
        return 80
    return max(72, measured_width - 16)


# ---------------------------------------------------------------------------
# P5-1: Custom AppHeader — replaces default Textual Header with a product
# status strip showing brand / project / connection / session / run state.
# Color is ALWAYS paired with text so the status is readable without color.
# ---------------------------------------------------------------------------


class AppHeader(Static):
    """Top status strip: My Agent | project | connection | run state.

    P5-2.1: session id is intentionally kept out of the default ready UI.
    It still lives inside the app for routing, but is not surfaced here.
    """

    DEFAULT_CSS = """
    AppHeader {
        dock: top;
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $text;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._brand = _BRAND_TEXT
        self._project_name = ""
        self._connection_label = "connecting"
        self._session_short = "-"
        self._run_state_label = "starting"

    def compose(self) -> ComposeResult:
        yield Static("", id="header-status")
        yield Button("Settings", id="settings-button")

    def set_state(
        self,
        *,
        project_name: str | None = None,
        connection_label: str | None = None,
        session_short: str | None = None,
        run_state_label: str | None = None,
    ) -> None:
        if project_name is not None:
            self._project_name = project_name
        if connection_label is not None:
            self._connection_label = connection_label
        if session_short is not None:
            self._session_short = session_short
        if run_state_label is not None:
            self._run_state_label = run_state_label
        self._refresh()

    def _refresh(self) -> None:
        # Truncate project name to a budget based on header visual width.
        # The header is a single line; long CJK names must be clipped.
        project = self._project_name or "-"
        max_project_chars = 24
        if len(project) > max_project_chars:
            project = project[: max_project_chars - 1] + "…"

        conn = self._connection_label
        if conn == "connected":
            conn_markup = f"[bold {_SUCCESS}]{conn}[/bold {_SUCCESS}]"
        elif conn in ("connecting", "setup needed"):
            conn_markup = f"[bold {_WARNING}]{conn}[/bold {_WARNING}]"
        elif conn == "disconnected":
            conn_markup = f"[bold {_ERROR}]{conn}[/bold {_ERROR}]"
        else:
            conn_markup = f"[bold]{conn}[/bold]"

        run = self._run_state_label
        if run == "running":
            run_markup = f"[bold {_ELECTRIC_CYAN}]{run}[/bold {_ELECTRIC_CYAN}]"
        elif run == "permission":
            run_markup = f"[bold {_WARNING}]{run}[/bold {_WARNING}]"
        elif run in ("offline", "error"):
            run_markup = f"[bold {_ERROR}]{run}[/bold {_ERROR}]"
        elif run == "ready":
            run_markup = f"[bold {_SUCCESS}]{run}[/bold {_SUCCESS}]"
        else:
            run_markup = f"[dim]{run}[/dim]"

        line = (
            f"[bold {_HOT_PINK}]{self._brand}[/bold {_HOT_PINK}] | "
            f"[bold]{escape(project)}[/bold] | "
            f"{conn_markup} | "
            f"{run_markup}"
        )
        self.update(line)
        try:
            self.query_one("#header-status", Static).update(line)
        except Exception:
            pass


class StartupPanel(Static):
    """Product-facing startup surface shown before the chat is ready.

    P5-2.1: this panel also renders the welcome empty-state banner, so the
    first screen never dumps welcome text into the RichLog chat timeline.
    """

    DEFAULT_CSS = """
    StartupPanel {
        height: auto;
        min-height: 5;
        margin: 1 2;
        padding: 1 2;
        background: $surface;
        border: round $primary;
        color: $text;
    }
    """

    _VISIBLE_STATES = frozenset(
        {
            STARTUP_STATE_STARTING,
            STARTUP_STATE_CONNECTING,
            STARTUP_STATE_CREATING_SESSION,
            STARTUP_STATE_SETUP_ERROR,
            STARTUP_STATE_WORKSPACE_IN_USE,
            STARTUP_STATE_READY,
        }
    )

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._state = STARTUP_STATE_STARTING
        self._project_name = ""
        self._detail = ""
        self._welcome_mode = False

    def set_state(
        self,
        state: str,
        *,
        project_name: str = "",
        detail: str = "",
    ) -> None:
        self._state = state
        self._welcome_mode = state == STARTUP_STATE_READY
        self.remove_class("welcome")
        if project_name:
            self._project_name = project_name
        self._detail = detail
        self.display = state in self._VISIBLE_STATES
        self._refresh()

    def show_welcome(self, *, project_name: str = "") -> None:
        """Switch to the welcome banner (READY empty-state)."""
        self._state = STARTUP_STATE_READY
        self._welcome_mode = True
        if project_name:
            self._project_name = project_name
        self._detail = ""
        self.display = True
        self.add_class("welcome")
        self._refresh()

    def hide(self) -> None:
        """Hide the banner once the user has started the chat."""
        self.display = False
        self.remove_class("welcome")

    def set_detail(self, detail: str) -> None:
        self._detail = detail
        self._refresh()

    def _refresh(self) -> None:
        project = escape(self._project_name or "current project")

        if self._welcome_mode:
            banner = "\n".join(
                (
                    "███╗   ███╗██╗   ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
                    "████╗ ████║╚██╗ ██╔╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
                    "██╔████╔██║ ╚████╔╝     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║",
                    "██║╚██╔╝██║  ╚██╔╝      ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║",
                    "██║ ╚═╝ ██║   ██║       ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║",
                    "╚═╝     ╚═╝   ╚═╝       ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝",
                )
            )
            self.update(
                f"[bold {_ELECTRIC_CYAN}]{banner}[/bold {_ELECTRIC_CYAN}]\n"
                f"[bold {_HOT_PINK}]MY AGENT[/bold {_HOT_PINK}]  [dim]::[/dim]  "
                f"[bold]{project}[/bold]\n"
                f"[dim]type a message and press Enter  ·  / triggers commands  ·  "
                f"Ctrl+Q exits[/dim]\n"
                f"[dim]/init  /compact  /tools  /copy[/dim]"
            )
            return

        if self._state == STARTUP_STATE_STARTING:
            phase = "Loading / Starting"
            description = "Preparing the project workspace"
            stamp_color = _WARNING
        elif self._state == STARTUP_STATE_CONNECTING:
            phase = "Loading / Connecting to Core"
            description = "Checking the local agent service"
            stamp_color = _WARNING
        elif self._state == STARTUP_STATE_CREATING_SESSION:
            phase = "Loading / Opening session"
            description = "Preparing a chat session for this project"
            stamp_color = _ELECTRIC_CYAN
        elif self._state == STARTUP_STATE_WORKSPACE_IN_USE:
            phase = "Workspace already open"
            description = "Another TUI is using an overlapping workspace"
            stamp_color = _ERROR
        else:
            phase = "Setup required"
            description = self._detail or "My Agent could not finish startup"
            stamp_color = _ERROR

        detail = f"\n[dim]{escape(self._detail)}[/dim]" if self._detail else ""
        stamp = f"[bold {stamp_color}]● {self._state}[/bold {stamp_color}]"
        self.update(
            f"[bold {_HOT_PINK}]MY AGENT[/bold {_HOT_PINK}]  [dim]::[/dim]  "
            f"[bold]{project}[/bold]\n"
            f"[bold]{phase}[/bold]  {stamp}\n"
            f"{description}{detail}"
        )


class LLMStreamBlock(Static):
    """A single assistant message that updates as llm.token events arrive."""

    def __init__(self) -> None:
        super().__init__("")
        self._text = ""
        self._finalized = False

    @property
    def text(self) -> str:
        return self._text

    def append_token(self, token: str) -> None:
        if self._finalized:
            return
        self._text += token
        self.update(self._text)

    def finalize_markdown(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._text.strip():
            self.update(Markdown(self._text))


class ChatMessageBlock(Static):
    """A user or non-streaming assistant chat message."""

    DEFAULT_CSS = """
    ChatMessageBlock {
        padding: 0 1;
    }
    ChatMessageBlock.user {
        background: $boost;
    }
    ChatMessageBlock.assistant {
        color: $text;
    }
    """

    def __init__(self, role: str, content: str) -> None:
        super().__init__("")
        self._role = role
        self._content = content
        self.add_class(role)
        self._refresh()

    @property
    def role(self) -> str:
        return self._role

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value
        self._refresh()

    def _refresh(self) -> None:
        if self._role == "user":
            self.update(f"[bold {_ELECTRIC_CYAN}]YOU //[/bold {_ELECTRIC_CYAN}]  {self._content}")
        else:
            self.update(f"[bold {_HOT_PINK}]AGENT //[/bold {_HOT_PINK}]  {self._content}")


class EventLineBlock(Static):
    """A low-emphasis infrastructure event line."""

    DEFAULT_CSS = """
    EventLineBlock {
        padding: 0 1;
        color: $text-muted;
    }
    EventLineBlock.permission {
        color: $warning;
    }
    EventLineBlock.skill {
        color: $accent;
    }
    EventLineBlock.subagent {
        color: $accent;
    }
    EventLineBlock.task {
        color: $accent;
    }
    EventLineBlock.scheduler {
        color: $accent;
    }
    """

    _TAG_MARKUP: dict[str, str] = {
        "run": "[dim]run[/dim]",
        "permission": "[bold]permission[/bold]",
        "skill": "[bold]skill[/bold]",
        "system": "[dim]system[/dim]",
        "subagent": "[bold]subagent[/bold]",
        "task": "[bold]task[/bold]",
        "scheduler": "[bold]scheduler[/bold]",
    }

    def __init__(self, kind: str, content: str) -> None:
        super().__init__("")
        self._kind = kind
        self._content = content
        if kind in ("permission", "skill", "run", "system", "subagent"):
            self.add_class(kind)
        self._refresh()

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value
        self._refresh()

    def _refresh(self) -> None:
        tag = self._TAG_MARKUP.get(self._kind, f"[dim]{self._kind}[/dim]")
        self.update(f"{tag} {self._content}")


def _param_summary(tool_name: str, params: dict[str, Any], max_len: int = 80) -> str:
    keys_by_tool = {
        "read_file": ("path",),
        "write_file": ("path",),
        "list_dir": ("path", "max_depth"),
        "bash": ("command",),
        "note_save": ("content",),
    }
    keys = keys_by_tool.get(tool_name, ())
    parts = [f"{key}={params[key]!r}" for key in keys if key in params]

    if not parts:
        parts = [f"{key}={value!r}" for key, value in list(params.items())[:2]]

    text = ", ".join(parts)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


class ToolCallBlock(Static):
    """A compact visual line for one tool call."""

    def __init__(self, tool_name: str, params: dict[str, Any]) -> None:
        super().__init__("")
        self._tool_name = tool_name
        self._params = params
        self._finished = False
        self._is_error = False
        self._elapsed_ms = 0
        self._error_message = ""
        self._expanded = False
        self.update(self._render_line())

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def expanded(self) -> bool:
        return self._expanded

    def toggle_details(self) -> None:
        self._expanded = not self._expanded
        self.update(self._render_line())

    def render_markup(self) -> str:
        return self._render_line()

    def set_result(
        self,
        elapsed_ms: int,
        *,
        is_error: bool = False,
        error_message: str = "",
    ) -> None:
        self._finished = True
        self._is_error = is_error
        self._elapsed_ms = elapsed_ms
        self._error_message = error_message
        self.update(self._render_line())

    def _render_line(self) -> str:
        params = _param_summary(self._tool_name, self._params)

        line = f"[bold {_HOT_PINK}]TOOL[/bold {_HOT_PINK}] / [bold]{self._tool_name}[/bold]"
        if params:
            line += f" / [dim]{params}[/dim]"

        if not self._finished:
            line += f" / [bold {_WARNING}]running[/bold {_WARNING}]"
            if self._expanded:
                line += self._render_details()
            return line

        if self._is_error:
            line += f" / [bold {_ERROR}]failed[/bold {_ERROR}] / [dim]{self._elapsed_ms}ms[/dim]"
            if self._error_message:
                line += f" / [red]{escape(self._error_message)}[/red]"
            if self._expanded:
                line += self._render_details()
            return line

        line += f" / [bold {_SUCCESS}]done[/bold {_SUCCESS}] / [dim]{self._elapsed_ms}ms[/dim]"
        if self._expanded:
            line += self._render_details()
        return line

    def _render_details(self) -> str:
        try:
            params = json.dumps(self._params, ensure_ascii=False, indent=2, sort_keys=True)
        except TypeError:
            params = repr(self._params)

        details = f"\n[dim]details[/dim]\n[dim]params[/dim]\n{escape(params)}"
        if self._is_error and self._error_message:
            details += f"\n[dim]error[/dim]\n[red]{escape(self._error_message)}[/red]"
        return details

    def plain_text_summary(self) -> str:
        """Return a plain-text summary of the tool call for copy/export."""
        parts = [f"TOOL / {self._tool_name}"]
        params = _param_summary(self._tool_name, self._params)
        if params:
            parts.append(f"params: {params}")

        if not self._finished:
            parts.append("status: running")
            return "\n".join(parts)

        parts.append(f"elapsed_ms: {self._elapsed_ms}")
        if self._is_error:
            parts.append("status: failed")
            if self._error_message:
                parts.append(f"error: {self._error_message}")
        else:
            parts.append("status: done")

        try:
            full_params = json.dumps(self._params, ensure_ascii=False, indent=2, sort_keys=True)
        except TypeError:
            full_params = repr(self._params)
        parts.append(f"params:\n{full_params}")
        return "\n".join(parts)


class RunStatusBlock(Static):
    """Compact status strip for the currently active agent run."""

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._run_id = ""
        self._status = "idle"
        self._tool_running = 0
        self._tool_done = 0
        self._tool_failed = 0
        self._waiting_permission = ""
        # P1-1: llm.usage state (low-noise status, not chat log)
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0
        self._usage_context_pct = 0.0
        self._usage_cache_read_tokens = 0
        self._usage_cache_creation_tokens = 0
        self._has_usage = False
        # P1-2: model name state (low-noise status, not chat log)
        self._model = ""
        self._model_strategy = ""
        # P1-4: step progress state (status strip only, not chat log)
        self._step = 0
        self._step_state = ""
        # P2-0.5: activity and spinner state
        self._activity = "idle"
        self._activity_detail = ""
        self._spinner_index = 0
        self._refresh()
        # P5-2.1: do not show the idle/debug status strip in the default ready UI.
        self.display = False

    def start(self, run_id: str) -> None:
        self._run_id = run_id
        self._status = "running"
        self._tool_running = 0
        self._tool_done = 0
        self._tool_failed = 0
        self._waiting_permission = ""
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0
        self._usage_context_pct = 0.0
        self._usage_cache_read_tokens = 0
        self._usage_cache_creation_tokens = 0
        self._has_usage = False
        self._model = ""
        self._model_strategy = ""
        self._step = 0
        self._step_state = ""
        self._activity = "running"
        self._activity_detail = ""
        self._spinner_index = 0
        self.display = True
        self._refresh()

    def set_activity(self, activity: str, detail: str = "") -> None:
        self._activity = activity
        self._activity_detail = detail
        # P5-2.1: surface the status strip only when something is actually happening.
        if activity not in ("idle", "done"):
            self.display = True
        self._refresh()

    def tick(self) -> None:
        if self._display_visible_for_tick():
            self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
            self._refresh()

    def _display_visible_for_tick(self) -> bool:
        """Return whether the spinner should tick based on visible state."""
        if not self.display:
            return False
        if self._status == "running":
            return True
        return self._activity not in ("idle", "done")

    def finish(self, status: str) -> None:
        self._status = status or "finished"
        self._waiting_permission = ""
        self._tool_running = 0
        self._activity = "done"
        self._activity_detail = ""
        self._refresh()
        # P5-2.1: collapse the status strip once the run is over so the ready UI
        # does not keep showing a static run id / tool count debug strip.
        self.display = False

    # Existing methods unchanged
    def wait_permission(self, tool_name: str) -> None:
        self._waiting_permission = tool_name
        self._refresh()

    def clear_permission(self) -> None:
        self._waiting_permission = ""
        self._refresh()

    def tool_started(self) -> None:
        self._tool_running += 1
        self._refresh()

    def tool_finished(self, *, failed: bool = False) -> None:
        self._tool_running = max(0, self._tool_running - 1)
        if failed:
            self._tool_failed += 1
        else:
            self._tool_done += 1
        self._refresh()

    def set_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        context_pct: float,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        self._usage_input_tokens = input_tokens
        self._usage_output_tokens = output_tokens
        self._usage_context_pct = context_pct
        self._usage_cache_read_tokens = cache_read_input_tokens
        self._usage_cache_creation_tokens = cache_creation_input_tokens
        self._has_usage = True
        self._refresh()

    def set_model(self, model: str, strategy: str = "") -> None:
        self._model = model
        self._model_strategy = strategy
        self._refresh()

    def step_started(self, step: int) -> None:
        self._step = step
        self._step_state = "running"
        self._refresh()

    def step_finished(self, step: int) -> None:
        self._step = step
        self._step_state = "done"
        self._refresh()

    @staticmethod
    def _short_run_id(run_id: str) -> str:
        if not run_id:
            return "-"
        return run_id[-8:] if len(run_id) > 8 else run_id

    def _refresh(self) -> None:
        run = RunStatusBlock._short_run_id(self._run_id)

        # Build head (leftmost): spinner + activity or status icon
        if self._status == "running" or self._activity not in ("idle", "done"):
            frame = self.SPINNER_FRAMES[self._spinner_index % len(self.SPINNER_FRAMES)]
            label = self._activity
            if self._activity_detail:
                label += f" {self._activity_detail}"
            head = f"{frame} {label}"
        elif self._status == "success":
            head = "✓ success"
        elif self._status in ("failed", "error"):
            head = f"✗ {self._status}"
        else:
            head = self._status

        line = f"[bold]{head}[/bold] | run [dim]{run}[/dim]"

        if self._step:
            line += f" | step {self._step} {self._step_state}"

        line += f" | tools {self._tool_running}/{self._tool_done}/{self._tool_failed}"

        if self._waiting_permission:
            line += f" | [yellow]permission {self._waiting_permission}[/yellow]"

        if self._has_usage:
            pct = (
                round(self._usage_context_pct * 100)
                if self._usage_context_pct <= 1.0
                else int(self._usage_context_pct)
            )
            line += f" | ctx {pct}%"

        if self._model:
            line += f" | {self._model}"

        self.update(line)


class _LegacyModelSelect(Static):
    _models: tuple[str, ...]
    _selected: int
    _current: str

    def _refresh(self) -> None:
        lines = [
            "[bold]Select DeepSeek model[/bold]  "
            "[dim]↑/↓ move · Enter select · Esc cancel[/dim]"
        ]
        for index, model in enumerate(self._models):
            marker = ">" if index == self._selected else " "
            current = " [dim](current)[/dim]" if model == self._current else ""
            lines.append(f"[bold cyan]{marker}[/bold cyan] {model}{current}")
        self.update("\n".join(lines))


class MyAgentTuiApp(App[None]):
    """Textual TUI that connects to Core and owns one chat session."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        global_env_file: Path | None = None,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._client: SocketClient | None = None
        self._session_id: str | None = None
        self._busy = False
        self._current_llm: LLMStreamBlock | None = None
        self._current_llm_log_range: tuple[int, int] | None = None
        # P2-0.6: streaming render throttle — avoid per-token RichLog rewrite
        self._llm_render_timer: asyncio.Task[None] | None = None
        self._llm_last_render_text: str = ""
        self._tool_blocks: dict[str, ToolCallBlock] = {}
        self._tool_log_ranges: dict[str, tuple[int, int]] = {}
        self._active_permission: PendingPermission | None = None
        self._permission_queue: deque[PendingPermission] = deque()
        self._accepted_run_ids: set[str] = set()
        self._awaiting_local_run_started = False
        self._slash_hint_shown = False
        # Bootstrap state
        self._global_env_file = global_env_file
        self._setup_mode = False
        self._setup_reason = ""
        self._current_model = next(iter(DEEPSEEK_MODEL_OPTIONS))
        # P5-1: product shell startup state machine. Single source of truth
        # for terminal title, AppHeader content, and prompt border title.
        self._startup_state: str = STARTUP_STATE_STARTING
        # Sanitized project name (cwd leaf), set on mount. Empty until then.
        self._project_name_raw: str = ""
        # Whether we have shown the welcome empty-state line yet.
        self._welcome_shown: bool = False
        # P5-2: latest exchange used by /copy.
        self._last_user_text: str = ""
        self._last_assistant_text: str = ""

    CSS = """
        $primary: #ff149d;
        $accent: #00e5ff;
        $background: #0d0d0d;
        $surface: #121212;
        $panel: #1a1a1a;
        $text: #f0f0f0;
        $text-muted: #a0a0a0;
        $warning: #ffbf00;
        $error: #ff4d4f;

        * {
            scrollbar-background: $surface;
            scrollbar-color: $primary;
            scrollbar-color-hover: $accent;
            scrollbar-color-active: $accent;
            scrollbar-background-hover: $background;
            scrollbar-background-active: $background;
            scrollbar-corner-color: $background;
        }

        Screen {
            background: $background;
            color: $text;
        }

        AppHeader {
            dock: top;
            background: $panel;
            color: $text;
        }

        #settings-button {
            dock: right;
            width: 12;
            height: 1;
            min-width: 12;
            padding: 0;
            background: $panel;
            color: $accent;
            border: none;
        }

        StartupPanel {
            width: 1fr;
            background: $surface;
            border: round $primary;
            color: $text;
        }

        StartupPanel.welcome {
            min-height: 8;
            margin: 1 1 0 1;
            padding: 0 1;
            background: $background;
            border: none;
        }

        RunStatusBlock {
            dock: top;
            padding: 0 1;
            background: $panel;
            color: $text;
        }

        RichLog {
            height: 1fr;
            margin: 0 1;
            background: $background;
            color: $text;
        }

        RichLog:focus {
            border: none;
            outline: none;
            background: $background;
        }

        TextArea {
            height: 6;
            min-height: 3;
            max-height: 12;
            border: round $primary;
            padding: 0 1;
            background: $surface;
            color: $text;
        }

        TextArea:focus {
            border: round $accent;
            outline: none;
        }

        Input {
            background: $surface;
            color: $text;
            border: round $primary;
        }

        Input:focus {
            border: round $accent;
            outline: none;
        }

        #setup-prompt {
            height: 3;
        }

        SlashPalette {
            padding: 0 1;
            background: $surface;
            border: tall $primary;
            color: $text;
        }

    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("enter", "submit_prompt", "Send"),
        ("ctrl+t", "toggle_tool_details", "Tool details"),
    ]

    async def action_submit_prompt(self) -> None:
        """Send or confirm the current TextArea content via Enter."""
        # Skip in setup mode (uses setup-prompt Input's own Enter handler)
        if self._setup_mode:
            return
        try:
            prompt = self.query_one("#prompt", TextArea)
        except Exception:
            return
        if prompt.disabled:
            return
        content = prompt.text
        if content.strip() or self._active_permission is not None:
            await self._submit_prompt_text(content)

    def action_toggle_tool_details(self) -> None:
        """Expand/collapse the most recent tool call block."""
        self._toggle_latest_tool_details(visible=True)

    def _toggle_latest_tool_details(self, *, visible: bool = False) -> bool:
        """Expand/collapse the most recent tool call block."""
        log = self.query_one("#log", RichLog)
        if not self._tool_blocks:
            if visible:
                self._leave_welcome_state()
                log.write("[italic yellow]no tool calls yet[/italic yellow]")
            return False
        latest_tool_use_id = next(reversed(self._tool_blocks))
        latest_block = self._tool_blocks[latest_tool_use_id]
        latest_block.toggle_details()
        if visible:
            self._render_tool_log_block(latest_tool_use_id)
        return True

    def compose(self) -> ComposeResult:
        yield AppHeader(id="app-header")
        yield RunStatusBlock(id="run-status")
        yield StartupPanel(id="startup-panel")
        yield RichLog(id="log", highlight=True, markup=True, wrap=True, auto_scroll=True)
        yield SlashPalette(id="slash-palette")
        yield PermissionSelect(id="permission-select")
        yield ModelSelect(id="model-select")
        yield PromptTextArea(id="prompt", show_line_numbers=False)
        # P3-0: setup-only Input preserves password masking for API key entry.
        # Hidden by default; shown only when entering setup mode.
        yield Input(
            placeholder="Paste DeepSeek API key and press Enter...",
            id="setup-prompt",
            password=True,
        )

    def on_mount(self) -> None:
        # P5-1: capture project name (cwd leaf) once, sanitized.
        try:
            self._project_name_raw = sanitize_project_name(Path.cwd().name)
        except Exception:
            self._project_name_raw = ""
        # Initial terminal title + header + prompt border title.
        self._update_startup_state(STARTUP_STATE_STARTING)
        # Hide setup-only Input; only shown in setup mode.
        self.query_one("#setup-prompt", Input).display = False
        # P5-1: do NOT dump "checking core at host:port" or
        # "Enter inserts newline; Ctrl+Enter sends message." into the chat
        # timeline — those are infrastructure / hint messages that belong
        # in the AppHeader / prompt border title, not the conversation.
        # Disable prompt until bootstrap decides what to do.
        self._set_prompt_enabled(False)
        # P2-0.5: periodic spinner refresh (every ~120ms)
        self.set_interval(0.12, self._tick_status_activity)
        self.set_interval(30.0, self.send_heartbeat)
        self.bootstrap_core()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "settings-button":
            return
        if self._busy or self._setup_mode:
            return
        self._open_settings_dialog()

    def _tick_status_activity(self) -> None:
        try:
            self.query_one("#run-status", RunStatusBlock).tick()
        except Exception:
            pass  # timer may fire during test teardown when widget is gone

    # ------------------------------------------------------------------
    # P5-1: startup state machine — single source of truth for terminal
    # title, AppHeader content, and prompt border title.
    # ------------------------------------------------------------------

    def _update_startup_state(self, state: str) -> None:
        """Update startup_state and propagate to title / header / prompt / widgets.

        P5-2.1: this also owns the visibility of the main surface widgets so the
        loading stage shows a focused banner instead of a disabled prompt and an
        empty RichLog, and the ready stage shows the welcome banner until the user
        sends the first message.
        """
        self._startup_state = state

        # Terminal title (app.title). sanitize_project_name is called again
        # inside format_terminal_title for defense-in-depth.
        self.title = format_terminal_title(state, self._project_name_raw)

        # AppHeader content. Session id is deliberately not displayed by default.
        try:
            header = self.query_one("#app-header", AppHeader)
        except Exception:
            header = None
        if header is not None:
            header.set_state(
                project_name=self._project_name_raw or "-",
                connection_label=connection_label_for_state(state),
                session_short="-",
                run_state_label=run_state_label_for_state(state),
            )

        try:
            startup = self.query_one("#startup-panel", StartupPanel)
        except Exception:
            startup = None
        # Determine whether the main chat surface should be visible in the
        # READY state. Must be computed *before* setting self._welcome_shown.
        ready_show_log = self._welcome_shown or self._has_chat_messages()
        if startup is not None:
            if state == STARTUP_STATE_READY:
                if ready_show_log:
                    startup.hide()
                else:
                    startup.show_welcome(project_name=self._project_name_raw)
                    self._welcome_shown = True
                # Leave early so the generic set_state path does not override
                # the welcome state.
                pass
            else:
                startup.set_state(
                    state,
                    project_name=self._project_name_raw,
                    detail=(
                        self._setup_reason
                        if state
                        in (STARTUP_STATE_SETUP_ERROR, STARTUP_STATE_WORKSPACE_IN_USE)
                        else ""
                    ),
                )

        try:
            log = self.query_one("#log", RichLog)
        except Exception:
            log = None
        if log is not None:
            if state in (
                STARTUP_STATE_STARTING,
                STARTUP_STATE_CONNECTING,
                STARTUP_STATE_CREATING_SESSION,
            ):
                # P5-2.1: loading stage — no empty chat log behind the banner.
                log.display = False
            elif state == STARTUP_STATE_SETUP_ERROR:
                # Setup diagnostics stay in the chat log; only the banner is shown.
                log.display = True
            elif state == STARTUP_STATE_READY:
                # Keep the empty log visible as the main content spacer while the
                # welcome banner is shown, so the prompt remains bottom anchored.
                log.display = True
            else:
                log.display = True

        try:
            prompt = self.query_one("#prompt", TextArea)
        except Exception:
            prompt = None
        if prompt is not None:
            if state in (
                STARTUP_STATE_STARTING,
                STARTUP_STATE_CONNECTING,
                STARTUP_STATE_CREATING_SESSION,
                STARTUP_STATE_SETUP_ERROR,
                STARTUP_STATE_WORKSPACE_IN_USE,
            ):
                # P5-2.1: hide the disabled prompt during startup/setup.
                prompt.display = False
            else:
                prompt.display = True

        # Prompt border title.
        self._set_prompt_border_title(prompt_border_title_for_state(state))

    def _has_chat_messages(self) -> bool:
        """Return whether any ChatMessageBlock has been mounted."""
        with suppress(Exception):
            return bool(list(self.query(ChatMessageBlock)))
        return False

    def _leave_welcome_state(self) -> None:
        """Hide the welcome banner and reveal the RichLog for the first message."""
        self._welcome_shown = True
        try:
            startup = self.query_one("#startup-panel", StartupPanel)
            startup.hide()
        except Exception:
            pass
        try:
            log = self.query_one("#log", RichLog)
            log.display = True
        except Exception:
            pass
        try:
            prompt = self.query_one("#prompt", TextArea)
            prompt.display = True
        except Exception:
            pass

    def _set_prompt_border_title(self, title: str) -> None:
        """Set the TextArea border_title, tolerating widget-not-mounted."""
        try:
            prompt = self.query_one("#prompt", TextArea)
            prompt.border_title = title
        except Exception:
            pass

    # P5-2.1: _maybe_show_welcome is a no-op kept for backward compat; welcome is now
    # rendered by the StartupPanel banner.
    def _maybe_show_welcome(self) -> None:
        return

    @work
    async def bootstrap_core(self) -> None:
        """Auto-start Core (or detect it's already running), then connect.

        Behavior:
        - ensure_core_started returns CoreStartOk -> connect_core()
        - returns CoreStartFailed with missing_deepseek_key -> enter setup mode
        - returns other CoreStartFailed -> show diagnostics, keep prompt disabled
        """
        # P5-1: transition to connecting; AppHeader + terminal title reflect
        # this. We no longer dump "checking core at host:port" or
        # "starting automatically..." status strings into the chat timeline.
        self._update_startup_state(STARTUP_STATE_CONNECTING)

        result = await ensure_core_started(self._host, self._port)

        from my_agent.core.lifecycle import CoreStartFailed, CoreStartOk

        if isinstance(result, CoreStartOk):
            await self._after_bootstrap_connect()
            return

        assert isinstance(result, CoreStartFailed)
        if result.missing_deepseek_key:
            await self._enter_setup_mode()
            return

        # Other failure: show stderr tail + foreground debug hint.
        # P5-1: this is a real diagnostic the user must see — it stays in the
        # chat timeline. State transitions to setup_error so the prompt
        # border title and terminal title reflect "setup required".
        self._update_startup_state(STARTUP_STATE_SETUP_ERROR)
        log = self.query_one("#log", RichLog)
        log.write("[bold red]core failed to start[/bold red]")
        if result.exit_code is not None:
            log.write(f"[dim]exit_code={result.exit_code}[/dim]")
        if result.stderr_tail:
            log.write("[dim]stderr:[/dim]")
            for line in result.stderr_tail.splitlines():
                log.write(f"[dim]{line}[/dim]")
        log.write("[dim]Run `my-agent-core` in the foreground to debug startup.[/dim]")

    async def _enter_setup_mode(self) -> None:
        """Open the unified first-run model and API-key configuration dialog."""
        self._setup_mode = True
        self._setup_reason = "DeepSeek API key is not configured."
        # P5-1: setup mode is a SETUP_ERROR flavor of the startup state —
        # the user must configure something before the app is usable.
        self._update_startup_state(STARTUP_STATE_SETUP_ERROR)
        log = self.query_one("#log", RichLog)
        log.write("[bold yellow]DeepSeek API key is not configured.[/bold yellow]")
        log.write("[dim]Configure the model and API key in the dialog.[/dim]")
        self._set_prompt_enabled(False)
        self._open_settings_dialog(initial=True)

    def _open_settings_dialog(self, *, initial: bool = False) -> None:
        """Open settings and remember whether cancellation should exit TUI."""
        self._setup_mode = initial
        self._set_prompt_enabled(False)
        self.push_screen(
            SettingsDialog(tuple(DEEPSEEK_MODEL_OPTIONS), self._current_model),
            lambda result: self._on_settings_result(result, initial=initial),
        )

    def _on_settings_result(
        self,
        result: SettingsSelection | None,
        *,
        initial: bool,
    ) -> None:
        if result is None:
            self._setup_mode = False
            if initial:
                self.exit()
            else:
                self._set_prompt_enabled(not self._busy)
                self._focus_prompt()
            return
        self._apply_settings(result, initial=initial)

    @work
    async def _apply_settings(self, result: SettingsSelection, *, initial: bool) -> None:
        env_file = self._global_env_file or Path("~/.my-agent/.env").expanduser()
        try:
            save_deepseek_api_key(result.api_key, env_file=env_file)
            self._current_model = save_deepseek_model(result.model)
        except (OSError, ValueError) as exc:
            self.query_one("#log", RichLog).write(
                f"[bold red]settings save failed:[/bold red] {escape(str(exc))}"
            )
            if initial:
                self.exit()
            else:
                self._setup_mode = False
                self._set_prompt_enabled(not self._busy)
            return

        if initial:
            self._setup_mode = False
            self._clear_prompt()
            self.query_one("#log", RichLog).write(
                "[bold green]success[/bold green] — "
                "DeepSeek model and API key configured."
            )
            self.bootstrap_core()
            return

        self._setup_mode = False
        log = self.query_one("#log", RichLog)
        log.write(
            "[bold green]settings saved[/bold green] — "
            "model and API key updated. Restart Core to apply the new key."
        )
        if self._client is not None:
            try:
                await self._client.send_command("config.model.set", {"model": result.model})
            except IpcError:
                pass
        self._set_prompt_enabled(not self._busy)
        self._focus_prompt()

    async def _exit_setup_mode_and_rebootstrap(self) -> None:
        self._setup_mode = False
        # Hide setup Input, restore TextArea prompt.
        setup_prompt = self.query_one("#setup-prompt", Input)
        setup_prompt.display = False
        setup_prompt.disabled = True
        setup_prompt.value = ""
        self._set_prompt_enabled(False)
        self._clear_prompt()
        # Re-run bootstrap
        self.bootstrap_core()

    async def _after_bootstrap_connect(self) -> None:
        """Called after bootstrap returns Ok; just delegates to connect_core."""
        # P5-1: do NOT write "connecting to host:port..." to the chat
        # timeline — that is low-level infrastructure status now surfaced
        # via AppHeader's connection label. Just transition state and connect.
        self._update_startup_state(STARTUP_STATE_CREATING_SESSION)
        self.connect_core()

    # ---- P5-2: copy helpers ----

    def _copy_to_system_clipboard(self, text: str) -> None:
        """Copy text to the OS clipboard using lightweight native commands.

        Raises RuntimeError if no suitable command is available or it fails.
        This avoids depending on heavyweight GUI/clipboard packages.
        """
        system = platform.system()
        if system == "Windows":
            clip_command = shutil.which("clip.exe") or shutil.which("clip") or "clip.exe"
            subprocess.run(
                [clip_command], input=text, text=True, check=True, timeout=5
            )
            return
        if system == "Darwin":
            subprocess.run(
                ["pbcopy"], input=text, text=True, check=True, timeout=5
            )
            return

        # Linux: prefer wl-copy (Wayland), fall back to xclip.
        if shutil.which("wl-copy"):
            subprocess.run(
                ["wl-copy"], input=text, text=True, check=True, timeout=5
            )
            return
        if shutil.which("xclip"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text,
                text=True,
                check=True,
                timeout=5,
            )
            return

        raise RuntimeError("no native clipboard command found")

    def _do_copy(self, text: str, label: str) -> None:
        """Copy text to the OS clipboard and report the result in RichLog."""
        log = self.query_one("#log", RichLog)
        if not text.strip():
            log.write(f"[italic yellow]nothing to copy ({label})[/italic yellow]")
            return

        try:
            self._copy_to_system_clipboard(text)
        except Exception:
            log.write(
                f"[bold red]copy failed:[/bold red] could not copy {label} "
                "to the system clipboard."
            )
        else:
            log.write(f"[dim]copied {label} to clipboard[/dim]")

    def _copy_last_exchange(self) -> None:
        """Copy the most recent user question and assistant answer together."""
        if self._current_llm is not None:
            assistant_text = self._current_llm.text
        else:
            assistant_text = self._last_assistant_text
        if not assistant_text:
            self._do_copy("", "last exchange")
            return
        if self._last_user_text:
            text = f"User:\n{self._last_user_text}\n\nAssistant:\n{assistant_text}"
        else:
            text = f"Assistant:\n{assistant_text}"
        self._do_copy(text, "last question and answer")

    # ---- P3-0: prompt helpers ----

    def _prompt(self) -> TextArea:
        return self.query_one("#prompt", TextArea)

    def _prompt_text(self) -> str:
        return self._prompt().text

    def _set_prompt_text(self, text: str) -> None:
        prompt = self._prompt()
        prompt.load_text(text)
        lines = text.split("\n")
        prompt.move_cursor((len(lines) - 1, len(lines[-1])))

    def _clear_prompt(self) -> None:
        self._prompt().load_text("")

    def _set_prompt_enabled(self, enabled: bool) -> None:
        self._prompt().disabled = not enabled

    def _focus_prompt(self) -> None:
        self._prompt().focus()

    # ---- Setup mode: dedicated Input handles password masking ----

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Only fires for Input widgets (setup-prompt), not TextArea."""
        if event.input.id != "setup-prompt":
            return
        if not self._setup_mode:
            return

        content = event.value.strip()
        if not content:
            return
        log = self.query_one("#log", RichLog)
        try:
            env_file = self._global_env_file or Path("~/.my-agent/.env").expanduser()
            save_deepseek_api_key(content, env_file=env_file)
        except ValueError:
            log.write("[italic yellow]API key cannot be empty[/italic yellow]")
            event.input.value = ""
            return
        except OSError as exc:
            log.write(f"[bold red]error:[/bold red] failed to save key: {exc}")
            event.input.value = ""
            return
        log.write(f"[dim]API key saved to {env_file}[/dim]")
        await self._exit_setup_mode_and_rebootstrap()

    # ---- Normal prompt submit path (TextArea -> Enter) ----

    async def _handle_model_command(self, content: str) -> None:
        """Show or change the DeepSeek model for subsequent runs."""
        palette = self.query_one("#slash-palette", SlashPalette)
        palette.close()
        self._clear_prompt()
        self._leave_welcome_state()
        log = self.query_one("#log", RichLog)

        requested = content.removeprefix("/model").strip()
        if not requested:
            current = "unknown"
            if self._client is not None:
                try:
                    result = await self._client.send_command("config.model.get", {})
                    current = str(result.get("model", current))
                except IpcError as exc:
                    log.write(f"[bold red]model error:[/bold red] {escape(str(exc))}")
                    return
            model_select = self.query_one("#model-select", ModelSelect)
            model_select.open(tuple(DEEPSEEK_MODEL_OPTIONS), current)
            self._set_prompt_enabled(False)
            model_select.focus()
            return

        selected_model = normalize_deepseek_model(requested)
        if selected_model is None:
            log.write(
                "[bold yellow]unknown model.[/bold yellow] "
                "Use /model to list DeepSeek V4 Pro and V4 Flash."
            )
            return
        if self._client is None:
            log.write("[italic yellow]not connected yet[/italic yellow]")
            return
        try:
            result = await self._client.send_command(
                "config.model.set", {"model": selected_model}
            )
        except IpcError as exc:
            log.write(f"[bold red]model switch failed:[/bold red] {escape(str(exc))}")
            return
        description = str(result.get("description", DEEPSEEK_MODEL_OPTIONS[selected_model]))
        self.query_one("#run-status", RunStatusBlock).set_model(selected_model, "static")
        log.write(
            f"[bold green]model switched:[/bold green] {selected_model} — {description}"
        )

    def _close_model_picker(self) -> None:
        self.query_one("#model-select", ModelSelect).close()
        self._set_prompt_enabled(not self._busy and not self._setup_mode)
        if not self._busy:
            self._focus_prompt()

    @work
    async def _select_model_from_picker(self) -> None:
        model_select = self.query_one("#model-select", ModelSelect)
        selected_model = model_select.selected_model
        self._close_model_picker()
        await self._handle_model_command(f"/model {selected_model}")

    async def _submit_prompt_text(self, raw_content: str) -> None:
        """Route submitted TextArea content through confirmation and send paths."""
        palette = self.query_one("#slash-palette", SlashPalette)
        content = raw_content.strip()
        if content in ("/tool", "/tools", "/tool-details"):
            palette.close()
            # P5-2.1: slash output belongs in the chat timeline, not under the banner.
            self._leave_welcome_state()
            self._toggle_latest_tool_details(visible=True)
            self._clear_prompt()
            self._focus_prompt()
            return

        if content == "/copy":
            palette.close()
            self._leave_welcome_state()
            self._clear_prompt()
            self._focus_prompt()
            self._copy_last_exchange()
            return

        if content == "/model" or content.startswith("/model "):
            await self._handle_model_command(content)
            return

        if content == "/settings":
            palette.close()
            self._clear_prompt()
            self._open_settings_dialog()
            return

        if content == "/init":
            palette.close()
            self._clear_prompt()
            self._focus_prompt()

            await self._add_chat("user", content)

            log = self.query_one("#log", RichLog)
            from my_agent.core.config import get_config
            from my_agent.core.memory.project_init import (
                ProjectInitService,
                create_init_provider,
            )

            def _report_progress(msg: str) -> None:
                log.write(f"[dim]{msg}[/dim]")

            config = get_config()
            provider = create_init_provider(config.llm.default_model)
            service = ProjectInitService(
                Path.cwd(),
                provider=provider,
                progress_callback=_report_progress,
            )
            status = self.query_one("#run-status", RunStatusBlock)
            status.set_activity("analyzing project")
            try:
                result = await service.run()
                status.finish("success")
                for msg in result.messages:
                    log.write(msg)
                # 展示生成的 AGENTS.md 内容
                agents_path = Path.cwd() / "AGENTS.md"
                if agents_path.exists():
                    log.write("[bold]AGENTS.md[/bold]")
                    log.write(
                        Markdown(agents_path.read_text(encoding="utf-8"))
                    )
            except Exception as exc:
                status.finish("failed")
                log.write(f"[bold red]init failed:[/bold red] {escape(str(exc))}")
            return

        if palette.display:
            self._accept_slash_selection()
            return

        # P2-0/P2-1: permission mode shortcuts
        if self._active_permission is not None:
            if len(content) == 1 and content in _DECISION_MAP:
                self._submit_permission_decision(_DECISION_MAP[content])
                return

            if not content:
                ps = self.query_one("#permission-select", PermissionSelect)
                if ps.display:
                    self._submit_permission_decision(ps.selected_decision)
                    return

            decision = _DECISION_MAP.get(content.lower())
            log = self.query_one("#log", RichLog)
            if decision is None:
                log.write(
                    "[italic yellow]choose with \u2191/\u2193 + Enter,"
                    " or press y/a/n/d[/italic yellow]"
                )
                self._clear_prompt()
                return

            self._submit_permission_decision(decision)
            return

        if not content:
            return

        if content == "/":
            self._open_slash_palette()
            self._clear_prompt()
            return

        await self._add_chat("user", content)
        self._clear_prompt()

        log = self.query_one("#log", RichLog)
        if self._client is None or self._session_id is None:
            log.write("[italic yellow]not connected yet[/italic yellow]")
            return
        if content.startswith("/compact"):
            if self._busy:
                log.write("[italic yellow]agent is still working[/italic yellow]")
                return

            focus = content.removeprefix("/compact").strip()
            self._busy = True
            self._awaiting_local_run_started = False
            self._set_prompt_enabled(False)
            # P5-1: /compact is a run-like activity; show "working" state.
            self._update_startup_state(STARTUP_STATE_RUNNING)
            self.query_one("#run-status", RunStatusBlock).set_activity("compacting")
            self.compact_session(focus)
            return
        if self._busy:
            log.write("[italic yellow]agent is still working[/italic yellow]")
            return

        self._busy = True
        self._current_llm = None
        self._awaiting_local_run_started = True
        self._set_prompt_enabled(False)
        # P5-1: user submitted a message — flip to RUNNING. Terminal title
        # gets "working" suffix; prompt border title becomes "agent is working".
        self._update_startup_state(STARTUP_STATE_RUNNING)
        self.query_one("#run-status", RunStatusBlock).set_activity("working")
        self.send_message(content)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt":
            return
        if self._active_permission is not None:
            return

        content = event.text_area.text
        palette = self.query_one("#slash-palette", SlashPalette)

        # P3-0: only trigger slash palette for single-line content starting with /
        is_single_line = "\n" not in content
        if is_single_line and content.startswith("/") and " " not in content:
            commands = self._load_slash_commands()
            if not palette.display:
                palette.open(commands, content)
            else:
                palette.filter(content)
            return

        if palette.display:
            palette.close()

    def _show_permission(self, permission: PendingPermission) -> None:
        """Display a single permission request in PermissionSelect and status."""
        self._active_permission = permission
        status = self.query_one("#run-status", RunStatusBlock)
        status.wait_permission(permission.tool_name)
        status.set_activity("permission", permission.tool_name)
        # P5-1: transition startup state so terminal title and prompt border
        # title reflect "permission required".
        self._update_startup_state(STARTUP_STATE_WAITING_PERMISSION)
        self.query_one("#permission-select", PermissionSelect).open(
            permission.tool_name,
            permission.param_preview,
            queued_count=len(self._permission_queue),
        )
        self._set_prompt_enabled(True)
        self._clear_prompt()
        self.query_one("#permission-select", PermissionSelect).focus()

    def _queue_permission(self, permission: PendingPermission) -> None:
        """Add a permission request. Shows immediately if none active, else queues."""
        if self._active_permission is None:
            self._show_permission(permission)
        else:
            self._permission_queue.append(permission)
            # Refresh queued count on open PermissionSelect
            ps = self.query_one("#permission-select", PermissionSelect)
            if ps.display:
                ps.update_queued_count(len(self._permission_queue))

    def _advance_permission_queue(self) -> None:
        """Move to next queued permission, or close if queue is empty."""
        if self._permission_queue:
            next_perm = self._permission_queue.popleft()
            self._show_permission(next_perm)
        else:
            self._active_permission = None
            ps = self.query_one("#permission-select", PermissionSelect)
            ps.close()
            status = self.query_one("#run-status", RunStatusBlock)
            status.clear_permission()
            status.set_activity("running")
            # P5-1: if we're still busy with a run, go back to RUNNING; if
            # the run has finished (or we never started one), go to READY.
            if self._busy:
                self._update_startup_state(STARTUP_STATE_RUNNING)
            else:
                self._update_startup_state(STARTUP_STATE_READY)
            self._set_prompt_enabled(True)
            self._clear_prompt()
            self._focus_prompt()

    def _clear_permission_state_after_error(self) -> None:
        """Close permission UI after a failed permission response."""
        self._active_permission = None
        self._permission_queue.clear()
        self.query_one("#run-status", RunStatusBlock).clear_permission()
        self.query_one("#permission-select", PermissionSelect).close()
        self._set_prompt_enabled(True)
        self._focus_prompt()

    def _clear_permission_state(self) -> None:
        """Clear active and queued permission UI state."""
        self._active_permission = None
        self._permission_queue.clear()
        self.query_one("#run-status", RunStatusBlock).clear_permission()
        self.query_one("#permission-select", PermissionSelect).close()

    def _drop_permission(self, tool_use_id: str) -> None:
        """Remove a permission request that Core has already resolved."""
        if self._active_permission is not None:
            if self._active_permission.tool_use_id == tool_use_id:
                self._active_permission = None
                self._advance_permission_queue()
                return

        original_len = len(self._permission_queue)
        self._permission_queue = deque(
            permission
            for permission in self._permission_queue
            if permission.tool_use_id != tool_use_id
        )
        if len(self._permission_queue) != original_len:
            ps = self.query_one("#permission-select", PermissionSelect)
            if ps.display:
                ps.update_queued_count(len(self._permission_queue))

    def _submit_permission_decision(self, decision: str) -> bool:
        """Respond to current active permission, then advance queue.

        Returns False if there is no active permission (idempotent guard).
        """
        if self._active_permission is None:
            return False

        permission = self._active_permission
        self._active_permission = None
        log = self.query_one("#log", RichLog)
        target = permission.tool_name
        if permission.param_preview:
            target += f" {permission.param_preview}"
        log.write(f"[dim]permission decision  {escape(target)} -> {escape(decision)}[/dim]")
        self.respond_permission(permission.tool_use_id, decision)
        self._advance_permission_queue()
        return True

    def _handle_permission_key(self, key: str) -> bool:
        permission_select = self.query_one("#permission-select", PermissionSelect)
        if self._active_permission is None or not permission_select.display:
            return False
        if key == "up":
            permission_select.move(-1)
            return True
        if key == "down":
            permission_select.move(1)
            return True
        if key == "enter":
            self._submit_permission_decision(permission_select.selected_decision)
            return True
        if key in _DECISION_MAP:
            decision = permission_select.select_key(key)
            if decision is not None:
                self._submit_permission_decision(decision)
                return True
        return False

    def _handle_model_key(self, key: str) -> bool:
        model_select = self.query_one("#model-select", ModelSelect)
        if not model_select.display:
            return False
        if key == "up":
            model_select.move(-1)
        elif key == "down":
            model_select.move(1)
        elif key == "enter":
            self._select_model_from_picker()
        elif key == "escape":
            self._close_model_picker()
        return True

    def on_key(self, event: Key) -> None:
        if isinstance(self.screen, SettingsDialog) and self.screen.handle_model_key(event.key):
            event.stop()
            return
        # Setup mode Esc handling: cancel setup, show hint, do not retry bootstrap.
        if self._setup_mode and event.key == "escape":
            self._setup_mode = False
            log = self.query_one("#log", RichLog)
            log.write(
                "[dim]setup cancelled; configure ~/.my-agent/.env manually "
                "or run my-agent-core to debug[/dim]"
            )
            setup_prompt = self.query_one("#setup-prompt", Input)
            setup_prompt.display = False
            setup_prompt.disabled = True
            setup_prompt.value = ""
            event.stop()
            return

        # P2-0/P3-0: PermissionSelect takes priority over slash palette and TextArea.
        if self._handle_permission_key(event.key):
            event.stop()
            return

        if self._handle_model_key(event.key):
            event.stop()
            return

        # P3-0: Ctrl+Enter / Alt+Enter submit is handled by App-level BINDINGS
        # (action_submit_prompt), which is more reliable than on_key for this.
        # See BINDINGS above.

        palette = self.query_one("#slash-palette", SlashPalette)
        if not palette.display:
            return

        if event.key == "up":
            palette.move(-1)
            event.stop()
        elif event.key == "down":
            palette.move(1)
            event.stop()
        elif event.key == "escape":
            palette.close()
            event.stop()
        elif event.key == "enter":
            self._accept_slash_selection()
            event.stop()

    @work
    async def connect_core(self) -> None:
        client = SocketClient(self._host, self._port)
        try:
            await client.connect()
        except (ConnectionRefusedError, OSError) as e:
            self.post_message(CoreDisconnected(str(e), transport_lost=True))
            return

        self._client = client
        client.on_event(self._handle_core_event)
        loop_task = asyncio.create_task(client.run_event_loop())

        try:
            await client.send_command(
                "event.subscribe",
                {
                    "type": "event.subscribe",
                    "topics": [
                        "session.*",
                        "run.*",
                        "step.*",
                        "tool.*",
                        "llm.*",
                        "permission.*",
                        "skill.*",
                        "context.*",
                        "subagent.*",
                        "task.*",
                        "scheduler.*",
                    ],
                    "scope": "global",
                },
            )

            created = await client.send_command(
                "session.create",
                {
                    "type": "session.create",
                    "mode": "chat",
                    "title": "",
                    "workspace_root": str(Path.cwd()),
                    "client_type": "tui",
                },
            )

            session_id = str(created["session_id"])

            self._session_id = session_id
            self.post_message(CoreConnected(session_id))

            await loop_task
            # run_event_loop returns when the Core socket closes. Treat that
            # as a normal disconnect so the TUI remains open and refreshes its
            # status instead of silently staying in connected/ready state.
            if not loop_task.cancelled():
                self.post_message(
                    CoreDisconnected("Core connection closed", transport_lost=True)
                )

        except IpcError as e:
            if isinstance(e.data, dict) and e.data.get("code") == "workspace_in_use":
                active_root = str(e.data.get("active_workspace_root", "unknown"))
                self._setup_reason = (
                    "This project is already open in another TUI.\n"
                    f"Overlapping active workspace: {active_root}\n"
                    "Close the other TUI or choose a non-overlapping project."
                )
                self._update_startup_state(STARTUP_STATE_WORKSPACE_IN_USE)
                log = self.query_one("#log", RichLog)
                log.write(f"[bold red]workspace in use[/bold red] {self._setup_reason}")
            else:
                self.post_message(CoreDisconnected(str(e)))
        except Exception as e:
            # See send_message for rationale: never let @work swallow errors.
            log = self.query_one("#log", RichLog)
            log.write(f"[bold red]error:[/bold red] {type(e).__name__}: {e}")
            self.post_message(CoreDisconnected(f"{type(e).__name__}: {e}"))
        finally:
            if not loop_task.done():
                loop_task.cancel()
                with suppress(asyncio.CancelledError):
                    await loop_task
            await client.close()
            self._client = None

    @work
    async def send_heartbeat(self) -> None:
        session_id = self._session_id
        if session_id is None or self._startup_state in (
            STARTUP_STATE_DISCONNECTED,
            STARTUP_STATE_SETUP_ERROR,
            STARTUP_STATE_WORKSPACE_IN_USE,
        ):
            return

        client = SocketClient(self._host, self._port)
        loop_task: asyncio.Task[None] | None = None
        try:
            await client.connect()
            loop_task = asyncio.create_task(client.run_event_loop())
            await client.send_command(
                "session.heartbeat",
                {"type": "session.heartbeat", "session_id": session_id},
            )
        except IpcError as e:
            if isinstance(e.data, dict) and e.data.get("code") == "session_stale":
                self._setup_reason = (
                    "This TUI session expired because Core stopped receiving heartbeats. "
                    "Please restart the TUI."
                )
                self._session_id = None
                self._update_startup_state(STARTUP_STATE_DISCONNECTED)
        except (ConnectionError, OSError):
            return
        finally:
            if loop_task is not None and not loop_task.done():
                loop_task.cancel()
                with suppress(asyncio.CancelledError):
                    await loop_task
            await client.close()

    @work
    async def send_message(self, content: str) -> None:
        if self._client is None or self._session_id is None:
            return

        try:
            result = await self._client.send_command(
                "session.send_message",
                {
                    "type": "session.send_message",
                    "session_id": self._session_id,
                    "content": content,
                },
            )
            run_id = result.get("run_id")
            if run_id:
                self._accepted_run_ids.add(str(run_id))
        except IpcError as e:
            self._busy = False
            self._awaiting_local_run_started = False
            self._set_prompt_enabled(True)
            self._focus_prompt()
            self.post_message(CoreDisconnected(str(e)))
        except Exception as e:
            # Non-IpcError exception (e.g. ConnectionResetError, RuntimeError).
            # Without this catch, Textual @work silently swallows the error,
            # leaving `self._busy = True` and prompt disabled forever — which
            # matches the symptom "input + Enter does nothing after first send".
            self._busy = False
            self._awaiting_local_run_started = False
            self._set_prompt_enabled(True)
            self._focus_prompt()
            log = self.query_one("#log", RichLog)
            log.write(f"[bold red]error:[/bold red] {type(e).__name__}: {e}")
            self.post_message(CoreDisconnected(f"{type(e).__name__}: {e}"))

    @work
    async def respond_permission(self, tool_use_id: str, decision: str) -> None:
        client = SocketClient(self._host, self._port)
        loop_task: asyncio.Task[None] | None = None
        try:
            await client.connect()
            loop_task = asyncio.create_task(client.run_event_loop())
            await client.send_command(
                "permission.respond",
                {
                    "type": "permission.respond",
                    "tool_use_id": tool_use_id,
                    "decision": decision,
                },
            )
        except IpcError as e:
            self._clear_permission_state_after_error()
            self.post_message(CoreDisconnected(str(e)))
        except Exception as e:
            # See send_message for rationale: never let @work swallow errors.
            self._clear_permission_state_after_error()
            log = self.query_one("#log", RichLog)
            log.write(f"[bold red]error:[/bold red] {type(e).__name__}: {e}")
            self.post_message(CoreDisconnected(f"{type(e).__name__}: {e}"))
        finally:
            if loop_task is not None and not loop_task.done():
                loop_task.cancel()
                with suppress(asyncio.CancelledError):
                    await loop_task
            await client.close()

    @work
    async def compact_session(self, focus: str) -> None:
        if self._client is None or self._session_id is None:
            return

        try:
            result = await self._client.send_command(
                "session.compact",
                {
                    "type": "session.compact",
                    "session_id": self._session_id,
                    "focus": focus,
                },
            )
            summary_tokens = result.get("summary_tokens", 0)
            saved_tokens = result.get("saved_tokens", 0)

            # P1-0: /compact is a user-initiated command; its result must stay visible.
            await self._add_event(
                "system",
                f"context compacted summary={summary_tokens} saved~={saved_tokens}",
                visible=True,
            )
        except IpcError as e:
            self.post_message(CoreDisconnected(str(e)))
        except Exception as e:
            # See send_message for rationale: never let @work swallow errors.
            log = self.query_one("#log", RichLog)
            log.write(f"[bold red]error:[/bold red] {type(e).__name__}: {e}")
            self.post_message(CoreDisconnected(f"{type(e).__name__}: {e}"))
        finally:
            self._busy = False
            self._set_prompt_enabled(True)
            self._focus_prompt()
            # P5-1: compact finished — back to READY (terminal title drops
            # "working" suffix, prompt border title returns to "type message").
            self._update_startup_state(STARTUP_STATE_READY)

    async def _handle_core_event(self, event: dict[str, Any]) -> None:
        self.post_message(CoreEvent(event))

    def on_core_connected(self, event: CoreConnected) -> None:
        # P5-2.1: do not write the raw session id into the chat timeline.
        self._session_id = event.session_id
        # P5-1: transition to READY. This drives the AppHeader, terminal
        # title, and prompt border title in one place.
        self._update_startup_state(STARTUP_STATE_READY)
        # Bootstrap success: enable prompt for normal chat.
        self._set_prompt_enabled(True)
        self._focus_prompt()

    def on_core_disconnected(self, event: CoreDisconnected) -> None:
        log = self.query_one("#log", RichLog)
        log.write(f"[bold red]disconnected[/bold red] {event.reason}")
        if event.transport_lost:
            self._session_id = None
            self._busy = False
            self._awaiting_local_run_started = False
            self._clear_permission_state()
            self._set_prompt_enabled(False)
        # P5-1: transition to DISCONNECTED so the AppHeader, terminal title,
        # and prompt border title all reflect the connection loss.
        self._update_startup_state(STARTUP_STATE_DISCONNECTED)

    def _should_handle_core_event(self, event: dict[str, Any]) -> bool:
        """Return whether this TUI session should render a pushed Core event."""
        session_id = event.get("session_id")
        if session_id is not None and self._session_id is not None:
            return str(session_id) == self._session_id

        parent_run_id = event.get("parent_run_id")
        if parent_run_id is not None:
            return str(parent_run_id) in self._accepted_run_ids

        run_id = event.get("run_id")
        if run_id is None:
            return True

        run_id_text = str(run_id)
        if run_id_text in self._accepted_run_ids:
            return True

        if event.get("type") == "run.started" and self._awaiting_local_run_started:
            self._accepted_run_ids.add(run_id_text)
            return True

        return False

    async def on_core_event(self, event: CoreEvent) -> None:
        event_type = event.event.get("type", "unknown")
        log = self.query_one("#log", RichLog)
        if not self._should_handle_core_event(event.event):
            return
        if event_type == "llm.token":
            token = str(event.event.get("token", ""))
            llm_block = await self._ensure_llm_block()
            llm_block.append_token(token)
            # P2-0.6.2: keep streaming tokens buffered while the status strip
            # shows "answering". Rendering partial Markdown/CJK text into the
            # RichLog caused visible reflow, narrow-column fragments, and
            # "sudden block completion" during long answers. We render the
            # complete assistant message once on run.finished.
            self.query_one("#run-status", RunStatusBlock).set_activity("answering")
            return

        if event_type == "llm.usage":
            # P1-1: low-noise status update; do NOT write to RichLog.lines
            self.query_one("#run-status", RunStatusBlock).set_usage(
                input_tokens=int(event.event.get("input_tokens", 0)),
                output_tokens=int(event.event.get("output_tokens", 0)),
                context_pct=float(event.event.get("context_pct", 0.0)),
                cache_read_input_tokens=int(event.event.get("cache_read_input_tokens", 0)),
                cache_creation_input_tokens=int(event.event.get("cache_creation_input_tokens", 0)),
            )
            return

        if event_type == "llm.model_selected":
            # P1-2: model name is run status, NOT chat log
            model = str(event.event.get("model", ""))
            strategy = str(event.event.get("strategy", ""))
            self.query_one("#run-status", RunStatusBlock).set_model(model, strategy)
            return

        if event_type == "permission.requested":
            tool_name = str(event.event.get("tool_name", "unknown"))
            tool_use_id = str(event.event.get("tool_use_id", ""))
            param_preview = str(event.event.get("param_preview", ""))

            permission = PendingPermission(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                param_preview=param_preview,
            )
            self._queue_permission(permission)

            # P1-0: permission prompts are user-actionable, keep them visible.
            await self._add_event(
                "permission",
                f"{tool_name} {param_preview}",
                visible=True,
            )
            return

        if event_type in ("permission.granted", "permission.denied"):
            tool_use_id = str(event.event.get("tool_use_id", ""))
            if tool_use_id:
                self._drop_permission(tool_use_id)
            return

        if event_type == "tool.call_started":
            tool_name = str(event.event.get("tool_name", "unknown"))
            params = event.event.get("params", {})
            if not isinstance(params, dict):
                params = {}

            tool_use_id = str(
                event.event.get("tool_use_id") or f"{tool_name}:{len(self._tool_blocks)}"
            )
            status = self.query_one("#run-status", RunStatusBlock)
            status.tool_started()
            status.set_activity("tool", tool_name)
            await self._add_tool_block(tool_use_id, tool_name, params)
            return

        if event_type == "tool.call_finished":
            tool_name = str(event.event.get("tool_name", "unknown"))
            tool_use_id = str(
                event.event.get("tool_use_id") or f"{tool_name}:{len(self._tool_blocks) - 1}"
            )
            elapsed_ms = int(event.event.get("elapsed_ms") or 0)

            tool_block = self._tool_blocks.get(tool_use_id)
            if tool_block is not None:
                tool_block.set_result(elapsed_ms)
                self._render_tool_log_block(tool_use_id)
            else:
                log.write(f"[dim]tool {tool_name} done {elapsed_ms}ms[/dim]")
            status = self.query_one("#run-status", RunStatusBlock)
            status.tool_finished()
            status.set_activity("running")
            return

        if event_type == "tool.call_failed":
            tool_name = str(event.event.get("tool_name", "unknown"))
            tool_use_id = str(
                event.event.get("tool_use_id") or f"{tool_name}:{len(self._tool_blocks) - 1}"
            )
            elapsed_ms = int(event.event.get("elapsed_ms") or 0)
            error_message = str(event.event.get("error_message", ""))

            tool_block = self._tool_blocks.get(tool_use_id)
            if tool_block is not None:
                tool_block.set_result(
                    elapsed_ms,
                    is_error=True,
                    error_message=error_message,
                )
                self._render_tool_log_block(tool_use_id)
            else:
                log.write(f"[red]tool {tool_name} failed {error_message}[/red]")
            status = self.query_one("#run-status", RunStatusBlock)
            status.tool_finished(failed=True)
            status.set_activity("running")
            return

        if event_type == "run.finished":
            # P2-0.6.2: cancel any legacy pending preview and render once with
            # the complete text. This avoids showing half-finished Markdown or
            # repeatedly re-wrapping long CJK paragraphs while the model is
            # still producing tokens.
            await self._cancel_llm_render()
            if self._current_llm is not None:
                text = self._current_llm.text
                if text and text != self._llm_last_render_text:
                    self._llm_last_render_text = text
                    wrap_width = self._stream_wrap_width()
                    wrapped = _soft_wrap_stream_text(text, wrap_width)
                    self._render_llm_log_line(wrapped)
                self._current_llm.finalize_markdown()
                # P5-2: remember the latest finalized assistant text for /copy.
                if text:
                    self._last_assistant_text = text
            self._current_llm = None
            self._busy = False
            self._awaiting_local_run_started = False
            self._clear_permission_state()
            self._set_prompt_enabled(True)
            self._focus_prompt()
            status = event.event.get("status", "")
            self.query_one("#run-status", RunStatusBlock).finish(str(status))
            # P5-1: run finished — go back to READY. Terminal title drops
            # "working"; prompt border title becomes "type message".
            self._update_startup_state(STARTUP_STATE_READY)
            entry = run_finished_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if event_type == "run.started":
            run_id = event.event.get("run_id", "")
            if run_id:
                self._accepted_run_ids.add(str(run_id))
            self._awaiting_local_run_started = False
            status = self.query_one("#run-status", RunStatusBlock)
            status.start(str(run_id))
            status.set_activity("running")
            # P1-0: run.started is low-level noise, keep it invisible by default.
            await self._add_event("run", f"started: {run_id}")
            return

        if event_type == "step.started":
            # P1-4: update step progress in status strip; do NOT write to chat log.
            step = int(event.event.get("step", 0))
            status = self.query_one("#run-status", RunStatusBlock)
            status.step_started(step)
            status.set_activity("thinking")
            return

        if event_type == "step.finished":
            # P1-4: update step progress in status strip; do NOT write to chat log.
            step = int(event.event.get("step", 0))
            status = self.query_one("#run-status", RunStatusBlock)
            status.step_finished(step)
            status.set_activity("running")
            return

        if event_type == "skill.invoked":
            # P1-0: skill.invoked is low-level noise, keep it invisible by default.
            entry = skill_invoked_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if event_type == "skill.tool_compatibility":
            # P6: structured diagnostics for third-party Skill allowed_tools.
            # Visible only when there is something to report (aliases or unresolved
            # tools); the SessionManager only emits this event when has_diagnostics
            # is True, so we render it as a compact visible line.
            entry = skill_tool_compatibility_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if event_type == "subagent.started":
            # P1-3: child agent started — visible so user knows work is delegated.
            child_run_id = str(event.event.get("run_id", ""))
            if child_run_id:
                self._accepted_run_ids.add(child_run_id)
            entry = subagent_started_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if event_type == "subagent.finished":
            # P1-3: child agent finished — visible. Does NOT touch parent run state.
            entry = subagent_finished_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if event_type == "context.compacted":
            # P1-2: context compaction is an important state change — visible.
            entry = context_compacted_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if event_type == "context.compaction_failed":
            # P1-2: compaction failure must be visible so the user knows.
            entry = context_compaction_failed_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if isinstance(event_type, str) and event_type.startswith("task."):
            entry = task_event_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        if isinstance(event_type, str) and event_type.startswith("scheduler."):
            entry = scheduler_event_entry(event.event)
            await self._add_event(entry.category, entry.content, visible=entry.visible)
            return

        # P1-0/G1: unrelated unknown events stay low-noise and hidden by default.
        entry = unknown_event_entry(event_type)
        await self._add_event(entry.category, entry.content, visible=entry.visible)

    # ── P2-0.6 streaming render throttle ──

    async def _legacy_schedule_llm_render(self) -> None:
        """Schedule a batched RichLog update; skip if one is already pending."""
        if self._llm_render_timer is not None:
            return
        # Fire after ~50ms — batches multiple rapid tokens into one render
        self._llm_render_timer = asyncio.create_task(self._flush_after_delay())

    async def _legacy_flush_after_delay(self) -> None:
        await asyncio.sleep(0.05)
        self._flush_llm_render()

    def _legacy_flush_llm_render(self) -> None:
        """Render current LLM block text to RichLog if changed since last flush.

        Uses CJK-aware soft-wrap before writing to RichLog, so streaming
        preview remains stable even for long Chinese text. The original
        LLMStreamBlock.text is never modified.
        """
        self._llm_render_timer = None
        if self._current_llm is None:
            return
        text = self._current_llm.text
        if text == self._llm_last_render_text:
            return
        self._llm_last_render_text = text
        wrap_width = self._stream_wrap_width()
        wrapped = _soft_wrap_stream_text(text, wrap_width)
        self._render_llm_log_line(wrapped)

    async def _legacy_cancel_and_flush_llm_render(self) -> None:
        """Cancel pending timer, wait briefly, then flush one final time."""
        if self._llm_render_timer is not None:
            self._llm_render_timer.cancel()
            self._llm_render_timer = None
        # Brief sleep to let any in-flight task settle
        await asyncio.sleep(0.06)
        self._flush_llm_render()

    async def _legacy_cancel_llm_render(self) -> None:
        """Cancel any pending streaming preview without rendering partial text."""
        if self._llm_render_timer is not None:
            self._llm_render_timer.cancel()
            self._llm_render_timer = None
        await asyncio.sleep(0)

    async def _legacy_ensure_llm_block(self) -> LLMStreamBlock:
        if self._current_llm is not None:
            return self._current_llm

        block = LLMStreamBlock()
        # Keep LLMStreamBlock as the raw-token state holder for tests and final
        # Markdown conversion, but do not render it directly. A visible mounted
        # Static inside RichLog is measured differently from RichLog text lines
        # and was the remaining source of CJK text collapsing into narrow
        # columns. The user-visible streaming preview is rendered only through
        # _render_llm_log_line().
        block.display = False
        self._current_llm = block
        self._current_llm_log_range = None
        # P2-0.6: reset throttle state for new streaming round
        self._llm_last_render_text = ""
        # Cancel any stale timer from previous stream
        if self._llm_render_timer is not None:
            self._llm_render_timer.cancel()
            self._llm_render_timer = None
        log = self.query_one("#log", RichLog)
        await log.mount(block)
        return block

    async def _legacy_add_tool_block(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolCallBlock:
        block = ToolCallBlock(tool_name, params)
        self._tool_blocks[tool_use_id] = block
        log = self.query_one("#log", RichLog)
        await log.mount(block)
        self._render_tool_log_block(tool_use_id)
        return block

    def _legacy_render_tool_log_block(self, tool_use_id: str) -> None:
        """Replace the visible RichLog snapshot for one tool call."""
        block = self._tool_blocks.get(tool_use_id)
        if block is None:
            return

        log = self.query_one("#log", RichLog)
        old_range = self._tool_log_ranges.pop(tool_use_id, None)
        if old_range is not None:
            start, end = old_range
            del log.lines[start:end]
            self._shift_log_ranges_after_deleted_range(start, end)
            log.refresh()

        start = len(log.lines)
        log.write(block.render_markup())
        self._tool_log_ranges[tool_use_id] = (start, len(log.lines))

    def _legacy_shift_log_ranges_after_deleted_range(self, start: int, end: int) -> None:
        removed = end - start
        if removed <= 0:
            return

        for tool_use_id, (range_start, range_end) in list(self._tool_log_ranges.items()):
            if range_start >= end:
                self._tool_log_ranges[tool_use_id] = (
                    range_start - removed,
                    range_end - removed,
                )

        if self._current_llm_log_range is not None:
            range_start, range_end = self._current_llm_log_range
            if range_start >= end:
                self._current_llm_log_range = (
                    range_start - removed,
                    range_end - removed,
                )

    async def _legacy_add_chat(self, role: str, content: str) -> ChatMessageBlock:
        # P5-2.1: first chat message hides the welcome banner and reveals the log.
        self._leave_welcome_state()
        block = ChatMessageBlock(role, content)
        log = self.query_one("#log", RichLog)
        if role == "user":
            log.write(f"[bold {_ELECTRIC_CYAN}]YOU //[/bold {_ELECTRIC_CYAN}]  {escape(content)}")
        else:
            log.write(f"[bold {_HOT_PINK}]AGENT //[/bold {_HOT_PINK}]  {escape(content)}")
        await log.mount(block)
        return block

    async def _legacy_add_event(
        self,
        kind: str,
        content: str,
        *,
        visible: bool = False,
    ) -> EventLineBlock:
        # P5-2.1: user-visible events (e.g. /compact results) should reveal the log
        # even if the welcome banner is currently shown.
        if visible:
            self._leave_welcome_state()
        block = EventLineBlock(kind, content)
        log = self.query_one("#log", RichLog)
        if visible:
            log.write(
                f"[bold {_HOT_PINK}]{escape(kind.upper())} //[/bold {_HOT_PINK}] "
                f"{escape(content)}"
            )
        await log.mount(block)
        return block

    def _legacy_stream_wrap_width(self) -> int:
        """Get usable display width for streaming soft-wrap."""
        # RichLog.size.width can briefly report a tiny value while Textual is
        # laying out/re-rendering the log. Trusting that transient value is what
        # creates the CJK "vertical column" effect during streaming. Prefer the
        # widest stable measurement available and then apply a generous floor.
        widths: list[int] = []

        with suppress(Exception):
            log = self.query_one("#log", RichLog)
            widths.append(int(getattr(log.size, "width", 0) or 0))

        with suppress(Exception):
            widths.append(int(getattr(self.size, "width", 0) or 0))

        with suppress(Exception):
            widths.append(int(getattr(self.console.size, "width", 0) or 0))

        return _stream_preview_width(max(widths, default=0))

    def _legacy_render_llm_log_line(self, content: str) -> None:
        """Render the current streaming assistant response into visible RichLog lines."""
        log = self.query_one("#log", RichLog)
        if self._current_llm_log_range is not None:
            start, end = self._current_llm_log_range
            del log.lines[start:end]
            log.refresh()

        start = len(log.lines)
        log.write(f"[bold {_HOT_PINK}]AGENT //[/bold {_HOT_PINK}]  {escape(content)}")
        self._current_llm_log_range = (start, len(log.lines))

    async def _schedule_llm_render(self) -> None:
        await rendering.schedule_llm_render(self)

    async def _flush_after_delay(self) -> None:
        await rendering.flush_after_delay(self)

    def _flush_llm_render(self) -> None:
        rendering.flush_llm_render(self)

    async def _cancel_and_flush_llm_render(self) -> None:
        await rendering.cancel_and_flush_llm_render(self)

    async def _cancel_llm_render(self) -> None:
        await rendering.cancel_llm_render(self)

    async def _ensure_llm_block(self) -> LLMStreamBlock:
        return cast(LLMStreamBlock, await rendering.ensure_llm_block(self))

    async def _add_tool_block(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolCallBlock:
        return cast(
            ToolCallBlock,
            await rendering.add_tool_block(self, tool_use_id, tool_name, params),
        )

    def _render_tool_log_block(self, tool_use_id: str) -> None:
        rendering.render_tool_log_block(self, tool_use_id)

    def _shift_log_ranges_after_deleted_range(self, start: int, end: int) -> None:
        rendering.shift_log_ranges_after_deleted_range(self, start, end)

    async def _add_chat(self, role: str, content: str) -> ChatMessageBlock:
        if role == "user":
            self._last_user_text = content
        return cast(ChatMessageBlock, await rendering.add_chat(self, role, content))

    async def _add_event(
        self,
        kind: str,
        content: str,
        *,
        visible: bool = False,
    ) -> EventLineBlock:
        return cast(
            EventLineBlock,
            await rendering.add_event(self, kind, content, visible=visible),
        )

    def _stream_wrap_width(self) -> int:
        return rendering.stream_wrap_width(self)

    def _render_llm_log_line(self, content: str) -> None:
        rendering.render_llm_log_line(self, content)

    def _open_slash_palette(self, query: str = "") -> None:
        commands = self._load_slash_commands()
        palette = self.query_one("#slash-palette", SlashPalette)
        palette.open(commands, query)

        if not commands:
            self._show_slash_help()

    def _accept_slash_selection(self) -> None:
        palette = self.query_one("#slash-palette", SlashPalette)
        command = palette.selected_command
        palette.close()
        if command is None:
            return

        self._set_prompt_text(command.insert_text)
        self._focus_prompt()

    def _load_slash_commands(self) -> list[SlashCommand]:
        commands = [
            SlashCommand(
                command="/compact <focus>",
                insert_text="/compact ",
                description="compact current session context",
            ),
            SlashCommand(
                command="/init",
                insert_text="/init",
                description="initialize project context (.my-agent/context.md + AGENTS.md)",
            ),
            SlashCommand(
                command="/tools",
                insert_text="/tools",
                description="toggle latest tool call details",
            ),
            SlashCommand(
                command="/copy",
                insert_text="/copy",
                description="copy the last question and answer",
            ),
            SlashCommand(
                command="/model [pro|flash]",
                insert_text="/model ",
                description="view or switch DeepSeek model",
            ),
            SlashCommand(
                command="/settings",
                insert_text="/settings",
                description="configure DeepSeek model and API key",
            ),
        ]

        try:
            loader = SkillLoader()
            skills = loader.list_all_skills()
        except Exception:
            return commands

        # 已硬编码的命令不重复添加
        hardcoded_names = {cmd.insert_text.strip() for cmd in commands}
        for skill in skills:
            skill_cmd = f"/{skill.name}"
            if skill_cmd in hardcoded_names:
                continue
            desc = skill.description.splitlines()[0] if skill.description else "run skill"
            commands.append(
                SlashCommand(
                    command=skill_cmd,
                    insert_text=f"/{skill.name} ",
                    description=desc,
                )
            )
        return commands

    def _show_slash_help(self) -> None:
        if self._slash_hint_shown:
            return

        self._slash_hint_shown = True
        log = self.query_one("#log", RichLog)
        log.write("[bold magenta]slash commands[/bold magenta]")
        log.write("[dim]/compact <focus>  compact current session context[/dim]")

        try:
            loader = SkillLoader()
            skills = loader.list_all_skills()
        except Exception as e:
            log.write(f"[red]failed to load skills: {e}[/red]")
            return

        for skill in skills:
            desc = skill.description.splitlines()[0] if skill.description else ""
            if desc:
                log.write(f"[dim]/{skill.name}  {desc}[/dim]")
            else:
                log.write(f"[dim]/{skill.name}[/dim]")
