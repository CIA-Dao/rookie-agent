from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Input, Static, TextArea

from my_agent.tui.theme import _ELECTRIC_CYAN, _HOT_PINK


@dataclass(frozen=True)
class SlashCommand:
    command: str
    insert_text: str
    description: str


class SlashPalette(Static):
    """Small selectable command list shown after typing slash."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._all_commands: list[SlashCommand] = []
        self._commands: list[SlashCommand] = []
        self._selected = 0
        self.display = False

    @property
    def selected_command(self) -> SlashCommand | None:
        if not self._commands:
            return None
        return self._commands[self._selected]

    def open(self, commands: list[SlashCommand], query: str = "") -> None:
        self._all_commands = commands
        self.filter(query)

    def filter(self, query: str) -> None:
        normalized = query.removeprefix("/").lower()
        if normalized:
            self._commands = [
                command
                for command in self._all_commands
                if normalized in command.command.removeprefix("/").lower()
            ]
        else:
            self._commands = list(self._all_commands)
        self._selected = 0
        self.display = bool(self._commands)
        self._refresh()

    def close(self) -> None:
        self.display = False

    def move(self, delta: int) -> None:
        if not self._commands:
            return
        self._selected = (self._selected + delta) % len(self._commands)
        self._refresh()

    def _refresh(self) -> None:
        if not self._commands:
            self.update("")
            return
        lines = []
        for index, command in enumerate(self._commands):
            if index == self._selected:
                prefix = f"[bold {_ELECTRIC_CYAN}]>[/bold {_ELECTRIC_CYAN}]"
                cmd = f"[bold {_HOT_PINK}]{command.command}[/bold {_HOT_PINK}]"
            else:
                prefix = " "
                cmd = f"[bold]{command.command}[/bold]"
            lines.append(f"{prefix} {cmd} [dim]{command.description}[/dim]")
        self.update("\n".join(lines))


@dataclass(frozen=True)
class PendingPermission:
    tool_use_id: str
    tool_name: str
    param_preview: str


@dataclass(frozen=True)
class PermissionOption:
    key: str
    decision: str
    label: str


class PermissionSelect(Static):
    can_focus = True

    OPTIONS: tuple[PermissionOption, ...] = (
        PermissionOption("y", "allow_once", "Allow once"),
        PermissionOption("a", "always_allow", "Always allow"),
        PermissionOption("n", "deny_once", "Deny once"),
        PermissionOption("d", "always_deny", "Always deny"),
    )

    DEFAULT_CSS = """
    PermissionSelect {
        height: 8;
        padding: 0 1;
        background: $surface;
        border: tall $primary;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._tool_name = ""
        self._param_preview = ""
        self._queued_count = 0
        self._selected = 0
        self.display = False

    @property
    def selected_decision(self) -> str:
        return self.OPTIONS[self._selected].decision

    def open(self, tool_name: str, param_preview: str = "", *, queued_count: int = 0) -> None:
        self._tool_name = tool_name
        self._param_preview = param_preview
        self._queued_count = queued_count
        self._selected = 0
        self.display = True
        self._refresh()

    def close(self) -> None:
        self.display = False
        self._tool_name = ""
        self._param_preview = ""
        self._queued_count = 0
        self._selected = 0
        self.update("")

    def update_queued_count(self, count: int) -> None:
        self._queued_count = count
        if self.display:
            self._refresh()

    def move(self, delta: int) -> None:
        self._selected = (self._selected + delta) % len(self.OPTIONS)
        self._refresh()

    def select_key(self, key: str) -> str | None:
        for index, option in enumerate(self.OPTIONS):
            if option.key == key:
                self._selected = index
                self._refresh()
                return option.decision
        return None

    def _refresh(self) -> None:
        target = self._tool_name
        if self._param_preview:
            target += f" {self._param_preview}"
        title = f"[bold {_HOT_PINK}]PERMISSION REQUIRED[/bold {_HOT_PINK}] :: {target}"
        if self._queued_count:
            title += f" [dim]({self._queued_count} more pending)[/dim]"
        lines = [title]
        for index, option in enumerate(self.OPTIONS):
            prefix = (
                f"[bold {_ELECTRIC_CYAN}]>[/bold {_ELECTRIC_CYAN}]"
                if index == self._selected
                else " "
            )
            lines.append(f"{prefix} [bold]{option.key}[/bold] {option.label}")
        self.update("\n".join(lines))


class ModelSelect(Static):
    """Keyboard-selectable DeepSeek model picker."""

    can_focus = True

    DEFAULT_CSS = """
    ModelSelect {
        dock: bottom;
        width: 72%;
        height: auto;
        max-height: 8;
        margin: 0 2 1 2;
        padding: 0 1;
        background: $surface;
        border: tall $accent;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._models: tuple[str, ...] = ()
        self._current = ""
        self._selected = 0
        self.display = False

    @property
    def selected_model(self) -> str:
        return self._models[self._selected]

    def open(self, models: tuple[str, ...], current: str) -> None:
        self._models = models
        self._current = current
        self._selected = models.index(current) if current in models else 0
        self.display = True
        self._refresh()

    def close(self) -> None:
        self.display = False
        self._models = ()
        self._current = ""
        self._selected = 0
        self.update("")

    def move(self, delta: int) -> None:
        if not self._models:
            return
        self._selected = (self._selected + delta) % len(self._models)
        self._refresh()

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


@dataclass(frozen=True)
class SettingsSelection:
    """Values submitted by the unified first-run/settings dialog."""

    model: str
    api_key: str


class SettingsDialog(ModalScreen[SettingsSelection | None]):
    """Unified DeepSeek model and API-key configuration dialog."""

    DEFAULT_CSS = """
    SettingsDialog {
        align: center middle;
        background: $background 85%;
    }

    #settings-dialog {
        width: 72;
        height: auto;
        min-height: 19;
        padding: 1 2;
        background: $panel;
        border: tall $accent;
    }

    #settings-title {
        height: 1;
        color: $accent;
        text-style: bold;
    }

    #settings-help {
        height: auto;
        margin: 1 0 0 0;
        color: $text-muted;
    }

    #settings-model {
        width: 100%;
        height: auto;
        max-height: 6;
        margin: 1 0 1 0;
        padding: 0 1;
        background: $surface;
        border: round $primary;
    }

    #settings-key {
        width: 100%;
        height: 3;
        margin: 0 0 1 0;
    }

    #settings-shortcuts {
        height: 1;
        margin: 1 0 0 0;
        color: $text-muted;
    }
    """

    def __init__(self, models: tuple[str, ...], current_model: str) -> None:
        super().__init__()
        self._models = models
        self._current_model = current_model
        self._selecting_model = False

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Settings · DeepSeek", id="settings-title"),
            Static(
                "Enter your API key first. After it passes the local check, "
                "use up/down to choose a model. Esc cancels.",
                id="settings-help",
            ),
            Input(
                placeholder="Enter DeepSeek API key...",
                password=True,
                id="settings-key",
            ),
            Static("", id="settings-error"),
            ModelSelect(id="settings-model"),
            Static("", id="settings-shortcuts"),
            id="settings-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#settings-key", Input).focus()
        self._refresh_shortcuts()

    def _refresh_shortcuts(self) -> None:
        target = (
            "[bold cyan]Enter[/bold cyan] Continue   "
            "[bold yellow]Esc[/bold yellow] Cancel"
            if not self._selecting_model
            else "[bold cyan]↑/↓[/bold cyan] Select model   "
            "[bold cyan]Enter[/bold cyan] Confirm   "
            "[bold yellow]Esc[/bold yellow] Cancel"
        )
        self.query_one("#settings-shortcuts", Static).update(target)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
            return
        if not self._selecting_model:
            return
        self.handle_model_key(event.key)
        event.stop()

    def handle_model_key(self, key: str) -> bool:
        """Handle model picker keys when the app-level handler sees them first."""
        if not self._selecting_model:
            return False
        picker = self.query_one("#settings-model", ModelSelect)
        if key == "up":
            picker.move(-1)
        elif key == "down":
            picker.move(1)
        elif key == "enter":
            self._selecting_model = False
            picker.display = False
            self.dismiss(
                SettingsSelection(
                    model=picker.selected_model,
                    api_key=self.query_one("#settings-key", Input).value.strip(),
                )
            )
        else:
            return False
        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "settings-key":
            return
        api_key = event.value.strip()
        from my_agent.core.user_config import is_valid_deepseek_api_key

        if not is_valid_deepseek_api_key(api_key):
            event.input.value = ""
            self.query_one("#settings-error", Static).update(
                "[bold yellow]格式不通过[/bold yellow]"
            )
            event.stop()
            return
        self.query_one("#settings-error", Static).update("")
        self._selecting_model = True
        picker = self.query_one("#settings-model", ModelSelect)
        picker.open(self._models, self._current_model)
        picker.focus()
        self._refresh_shortcuts()
        event.stop()


class PromptTextArea(TextArea):
    """Main prompt textarea that lets permission shortcuts win when active."""

    BINDINGS = [
        Binding("enter", "app.submit_prompt", show=False, priority=True),
        Binding("ctrl+enter", "insert_newline", show=False, priority=True),
    ]

    def action_insert_newline(self) -> None:
        self.insert("\n", maintain_selection_offset=False)

    def action_undo(self) -> None:
        """Undo safely; a Textual history error must not crash the TUI."""
        try:
            super().action_undo()
        except Exception:
            # A stale internal edit batch makes Ctrl+Z a no-op instead of
            # terminating the application.
            return

    def on_key(self, event: Key) -> None:
        handler = getattr(self.app, "_handle_permission_key", None)
        if callable(handler) and handler(event.key):
            event.stop()
