from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# P5-1 TUI product shell — terminal title + project name helpers.
#
# These helpers are isolated from the Textual app so they can be unit-tested
# without spinning up the TUI. They never log, never print, and never embed
# raw secret values into the formatted title.
# ---------------------------------------------------------------------------

# Lifecycle states used by the TUI startup shell. These intentionally use
# lowercase string literals (not an Enum) so tests can compare against plain
# strings and the values can be passed across test boundaries trivially.
STARTUP_STATE_STARTING = "starting"
STARTUP_STATE_CONNECTING = "connecting"
STARTUP_STATE_CREATING_SESSION = "creating_session"
STARTUP_STATE_READY = "ready"
STARTUP_STATE_RUNNING = "running"
STARTUP_STATE_WAITING_PERMISSION = "waiting_permission"
STARTUP_STATE_DISCONNECTED = "disconnected"
STARTUP_STATE_SETUP_ERROR = "setup_error"
STARTUP_STATE_WORKSPACE_IN_USE = "workspace_in_use"

# All recognized states — used for input validation in tests/debug only.
STARTUP_STATES: frozenset[str] = frozenset(
    {
        STARTUP_STATE_STARTING,
        STARTUP_STATE_CONNECTING,
        STARTUP_STATE_CREATING_SESSION,
        STARTUP_STATE_READY,
        STARTUP_STATE_RUNNING,
        STARTUP_STATE_WAITING_PERMISSION,
        STARTUP_STATE_DISCONNECTED,
        STARTUP_STATE_SETUP_ERROR,
        STARTUP_STATE_WORKSPACE_IN_USE,
    }
)

_BRAND = "My Agent"

# Maximum visual width (in characters) for the project-name segment. Long
# paths such as deeply nested monorepo subdirectories should be truncated so
# the title stays readable in standard terminal widths.
_MAX_PROJECT_NAME_LEN = 48

# Patterns that indicate a leaked secret inside what was supposed to be a
# project name (e.g. the user `cd`-ed into a directory literally named
# `sk-12345...` by accident). We never echo these into terminal titles.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # DeepSeek / OpenAI-style keys: sk-...
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    # Generic api_key=VALUE / api_keyVALUE — value can be 3+ non-whitespace
    # chars (covers short test secrets as well as real ones).
    re.compile(r"api_key[\s=_-]+[^\s]{3,}", re.IGNORECASE),
    # token=VALUE / tokenVALUE
    re.compile(r"token[\s=_-]+[^\s]{3,}", re.IGNORECASE),
    # password=VALUE
    re.compile(r"password[\s=_-]+[^\s]{3,}", re.IGNORECASE),
    # Bearer JWT-like fragments (three base64 chunks separated by dots).
    re.compile(r"eyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}"),
)


def sanitize_project_name(name: str) -> str:
    """Return a project-name safe for display in terminal titles.

    - Strips substrings that look like leaked secrets (api keys, tokens).
    - Truncates very long names so the title stays readable.
    - Replaces the literal `.env` filename with a non-secret label so we
      don't surface that the user is inside a secrets directory.
    - Returns the input unchanged for ordinary project names, including CJK.
    """
    if not name:
        return ""

    cleaned = name
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[redacted]", cleaned)

    # If the directory is literally ".env" or contains ".env" as a leaf name,
    # don't surface that fact verbatim — it leaks that the user is in a
    # secrets-bearing location.
    if cleaned.rstrip("/\\") == ".env":
        return "env"
    cleaned = cleaned.replace(".env", "[env]")

    if len(cleaned) > _MAX_PROJECT_NAME_LEN:
        cleaned = cleaned[: _MAX_PROJECT_NAME_LEN - 1] + "…"

    return cleaned


def format_terminal_title(state: str, project_name: str | None) -> str:
    """Return the terminal title for a given startup state and project name.

    Rules (P5-1 §4.4):
    - starting → "My Agent - starting"
    - ready    → "My Agent - <project>"
    - running  → "My Agent - <project> - working"
    - permission → "My Agent - <project> - permission required"
    - disconnected → "My Agent - <project> - disconnected"
    - setup_error → "My Agent - <project> - setup required"
    - connecting / creating_session → "My Agent - starting" (single coherent
      startup phase before the project is "live")

    `project_name` is sanitized internally — callers may pass the raw cwd name.
    """
    safe_name = sanitize_project_name(project_name) if project_name else ""

    if state == STARTUP_STATE_STARTING:
        return f"{_BRAND} - starting"
    if state in (STARTUP_STATE_CONNECTING, STARTUP_STATE_CREATING_SESSION):
        # Keep "starting" suffix during the pre-ready phases so the user
        # sees one coherent startup label instead of three rapid changes.
        return f"{_BRAND} - starting"
    if state == STARTUP_STATE_DISCONNECTED:
        return f"{_BRAND} - {safe_name} - disconnected" if safe_name else (
            f"{_BRAND} - disconnected"
        )
    if state == STARTUP_STATE_WAITING_PERMISSION:
        return f"{_BRAND} - {safe_name} - permission required" if safe_name else (
            f"{_BRAND} - permission required"
        )
    if state == STARTUP_STATE_RUNNING:
        return f"{_BRAND} - {safe_name} - working" if safe_name else (
            f"{_BRAND} - working"
        )
    if state == STARTUP_STATE_SETUP_ERROR:
        return f"{_BRAND} - {safe_name} - setup required" if safe_name else (
            f"{_BRAND} - setup required"
        )
    if state == STARTUP_STATE_WORKSPACE_IN_USE:
        return f"{_BRAND} - {safe_name} - workspace in use" if safe_name else (
            f"{_BRAND} - workspace in use"
        )

    # Default: ready (or any unknown state treated as ready-ish).
    return f"{_BRAND} - {safe_name}" if safe_name else _BRAND


def prompt_border_title_for_state(state: str) -> str:
    """Return the TextArea border title for a given startup state.

    These map 1:1 with P5-1 §4.3. Each title is short, human-readable, and
    communicates the current interaction affordance.
    """
    if state in (STARTUP_STATE_STARTING, STARTUP_STATE_CONNECTING,
                 STARTUP_STATE_CREATING_SESSION):
        return "connecting..."
    if state == STARTUP_STATE_READY:
        return "type message"
    if state == STARTUP_STATE_RUNNING:
        return "agent is working"
    if state == STARTUP_STATE_WAITING_PERMISSION:
        return "permission required"
    if state == STARTUP_STATE_DISCONNECTED:
        return "waiting for Core"
    if state == STARTUP_STATE_SETUP_ERROR:
        return "setup required"
    if state == STARTUP_STATE_WORKSPACE_IN_USE:
        return "workspace already open"
    return "type message"


def connection_label_for_state(state: str) -> str:
    """Return the connection-status label shown inside AppHeader.

    Distinct from prompt_border_title: this label speaks about the Core
    connection itself, while the prompt title speaks about the input
    affordance. P5-1 §4.1 requires text+color, never color alone.
    """
    if state in (STARTUP_STATE_STARTING, STARTUP_STATE_CONNECTING,
                 STARTUP_STATE_CREATING_SESSION):
        return "connecting"
    if state == STARTUP_STATE_READY:
        return "connected"
    if state == STARTUP_STATE_RUNNING:
        return "connected"
    if state == STARTUP_STATE_WAITING_PERMISSION:
        return "connected"
    if state == STARTUP_STATE_DISCONNECTED:
        return "disconnected"
    if state == STARTUP_STATE_SETUP_ERROR:
        return "setup needed"
    if state == STARTUP_STATE_WORKSPACE_IN_USE:
        return "workspace in use"
    return "unknown"


def run_state_label_for_state(state: str) -> str:
    """Return the run-state label shown inside AppHeader."""
    if state in (STARTUP_STATE_STARTING, STARTUP_STATE_CONNECTING,
                 STARTUP_STATE_CREATING_SESSION):
        return "starting"
    if state == STARTUP_STATE_READY:
        return "ready"
    if state == STARTUP_STATE_RUNNING:
        return "running"
    if state == STARTUP_STATE_WAITING_PERMISSION:
        return "permission"
    if state == STARTUP_STATE_DISCONNECTED:
        return "offline"
    if state == STARTUP_STATE_SETUP_ERROR:
        return "error"
    if state == STARTUP_STATE_WORKSPACE_IN_USE:
        return "blocked"
    return "idle"
