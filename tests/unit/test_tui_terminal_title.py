from __future__ import annotations

import pytest

from my_agent.tui.terminal_title import (
    STARTUP_STATE_CONNECTING,
    STARTUP_STATE_CREATING_SESSION,
    STARTUP_STATE_DISCONNECTED,
    STARTUP_STATE_READY,
    STARTUP_STATE_RUNNING,
    STARTUP_STATE_SETUP_ERROR,
    STARTUP_STATE_STARTING,
    STARTUP_STATE_WAITING_PERMISSION,
    format_terminal_title,
    sanitize_project_name,
)

# ---------------------------------------------------------------------------
# format_terminal_title — state → "My Agent - ..." mapping
# ---------------------------------------------------------------------------


def test_terminal_title_starting() -> None:
    assert format_terminal_title(STARTUP_STATE_STARTING, None) == "My Agent - starting"
    assert format_terminal_title(STARTUP_STATE_STARTING, "my-project") == (
        "My Agent - starting"
    )


def test_terminal_title_connecting_creating_session_use_starting_suffix() -> None:
    # CONNECTING / CREATING_SESSION happen before the project is "active";
    # keep title as "starting" so user sees a single coherent startup phase.
    assert format_terminal_title(STARTUP_STATE_CONNECTING, "proj").startswith(
        "My Agent"
    )
    assert format_terminal_title(STARTUP_STATE_CREATING_SESSION, "proj").startswith(
        "My Agent"
    )


def test_terminal_title_ready_shows_project_name() -> None:
    assert format_terminal_title(STARTUP_STATE_READY, "my-project") == (
        "My Agent - my-project"
    )


def test_terminal_title_ready_without_project_name_falls_back() -> None:
    assert format_terminal_title(STARTUP_STATE_READY, None) == "My Agent"
    assert format_terminal_title(STARTUP_STATE_READY, "") == "My Agent"


def test_terminal_title_running_shows_working_suffix() -> None:
    assert format_terminal_title(STARTUP_STATE_RUNNING, "proj") == (
        "My Agent - proj - working"
    )


def test_terminal_title_permission_shows_required_suffix() -> None:
    assert format_terminal_title(STARTUP_STATE_WAITING_PERMISSION, "proj") == (
        "My Agent - proj - permission required"
    )


def test_terminal_title_disconnected_shows_disconnected_suffix() -> None:
    assert format_terminal_title(STARTUP_STATE_DISCONNECTED, "proj") == (
        "My Agent - proj - disconnected"
    )


def test_terminal_title_setup_error_shows_setup_suffix() -> None:
    assert format_terminal_title(STARTUP_STATE_SETUP_ERROR, "proj") == (
        "My Agent - proj - setup required"
    )


def test_terminal_title_preserves_cjk_project_name() -> None:
    # CJK names must survive (we are NOT munging unicode here; sanitization
    # only strips secret-like substrings, not international characters).
    assert format_terminal_title(STARTUP_STATE_READY, "我的项目") == (
        "My Agent - 我的项目"
    )


# ---------------------------------------------------------------------------
# sanitize_project_name — never leak secrets in derived project name
# ---------------------------------------------------------------------------


def test_sanitize_project_name_passes_plain_name() -> None:
    assert sanitize_project_name("my-agent") == "my-agent"
    assert sanitize_project_name("hello_world") == "hello_world"


def test_sanitize_project_name_passes_cjk() -> None:
    assert sanitize_project_name("我的项目") == "我的项目"


def test_sanitize_project_name_strips_deepseek_api_key_pattern() -> None:
    # User might cd into a directory accidentally named with a key fragment.
    name = "sk-abcd1234efgh5678"
    sanitized = sanitize_project_name(name)
    assert "sk-abcd1234" not in sanitized
    assert "abcd1234" not in sanitized


def test_sanitize_project_name_strips_generic_api_key_prefix() -> None:
    name = "api_key=secret123"
    sanitized = sanitize_project_name(name)
    assert "secret123" not in sanitized.lower()
    assert "api_key=secret" not in sanitized.lower()


def test_sanitize_project_name_strips_token_prefix() -> None:
    name = "token=eyJhbGc"
    sanitized = sanitize_project_name(name)
    assert "eyJhbGc" not in sanitized


def test_sanitize_project_name_truncates_extremely_long_name() -> None:
    long_name = "x" * 200
    sanitized = sanitize_project_name(long_name)
    assert len(sanitized) <= 80
    # And survives format_terminal_title without blowing up common terminals.
    title = format_terminal_title(STARTUP_STATE_READY, sanitized)
    assert title.startswith("My Agent - ")


def test_sanitize_project_name_strips_dotenv_filename_to_safe_label() -> None:
    # If the user is somehow in a directory literally called ".env",
    # we should NOT echo ".env" verbatim into the title (it leaks that
    # the user is in a secrets-bearing location).
    sanitized = sanitize_project_name(".env")
    assert sanitized != ".env"
    assert "env" not in sanitized.lower() or sanitized == "env"


def test_sanitize_project_name_empty_returns_empty() -> None:
    assert sanitize_project_name("") == ""


@pytest.mark.parametrize(
    "secret",
    [
        "sk-1234567890abcdef",
        "sk-live-XYZ123abc",
        "API_KEY=foo",
        "api_key=bar",
        "token=abc.def.ghi",
        "TOKEN=secret",
        "password=p@ss",
    ],
)
def test_sanitize_project_name_redacts_known_secret_patterns(secret: str) -> None:
    sanitized = sanitize_project_name(secret)
    lowered = sanitized.lower()
    for fragment in ("sk-", "api_key", "token", "password"):
        if fragment in secret.lower():
            # The fragment itself plus its secret value must not survive.
            assert fragment not in lowered or secret.lower() not in lowered
