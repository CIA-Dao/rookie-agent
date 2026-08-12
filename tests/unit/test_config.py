from __future__ import annotations

from pytest import MonkeyPatch

from my_agent.core.config import (
    DEEPSEEK_MODEL_OPTIONS,
    Config,
    _apply_env,
    _apply_toml,
    normalize_deepseek_model,
)


def test_deepseek_model_aliases_normalize_to_supported_ids() -> None:
    assert normalize_deepseek_model("pro") == "deepseek-v4-pro"
    assert normalize_deepseek_model(" FLASH ") == "deepseek-v4-flash"
    assert normalize_deepseek_model("deepseek-unknown") is None
    assert set(DEEPSEEK_MODEL_OPTIONS) == {"deepseek-v4-pro", "deepseek-v4-flash"}


def test_core_runs_dir_toml_overrides_default() -> None:
    config = Config()

    _apply_toml(config, {"core": {"runs_dir": "data/runs"}})

    assert config.runs_dir == "data/runs"


def test_runs_dir_env_overrides_default(monkeypatch: MonkeyPatch) -> None:
    config = Config()
    monkeypatch.setenv("MY_AGENT_RUNS_DIR", "env/runs")

    _apply_env(config)

    assert config.runs_dir == "env/runs"


def test_compact_config_defaults() -> None:
    config = Config()

    assert config.compact.enabled is True
    assert config.compact.token_threshold == 120_000
    assert config.compact.tool_result_limit == 8_000
    assert config.compact.tool_result_keep == 4_000
    assert config.compact.context_ratio == 0.0


def test_compact_toml_overrides_defaults() -> None:
    config = Config()

    _apply_toml(
        config,
        {
            "compact": {
                "enabled": False,
                "token_threshold": 10_000,
                "tool_result_limit": 1_000,
                "tool_result_keep": 500,
                "context_ratio": 0.8,
            }
        },
    )

    assert config.compact.enabled is False
    assert config.compact.token_threshold == 10_000
    assert config.compact.tool_result_limit == 1_000
    assert config.compact.tool_result_keep == 500
    assert config.compact.context_ratio == 0.8


def test_compact_toml_rejects_unknown_keys() -> None:
    config = Config()

    try:
        _apply_toml(config, {"compact": {"surprise": True}})
    except SystemExit as exc:
        assert "Unknown compact config keys: surprise" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_compact_toml_rejects_invalid_values() -> None:
    config = Config()

    try:
        _apply_toml(config, {"compact": {"tool_result_keep": 0}})
    except SystemExit as exc:
        assert "compact.tool_result_keep must be a positive integer" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_compact_env_overrides_defaults(monkeypatch: MonkeyPatch) -> None:
    config = Config()
    monkeypatch.setenv("MY_AGENT_COMPACT_ENABLED", "false")
    monkeypatch.setenv("MY_AGENT_COMPACT_TOKEN_THRESHOLD", "10000")
    monkeypatch.setenv("MY_AGENT_COMPACT_TOOL_RESULT_LIMIT", "1000")
    monkeypatch.setenv("MY_AGENT_COMPACT_TOOL_RESULT_KEEP", "500")
    monkeypatch.setenv("MY_AGENT_COMPACT_CONTEXT_RATIO", "0.75")

    _apply_env(config)

    assert config.compact.enabled is False
    assert config.compact.token_threshold == 10_000
    assert config.compact.tool_result_limit == 1_000
    assert config.compact.tool_result_keep == 500
    assert config.compact.context_ratio == 0.75


def test_compact_env_rejects_invalid_integer(monkeypatch: MonkeyPatch) -> None:
    config = Config()
    monkeypatch.setenv("MY_AGENT_COMPACT_TOKEN_THRESHOLD", "nope")

    try:
        _apply_env(config)
    except SystemExit as exc:
        assert "MY_AGENT_COMPACT_TOKEN_THRESHOLD must be an integer" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_compact_env_rejects_non_positive_integer(monkeypatch: MonkeyPatch) -> None:
    config = Config()
    monkeypatch.setenv("MY_AGENT_COMPACT_TOOL_RESULT_LIMIT", "0")

    try:
        _apply_env(config)
    except SystemExit as exc:
        assert "MY_AGENT_COMPACT_TOOL_RESULT_LIMIT must be positive" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_compact_toml_rejects_invalid_context_ratio() -> None:
    config = Config()

    try:
        _apply_toml(config, {"compact": {"context_ratio": 1.5}})
    except SystemExit as exc:
        assert "compact.context_ratio must be between 0 and 1" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_compact_env_rejects_invalid_context_ratio(monkeypatch: MonkeyPatch) -> None:
    config = Config()
    monkeypatch.setenv("MY_AGENT_COMPACT_CONTEXT_RATIO", "1.5")

    try:
        _apply_env(config)
    except SystemExit as exc:
        assert "MY_AGENT_COMPACT_CONTEXT_RATIO must be between 0 and 1" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_mcp_toml_parses_stdio_and_tcp_servers() -> None:
    config = Config()

    _apply_toml(
        config,
        {
            "mcp": {
                "servers": [
                    {
                        "name": "fs",
                        "transport": "stdio",
                        "command": "mcp-filesystem",
                        "args": ["D:/project"],
                        "env": {"TOKEN": "x"},
                    },
                    {
                        "name": "docs",
                        "transport": "tcp",
                        "host": "127.0.0.1",
                        "port": 9999,
                    },
                ]
            }
        },
    )

    assert len(config.mcp.servers) == 2
    assert config.mcp.servers[0].name == "fs"
    assert config.mcp.servers[0].transport == "stdio"
    assert config.mcp.servers[0].command == "mcp-filesystem"
    assert config.mcp.servers[0].args == ["D:/project"]
    assert config.mcp.servers[0].env == {"TOKEN": "x"}
    assert config.mcp.servers[1].name == "docs"
    assert config.mcp.servers[1].transport == "tcp"
    assert config.mcp.servers[1].host == "127.0.0.1"
    assert config.mcp.servers[1].port == 9999


def test_mcp_toml_rejects_unknown_keys() -> None:
    config = Config()

    try:
        _apply_toml(config, {"mcp": {"surprise": True}})
    except SystemExit as exc:
        assert "Unknown mcp config keys: surprise" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_mcp_toml_rejects_invalid_transport() -> None:
    config = Config()

    try:
        _apply_toml(config, {"mcp": {"servers": [{"name": "bad", "transport": "http"}]}})
    except SystemExit as exc:
        assert "mcp.servers[0].transport must be 'stdio' or 'tcp'" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_mcp_toml_rejects_missing_name() -> None:
    config = Config()

    try:
        _apply_toml(config, {"mcp": {"servers": [{"transport": "tcp"}]}})
    except SystemExit as exc:
        assert "mcp.servers[0].name must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
