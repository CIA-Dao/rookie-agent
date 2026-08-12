from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from my_agent.core.config import get_config

# 这些环境变量会影响 get_config()，测试前必须清掉
_ENV_KEYS_TO_CLEAN = (
    "MY_AGENT_CONFIG",
    "MY_AGENT_HOST",
    "MY_AGENT_PORT",
    "DEEPSEEK_API_KEY",
)


def _clean_env(monkeypatch: MonkeyPatch) -> None:
    for key in _ENV_KEYS_TO_CLEAN:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """把 USERPROFILE / HOME 指向 tmp_path，让 ~/.my-agent/.env 落在可控位置。"""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_get_config_loads_global_env_as_fallback(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    isolated_home: Path,
) -> None:
    """cwd 没有 .env，~/.my-agent/.env 提供全局 fallback。"""
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    # 模拟全局 .env
    global_dir = isolated_home / ".my-agent"
    global_dir.mkdir()
    (global_dir / ".env").write_text("MY_AGENT_PORT=9001\n", encoding="utf-8")

    config = get_config()
    assert config.port == 9001


def test_project_env_wins_over_global_env(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    isolated_home: Path,
) -> None:
    """cwd/.env 优先于 ~/.my-agent/.env。"""
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    # 项目级 .env
    (tmp_path / ".env").write_text("MY_AGENT_PORT=9002\n", encoding="utf-8")

    # 全局 .env
    global_dir = isolated_home / ".my-agent"
    global_dir.mkdir()
    (global_dir / ".env").write_text("MY_AGENT_PORT=9001\n", encoding="utf-8")

    config = get_config()
    assert config.port == 9002


def test_os_env_wins_over_dotenv_files(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    isolated_home: Path,
) -> None:
    """OS 环境变量优先于两个 dotenv 文件。"""
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MY_AGENT_PORT", "9003")

    # 项目级 + 全局 .env 都设了不同值
    (tmp_path / ".env").write_text("MY_AGENT_PORT=9002\n", encoding="utf-8")
    global_dir = isolated_home / ".my-agent"
    global_dir.mkdir()
    (global_dir / ".env").write_text("MY_AGENT_PORT=9001\n", encoding="utf-8")

    config = get_config()
    assert config.port == 9003


def test_global_env_can_point_to_my_agent_config(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    isolated_home: Path,
) -> None:
    """全局 ~/.my-agent/.env 里的 MY_AGENT_CONFIG 仍能指定 TOML 配置路径。"""
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "custom-config.toml"
    config_file.write_text("[core]\nport = 9010\n", encoding="utf-8")

    global_dir = isolated_home / ".my-agent"
    global_dir.mkdir()
    (global_dir / ".env").write_text(
        f"MY_AGENT_CONFIG={config_file}\n",
        encoding="utf-8",
    )

    config = get_config()
    assert config.port == 9010
