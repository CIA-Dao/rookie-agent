from __future__ import annotations

from pathlib import Path

import pytest

from my_agent.core.user_config import (
    GLOBAL_ENV_FILE,
    is_valid_deepseek_api_key,
    save_deepseek_api_key,
    save_deepseek_model,
    upsert_env_var,
)

# Sentinel: a real-looking key format, but explicitly not a live secret.
# Used to verify the helper does NOT echo key value in errors/returns.
_FAKE_KEY = "sk-test-AAAA-BBBB-CCCC-DDDD"


def test_is_valid_deepseek_api_key_uses_conservative_format_check() -> None:
    assert is_valid_deepseek_api_key(_FAKE_KEY)
    assert not is_valid_deepseek_api_key("")
    assert not is_valid_deepseek_api_key("   ")
    assert not is_valid_deepseek_api_key("deepseek-key")
    assert not is_valid_deepseek_api_key("sk-invalid key")


def test_upsert_env_var_creates_file_when_missing(tmp_path: Path) -> None:
    """文件不存在时 upsert 应创建文件，包含一行 KEY=value。"""
    env_file = tmp_path / ".my-agent" / ".env"

    upsert_env_var(env_file, "DEEPSEEK_API_KEY", _FAKE_KEY)

    assert env_file.exists()
    assert env_file.read_text(encoding="utf-8").strip() == f"DEEPSEEK_API_KEY={_FAKE_KEY}"


def test_upsert_env_var_creates_parent_directory(tmp_path: Path) -> None:
    """父目录不存在时自动创建。"""
    env_file = tmp_path / "nested" / "deeper" / ".env"

    upsert_env_var(env_file, "DEEPSEEK_API_KEY", _FAKE_KEY)

    assert env_file.exists()
    assert env_file.parent.is_dir()


def test_upsert_env_var_preserves_other_lines(tmp_path: Path) -> None:
    """文件已存在且包含其它变量时，保留其它行，仅新增目标行。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTHER_VAR=keep-me\n# a comment\nANOTHER=1\n",
        encoding="utf-8",
    )

    upsert_env_var(env_file, "DEEPSEEK_API_KEY", _FAKE_KEY)

    content = env_file.read_text(encoding="utf-8")
    assert "OTHER_VAR=keep-me" in content
    assert "ANOTHER=1" in content
    assert "# a comment" in content
    assert f"DEEPSEEK_API_KEY={_FAKE_KEY}" in content


def test_upsert_env_var_updates_existing_key(tmp_path: Path) -> None:
    """已有 DEEPSEEK_API_KEY 时更新该行，不重复，不影响其它行。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=sk-old-value\nOTHER=keep\n",
        encoding="utf-8",
    )

    upsert_env_var(env_file, "DEEPSEEK_API_KEY", _FAKE_KEY)

    content = env_file.read_text(encoding="utf-8")
    assert content.count("DEEPSEEK_API_KEY=") == 1
    assert f"DEEPSEEK_API_KEY={_FAKE_KEY}" in content
    assert "sk-old-value" not in content
    assert "OTHER=keep" in content


def test_upsert_env_var_preserves_surrounding_lines_when_updating(tmp_path: Path) -> None:
    """更新时保留前后行，不丢失内容。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIRST=1\nDEEPSEEK_API_KEY=sk-old\nLAST=2\n",
        encoding="utf-8",
    )

    upsert_env_var(env_file, "DEEPSEEK_API_KEY", _FAKE_KEY)

    content = env_file.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line]
    assert lines[0] == "FIRST=1"
    assert lines[1] == f"DEEPSEEK_API_KEY={_FAKE_KEY}"
    assert lines[2] == "LAST=2"


def test_save_deepseek_api_key_uses_utf8_and_default_path(tmp_path: Path) -> None:
    """save_deepseek_api_key 接受 env_file 参数，写入 UTF-8。"""
    env_file = tmp_path / ".env"

    save_deepseek_api_key(_FAKE_KEY, env_file=env_file)

    raw = env_file.read_bytes()
    assert raw.decode("utf-8").strip() == f"DEEPSEEK_API_KEY={_FAKE_KEY}"


def test_save_deepseek_api_key_rejects_empty_value(tmp_path: Path) -> None:
    """空 key 抛 ValueError，且不写文件。"""
    env_file = tmp_path / ".env"

    with pytest.raises(ValueError, match="empty"):
        save_deepseek_api_key("   ", env_file=env_file)

    assert not env_file.exists()


def test_save_deepseek_api_key_strips_whitespace(tmp_path: Path) -> None:
    """key 首尾空白被去除后写入。"""
    env_file = tmp_path / ".env"

    save_deepseek_api_key(f"  {_FAKE_KEY}  \n", env_file=env_file)

    assert env_file.read_text(encoding="utf-8").strip() == f"DEEPSEEK_API_KEY={_FAKE_KEY}"


def test_global_env_file_is_absolute_path() -> None:
    """GLOBAL_ENV_FILE 常量解析为绝对路径（expanduser 后）。"""
    assert GLOBAL_ENV_FILE.is_absolute()
    assert GLOBAL_ENV_FILE.name == ".env"


def test_upsert_env_var_value_with_special_chars(tmp_path: Path) -> None:
    """key 中含 = 号时不破坏行解析（值部分允许 =）。"""
    env_file = tmp_path / ".env"
    value = "sk-proj-abc==def"

    upsert_env_var(env_file, "DEEPSEEK_API_KEY", value)

    content = env_file.read_text(encoding="utf-8").strip()
    assert content == "DEEPSEEK_API_KEY=sk-proj-abc==def"


def test_save_deepseek_model_accepts_alias_and_creates_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"

    assert save_deepseek_model("flash", config_file=config_file) == "deepseek-v4-flash"
    assert config_file.read_text(encoding="utf-8") == (
        '[llm]\ndefault_model = "deepseek-v4-flash"\n'
    )


def test_save_deepseek_model_preserves_existing_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[core]\nport = 7437\n\n[llm]\ndefault_model = "deepseek-v4-pro"\nrouter = "static"\n',
        encoding="utf-8",
    )

    save_deepseek_model("deepseek-v4-flash", config_file=config_file)

    content = config_file.read_text(encoding="utf-8")
    assert 'port = 7437' in content
    assert 'default_model = "deepseek-v4-flash"' in content
    assert 'router = "static"' in content
    assert content.count("default_model") == 1


def test_save_deepseek_model_rejects_unsupported_model(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"

    with pytest.raises(ValueError, match="unsupported DeepSeek model"):
        save_deepseek_model("deepseek-unknown", config_file=config_file)

    assert not config_file.exists()
