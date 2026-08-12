from __future__ import annotations

import re
from pathlib import Path

from my_agent.core.config import normalize_deepseek_model

GLOBAL_ENV_FILE: Path = Path("~/.my-agent/.env").expanduser()
GLOBAL_CONFIG_FILE: Path = Path("~/.my-agent/config.toml").expanduser()


def is_valid_deepseek_api_key(value: str) -> bool:
    """Perform a conservative local format check without contacting DeepSeek."""
    return re.fullmatch(r"sk-[^\s]+", value.strip()) is not None


def upsert_env_var(path: Path, key: str, value: str) -> None:
    """Create or update one KEY=value line in a dotenv file.

    - Auto-creates parent directories.
    - Creates the file if missing.
    - Preserves other lines when the file exists.
    - If `KEY=` already exists, replaces just that line in place.
    - Writes UTF-8 with a trailing newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    target_prefix = f"{key}="

    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines: list[str] = []
        for line in existing_lines:
            if line.startswith(target_prefix):
                new_lines.append(f"{key}={value}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{key}={value}")
    else:
        new_lines = [f"{key}={value}"]

    # Use newline="\n" so behavior is deterministic across Windows/POSIX.
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8", newline="\n")


def save_deepseek_api_key(value: str, *, env_file: Path = GLOBAL_ENV_FILE) -> None:
    """Persist DEEPSEEK_API_KEY to a dotenv file.

    Raises ValueError if the value is empty after stripping.
    Never returns the key. Never logs the key.
    """
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("API key cannot be empty")
    upsert_env_var(env_file, "DEEPSEEK_API_KEY", cleaned)


def save_deepseek_model(
    value: str,
    *,
    config_file: Path = GLOBAL_CONFIG_FILE,
) -> str:
    """Persist a supported DeepSeek model in the global TOML config."""
    model = normalize_deepseek_model(value)
    if model is None:
        raise ValueError(f"unsupported DeepSeek model: {value}")

    config_file.parent.mkdir(parents=True, exist_ok=True)
    lines = config_file.read_text(encoding="utf-8").splitlines() if config_file.exists() else []
    in_llm = False
    replaced = False
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            if in_llm and not replaced:
                output.append(f'default_model = "{model}"')
                replaced = True
            in_llm = stripped == "[llm]"
        if in_llm and stripped.startswith("default_model"):
            if not replaced:
                output.append(f'default_model = "{model}"')
                replaced = True
            continue
        output.append(line)

    if in_llm and not replaced:
        output.append(f'default_model = "{model}"')
        replaced = True
    if not replaced:
        if output and output[-1].strip():
            output.append("")
        output.extend(["[llm]", f'default_model = "{model}"'])

    config_file.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
    return model
