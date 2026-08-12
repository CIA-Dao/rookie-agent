from __future__ import annotations

from pathlib import Path

_DEFAULT_POLICY_PATH = Path("~/.my-agent/policy.toml")


def load_policy_file(path: Path | None = None) -> dict[str, str]:
    policy_path = (path or _DEFAULT_POLICY_PATH).expanduser()

    if not policy_path.exists():
        return {}

    result: dict[str, str] = {}

    in_always = False

    for line in policy_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[always]":
            in_always = True
            continue
        if stripped.startswith("["):
            in_always = False
            continue
        if in_always and "=" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition("=")
            decision = value.strip().strip('"')
            if decision in ("allow", "deny"):
                result[key.strip()] = decision

    return result


def save_policy_file(always: dict[str, str], path: Path | None = None) -> None:
    policy_path = (path or _DEFAULT_POLICY_PATH).expanduser()
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Managed by my-agent.",
        "",
        "[always]",
    ]

    for tool_name, decision in sorted(always.items()):
        lines.append(f'{tool_name} = "{decision}"')

    policy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
