from __future__ import annotations

from pathlib import Path

from my_agent.core.permissions.storage import load_policy_file, save_policy_file


def test_load_policy_file_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_policy_file(tmp_path / "policy.toml") == {}


def test_save_policy_file_creates_parent_and_loads_always_section(tmp_path: Path) -> None:
    policy_file = tmp_path / "nested" / "policy.toml"

    save_policy_file({"bash": "allow", "write_file": "deny"}, policy_file)

    assert policy_file.exists()
    assert load_policy_file(policy_file) == {
        "bash": "allow",
        "write_file": "deny",
    }


def test_load_policy_file_ignores_other_sections(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.toml"
    policy_file.write_text(
        "\n".join(
            [
                "[always]",
                'bash = "allow"',
                "",
                "[other]",
                'write_file = "deny"',
            ]
        ),
        encoding="utf-8",
    )

    assert load_policy_file(policy_file) == {"bash": "allow"}
