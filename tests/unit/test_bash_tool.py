from pathlib import Path

from my_agent.core.runtime.windows import WindowsRuntime
from my_agent.core.tools.builtin import BashTool
from my_agent.core.tools.builtin.bash import (
    _decode_process_output,
    _prepare_shell_command,
)


async def test_bash_tool_accepts_string_shell_commands(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = await BashTool().invoke({"command": "echo recoverable"})

    assert not result.is_error
    assert "recoverable" in result.content


async def test_bash_tool_keeps_array_commands_as_direct_processes(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).invoke({"command": ["python", "-c", "print('ok')"]})

    assert not result.is_error
    assert result.content.strip() == "ok"


async def test_bash_tool_rejects_directory_change_outside_workspace(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).invoke(
        {"command": "cd /home/user/tank && echo should-not-run"}
    )

    assert result.is_error
    assert result.error_type == "path_error"
    assert "outside workspace" in result.content


async def test_bash_tool_allows_directory_change_inside_workspace(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    result = await BashTool(tmp_path).invoke({"command": "cd child && echo ok"})

    assert not result.is_error
    assert "ok" in result.content


async def test_bash_tool_rejects_unix_inspection_commands_on_windows(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).invoke({"command": "wc -l tank.html"})

    assert result.is_error
    assert result.error_type == "unsupported-platform-command"
    assert "file_metadata" in result.content


def test_bash_output_decoder_preserves_utf16_and_utf8_text() -> None:
    assert _decode_process_output("中文输出".encode("utf-16")) == "中文输出"
    assert _decode_process_output("中文输出".encode()) == "中文输出"


def test_windows_shell_command_requests_utf8_code_page_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("my_agent.core.tools.builtin.bash.os.name", "nt")

    assert _prepare_shell_command("echo 中文") == "chcp 65001>nul & echo 中文"


def test_windows_shell_uses_explicit_cmd_boundary(monkeypatch) -> None:
    monkeypatch.setattr("my_agent.core.runtime.windows.os.name", "nt")
    runtime = WindowsRuntime()

    argv = runtime.shell_argv("echo ok")

    assert argv is not None
    assert argv[1:3] == ["/d", "/s"]


def test_windows_shell_routes_powershell_syntax(monkeypatch) -> None:
    monkeypatch.setattr("my_agent.core.runtime.windows.os.name", "nt")
    monkeypatch.setattr(
        "my_agent.core.runtime.windows.WindowsRuntime.available_shells",
        property(lambda _: {"powershell": "powershell.exe", "cmd": "cmd.exe", "bash": None}),
    )
    argv = WindowsRuntime().shell_argv("Get-Content app.txt")

    assert argv is not None
    assert argv[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert "Get-Content app.txt" in argv[-1]


def test_windows_shell_routes_non_ascii_output_to_utf8_powershell(monkeypatch) -> None:
    monkeypatch.setattr("my_agent.core.runtime.windows.os.name", "nt")
    monkeypatch.setattr(
        "my_agent.core.runtime.windows.WindowsRuntime.available_shells",
        property(lambda _: {"powershell": "powershell.exe", "cmd": "cmd.exe", "bash": None}),
    )
    argv = WindowsRuntime().shell_argv("echo 中文输出")

    assert argv is not None
    assert argv[0] == "powershell.exe"
    assert "OutputEncoding" in argv[-1]
