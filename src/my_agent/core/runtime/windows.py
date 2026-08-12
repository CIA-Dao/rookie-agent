from __future__ import annotations

import asyncio
import locale
import os
import re
import shutil
from pathlib import Path


class WindowsRuntime:
    """Shared Windows process policy used by command-capable tools."""

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._bash_probe: bool | None = None

    @property
    def available_shells(self) -> dict[str, str | None]:
        return {
            "powershell": shutil.which("powershell.exe") or shutil.which("pwsh.exe"),
            "cmd": shutil.which("cmd.exe"),
            "bash": shutil.which("bash.exe"),
        }

    def normalize_argv(self, argv: list[str]) -> list[str]:
        if not argv or os.name != "nt":
            return argv
        executable = argv[0]
        if Path(executable).suffix:
            return argv
        cmd_executable = shutil.which(f"{executable}.cmd")
        if cmd_executable:
            return [f"{executable}.cmd", *argv[1:]]
        return argv

    def prepare_shell_command(self, command: str) -> str:
        if os.name == "nt":
            return f"chcp 65001>nul & {command}"
        return command

    def shell_argv(self, command: str) -> list[str] | None:
        if os.name != "nt":
            return None
        bash_match = re.match(r"^\s*bash(?:\.exe)?\s+-lc\s+(.+)$", command, re.IGNORECASE)
        if bash_match and self.available_shells["bash"]:
            script = bash_match.group(1).strip()
            if len(script) >= 2 and script[0] == script[-1] and script[0] in "\"'":
                script = script[1:-1]
            return [self.available_shells["bash"] or "bash.exe", "-lc", script]
        if _looks_like_powershell(command) or _contains_non_ascii(command):
            powershell = self.available_shells["powershell"]
            if powershell:
                utf8_command = (
                    "$OutputEncoding = New-Object System.Text.UTF8Encoding($false); "
                    "[Console]::OutputEncoding = $OutputEncoding; "
                    + command
                )
                return [powershell, "-NoProfile", "-NonInteractive", "-Command", utf8_command]
        cmd = self.available_shells["cmd"] or "cmd.exe"
        return [cmd, "/d", "/s", "/c", self.prepare_shell_command(command)]

    async def bash_is_usable(self) -> bool:
        if os.name != "nt":
            return True
        if self._bash_probe is not None:
            return self._bash_probe
        bash = self.available_shells["bash"]
        if not bash:
            self._bash_probe = False
            return False
        try:
            process = await asyncio.create_subprocess_exec(
                bash,
                "-lc",
                "exit 0",
                cwd=self.workspace_root,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=3.0)
            self._bash_probe = process.returncode == 0
        except (OSError, TimeoutError):
            self._bash_probe = False
        return self._bash_probe

    @staticmethod
    def decode_output(data: bytes) -> str:
        if not data:
            return ""
        for encoding in ("utf-8-sig", "utf-16", locale.getpreferredencoding(False), "mbcs"):
            try:
                return data.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return data.decode("utf-8", errors="replace")


_POWERSHELL_RE = re.compile(
    r"(?:^|\s)(?:Get-|Set-|Select-|Where-|Test-|Write-|Out-|Measure-|Convert-|Resolve-)[A-Za-z-]*",
    re.IGNORECASE,
)


def _looks_like_powershell(command: str) -> bool:
    return bool(_POWERSHELL_RE.search(command))


def _contains_non_ascii(command: str) -> bool:
    return any(ord(char) > 127 for char in command)
