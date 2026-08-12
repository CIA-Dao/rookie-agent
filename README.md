# Rookie Agent

Rookie Agent is a local coding agent for working in real project folders. It
uses a background Core process and a terminal UI or CLI client, so sessions,
tools, permissions, and event history remain observable and recoverable.

The public product name is **Rookie Agent**. The current command-line entry
points remain `my-agent`, `my-agent-tui`, and `my-agent-core` for compatibility.

## Install on Windows

The supported friend-install path is a version-pinned GitHub installer. Run
the installer from a trusted release tag:

```powershell
irm https://raw.githubusercontent.com/CIA-Dao/rookie-agent/v0.0.1/scripts/install.ps1 -OutFile install-rookie-agent.ps1
.\install-rookie-agent.ps1 -Version v0.0.1
Remove-Item .\install-rookie-agent.ps1
```

The installer bootstraps `uv` if needed, installs a non-editable tagged
revision, updates the user PATH, and verifies the command. It does not ask for
or handle an API key.

The installer also handles existing installations conservatively:

- no existing tool: install Rookie Agent directly;
- existing non-editable `rookie-agent`: upgrade it and restore the previous
  requirement if the upgrade fails;
- existing non-editable legacy `my-agent`: preflight the new package, migrate
  the tool, and attempt to restore the legacy requirement if migration fails;
- editable development install: stop without overwriting it;
- same-named command outside uv's tool directory: stop and report the PATH
  conflict.

For isolated installer testing, `-SkipPathUpdate` prevents the temporary uv
tool bin directory from being persisted to the user PATH. Normal installations
should leave this switch off.

The installer can also be run from a checked-out copy:

```powershell
.\scripts\install.ps1 -Version v0.0.1
```

Open a new terminal after installation, then run:

```powershell
my-agent
```

On first launch, the TUI asks for the DeepSeek API key and model. The key is
stored locally under `~/.my-agent/.env`; it is never part of the installer
arguments or repository files.

## Development setup

For local development, Python 3.12 and uv are required:

```powershell
uv sync --dev
uv run my-agent
```

Useful verification commands:

```powershell
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

## Architecture

```text
my-agent CLI / TUI
        |
        | local JSON-line RPC over TCP
        v
my-agent Core
        |
        +-- sessions and runs
        +-- tools and permissions
        +-- AgentRunner and AgentLoop
        +-- event and trace records
```

The agent operates relative to the project directory from which the command
is launched. Runtime logs and session data are local artifacts and should not
be committed.

## Release scope

The first release supports DeepSeek as its provider and offers the
`deepseek-v4-pro` and `deepseek-v4-flash` models. PyPI and Node package
distribution are later release channels, not prerequisites for the first
GitHub installer.

## Security

Never commit `.env`, API keys, tokens, runtime logs, or local configuration.
Report security issues privately to the project maintainers rather than
including credentials in an issue or pull request.
