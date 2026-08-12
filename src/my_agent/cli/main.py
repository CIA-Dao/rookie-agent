from __future__ import annotations

import argparse
import sys

from my_agent.cli.commands.chat import cmd_chat
from my_agent.cli.commands.core import cmd_core_start, cmd_core_status, cmd_core_stop
from my_agent.cli.commands.init import cmd_init
from my_agent.cli.commands.ping import cmd_ping
from my_agent.cli.commands.run import cmd_run
from my_agent.core.config import get_config


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdio()

    # Default behavior (no args): launch the TUI to provide the product entrypoint.
    # `--help`, `--version`, and all subcommands go through argparse as usual.
    if len(sys.argv) == 1:
        from my_agent.tui.__main__ import main as tui_main

        tui_main()
        return

    parser = argparse.ArgumentParser(prog="my-agent", description="My Agent CLI")

    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("ping", help="Ping the core daemon")
    subparsers.add_parser("chat", help="Start a chat session")
    subparsers.add_parser("init", help="Initialize project context for my-agent")

    run_parser = subparsers.add_parser("run", help="Run the agent")
    run_parser.add_argument("goal", help="Goal to send to the agent")

    # 嵌套子命令：core status（后续 core start/stop 也会挂这里）
    core_parser = subparsers.add_parser("core", help="Manage the core daemon")
    core_subparsers = core_parser.add_subparsers(dest="core_command")
    core_subparsers.add_parser("status", help="Check if core daemon is running")
    core_subparsers.add_parser("start", help="Start the core daemon in the background")
    core_subparsers.add_parser("stop", help="Stop the core daemon")

    args = parser.parse_args()

    if args.version:
        from my_agent import __version__

        print(__version__)
        return

    if args.command == "ping":
        config = get_config()
        cmd_ping(config)
    elif args.command == "chat":
        config = get_config()
        cmd_chat(config)
    elif args.command == "init":
        cmd_init()
    elif args.command == "run":
        config = get_config()
        cmd_run(args.goal, config)
    elif args.command == "core":
        if args.core_command == "status":
            config = get_config()
            cmd_core_status(config.host, config.port)
        elif args.core_command == "start":
            config = get_config()
            cmd_core_start(config.host, config.port)
        elif args.core_command == "stop":
            config = get_config()
            cmd_core_stop(config.host, config.port)
        else:
            parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
