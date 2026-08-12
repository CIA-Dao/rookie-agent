from __future__ import annotations

import argparse

from my_agent.core.config import get_config
from my_agent.core.user_config import GLOBAL_ENV_FILE
from my_agent.tui.app import MyAgentTuiApp


def main() -> None:
    parser = argparse.ArgumentParser(prog="my-agent-tui", description="My Agent TUI")
    parser.parse_args()

    config = get_config()
    app = MyAgentTuiApp(
        config.host,
        config.port,
        global_env_file=GLOBAL_ENV_FILE,
    )
    app.run()


if __name__ == "__main__":
    main()
