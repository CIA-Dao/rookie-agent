from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from my_agent.core.config import Config

_TEXT_FMT = 'level=%(levelname)s ts=%(asctime)s source=%(name)s msg="%(message)s"'
_JSON_FMT = '{"level":"%(levelname)s","ts":"%(asctime)s","source":"%(name)s","msg":"%(message)s"}'


def setup_logging(config: Config) -> None:
    """根据 pytest 配置设置 logging"""
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    fmt = _JSON_FMT if config.logging.format == "json" else _TEXT_FMT
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # 清掉 Python 自带的默认 handler

    # stderr 输出（必须有）
    stderr_handler = logging.StreamHandler(sys.stderr)  # 输出到标准错误
    stderr_handler.setFormatter(formatter)  # 设置格式
    root.addHandler(stderr_handler)  # 添加到 root logger

    # 文件输出（可选）
    if config.logging.file:
        log_path = Path(config.logging.file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
