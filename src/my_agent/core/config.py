from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── 默认值 ──
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7437
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FILE = "~/.my-agent/logs/core.log"
_DEFAULT_LOG_FORMAT = "text"
_DEFAULT_CONFIG_PATH = "~/.my-agent/config.toml"
_DEFAULT_MAX_STEPS = 20
_DEFAULT_MODEL = "deepseek-v4-pro"
_DEFAULT_TRACE_FILE = "~/.my-agent/traces/daemon.jsonl"
_DEFAULT_RUNS_DIR = "runs"
_DEFAULT_COMPACT_ENABLED = True
_DEFAULT_COMPACT_TOKEN_THRESHOLD = 120_000
_DEFAULT_TOOL_RESULT_LIMIT = 8_000
_DEFAULT_TOOL_RESULT_KEEP = 4_000
_DEFAULT_COMPACT_CONTEXT_RATIO = 0.0

# dotenv 来源：当前项目 .env 优先于全局 ~/.my-agent/.env（跨项目 fallback）
_DEFAULT_ENV_PATH = ".env"
_DEFAULT_GLOBAL_ENV_PATH = "~/.my-agent/.env"

# env 前缀：用项目名，防止跟其他程序的环境变量冲突
# 比如 PORT=7437 太泛了，MY_AGENT_PORT=7437 一看就知道是谁的
_ENV_PREFIX = "MY_AGENT"

DEEPSEEK_MODEL_OPTIONS: dict[str, str] = {
    "deepseek-v4-pro": "DeepSeek V4 Pro（更强，适合复杂任务）",
    "deepseek-v4-flash": "DeepSeek V4 Flash（更快，适合日常任务）",
}


def normalize_deepseek_model(value: str) -> str | None:
    """Return a supported DeepSeek model id, accepting short UI aliases."""
    aliases = {"pro": "deepseek-v4-pro", "flash": "deepseek-v4-flash"}
    model = aliases.get(value.strip().casefold(), value.strip())
    return model if model in DEEPSEEK_MODEL_OPTIONS else None


@dataclass
class LoggingConfig:
    level: str = _DEFAULT_LOG_LEVEL
    file: str = _DEFAULT_LOG_FILE
    format: str = _DEFAULT_LOG_FORMAT  # "text" | "json"


@dataclass
class AgentConfig:
    max_steps: int = _DEFAULT_MAX_STEPS


@dataclass
class LlmConfig:
    default_model: str = _DEFAULT_MODEL
    router: str = "static"  # "static" | "rule_based" (S4) | "cost_budget" (S6)


@dataclass
class TraceConfig:
    enabled: bool = True  # 是否开启 trace
    file: str = _DEFAULT_TRACE_FILE  # 写到哪里
    include_llm_payload: bool = True  # 是否记录完整 LLM messages/tools/response


@dataclass
class CompactConfig:
    enabled: bool = _DEFAULT_COMPACT_ENABLED
    token_threshold: int = _DEFAULT_COMPACT_TOKEN_THRESHOLD
    tool_result_limit: int = _DEFAULT_TOOL_RESULT_LIMIT
    tool_result_keep: int = _DEFAULT_TOOL_RESULT_KEEP
    context_ratio: float = _DEFAULT_COMPACT_CONTEXT_RATIO


@dataclass
class McpServerConfig:
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    host: str = "localhost"
    port: int = 3000


@dataclass
class McpConfig:
    servers: list[McpServerConfig] = field(default_factory=list)


@dataclass
class Config:
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    runs_dir: str = _DEFAULT_RUNS_DIR
    compact: CompactConfig = field(default_factory=CompactConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


# 加载项目 .env 与全局 ~/.my-agent/.env；返回实际加载成功的路径列表
# 优先级：OS 环境变量最高（override=False）；项目 .env 先加载所以优先于全局 .env；
# 全局 .my-agent/.env 作为跨项目 fallback（用于任意目录启动 daemon 的密钥）
def _load_dotenv_files() -> list[Path]:
    loaded: list[Path] = []
    project_env = Path(_DEFAULT_ENV_PATH)
    global_env = Path(_DEFAULT_GLOBAL_ENV_PATH).expanduser()

    for path in (project_env, global_env):
        if path.exists():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


# 构建并返回运行时配置：默认值 → TOML → .env → 环境变量（后者优先级最高）
def get_config() -> Config:
    config = Config()

    # .env 必须在读 MY_AGENT_CONFIG 之前加载，因为 .env 里可能指定了自定义配置路径
    _load_dotenv_files()

    config_path_str = os.environ.get(f"{_ENV_PREFIX}_CONFIG", _DEFAULT_CONFIG_PATH)
    config_path = Path(config_path_str).expanduser()

    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise SystemExit(f"Config parse error ({config_path}): {e}") from e
        _apply_toml(config, data)

    _apply_env(config)
    return config


# 将已解析的 TOML 根表写入 config；未知 key 直接退出
def _apply_toml(config: Config, data: dict[str, Any]) -> None:
    unknown = set(data.keys()) - {"core", "logging", "agent", "llm", "trace", "compact", "mcp"}
    if unknown:
        raise SystemExit(f"Unknown top-level config keys: {', '.join(sorted(unknown))}")

    if "core" in data:
        core = data["core"]
        if not isinstance(core, dict):
            raise SystemExit("Config error: [core] must be a table")
        if "host" in core:
            config.host = core["host"]
        if "port" in core:
            config.port = core["port"]
        if "runs_dir" in core:
            val = core["runs_dir"]
            if not isinstance(val, str):
                raise SystemExit("Config error: core.runs_dir must be a string")
            config.runs_dir = val

    if "logging" in data:
        log = data["logging"]
        if not isinstance(log, dict):
            raise SystemExit("Config error: [logging] must be a table")
        if "level" in log:
            config.logging.level = log["level"]
        if "file" in log:
            config.logging.file = log["file"]
        if "format" in log:
            config.logging.format = log["format"]

    if "agent" in data:
        agent = data["agent"]
        if not isinstance(agent, dict):
            raise SystemExit("Config error: [agent] must be a table")
        if "max_steps" in agent:
            val = agent["max_steps"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit("Config error: agent.max_steps must be a positive integer")
            config.agent.max_steps = val

    if "llm" in data:
        llm = data["llm"]
        if not isinstance(llm, dict):
            raise SystemExit("Config error: [llm] must be a table")
        if "default_model" in llm:
            val = llm["default_model"]
            if not isinstance(val, str):
                raise SystemExit("Config error: llm.default_model must be a string")
            config.llm.default_model = val
        if "router" in llm:
            val = llm["router"]
            if not isinstance(val, str):
                raise SystemExit("Config error: llm.router must be a string")
            config.llm.router = val

    if "trace" in data:
        trace = data["trace"]
        if not isinstance(trace, dict):
            raise SystemExit("Config error: [trace] must be a table")

        unknown_trace = set(trace.keys()) - {"enabled", "file", "include_llm_payload"}
        if unknown_trace:
            raise SystemExit(f"Unknown trace config keys: {', '.join(sorted(unknown_trace))}")

        if "enabled" in trace:
            val = trace["enabled"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: trace.enabled must be a boolean")
            config.trace.enabled = val

        if "file" in trace:
            val = trace["file"]
            if not isinstance(val, str):
                raise SystemExit("Config error: trace.file must be a string")
            config.trace.file = val

        if "include_llm_payload" in trace:
            val = trace["include_llm_payload"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: trace.include_llm_payload must be a boolean")
            config.trace.include_llm_payload = val

    if "compact" in data:
        compact = data["compact"]
        if not isinstance(compact, dict):
            raise SystemExit("Config error: [compact] must be a table")

        unknown_compact = set(compact.keys()) - {
            "enabled",
            "token_threshold",
            "tool_result_limit",
            "tool_result_keep",
            "context_ratio",
        }
        if unknown_compact:
            raise SystemExit(
                f"Unknown compact config keys: {', '.join(sorted(unknown_compact))}"
            )

        if "enabled" in compact:
            val = compact["enabled"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: compact.enabled must be a boolean")
            config.compact.enabled = val

        if "token_threshold" in compact:
            val = compact["token_threshold"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit("Config error: compact.token_threshold must be a positive integer")
            config.compact.token_threshold = val

        if "tool_result_limit" in compact:
            val = compact["tool_result_limit"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit(
                    "Config error: compact.tool_result_limit must be a positive integer"
                )
            config.compact.tool_result_limit = val

        if "tool_result_keep" in compact:
            val = compact["tool_result_keep"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit(
                    "Config error: compact.tool_result_keep must be a positive integer"
                )
            config.compact.tool_result_keep = val

        if "context_ratio" in compact:
            val = compact["context_ratio"]
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise SystemExit("Config error: compact.context_ratio must be between 0 and 1")
            config.compact.context_ratio = float(val)

    if "mcp" in data:
        mcp = data["mcp"]
        if not isinstance(mcp, dict):
            raise SystemExit("Config error: [mcp] must be a table")

        unknown_mcp = set(mcp.keys()) - {"servers"}
        if unknown_mcp:
            raise SystemExit(f"Unknown mcp config keys: {', '.join(sorted(unknown_mcp))}")

        servers_raw = mcp.get("servers", [])
        if not isinstance(servers_raw, list):
            raise SystemExit("Config error: mcp.servers must be an array")

        for i, item in enumerate(servers_raw):
            if not isinstance(item, dict):
                raise SystemExit(f"Config error: mcp.servers[{i}] must be a table")

            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise SystemExit(f"Config error: mcp.servers[{i}].name must be a non-empty string")

            transport = item.get("transport", "stdio")
            if transport not in ("stdio", "tcp"):
                raise SystemExit(
                    f"Config error: mcp.servers[{i}].transport must be 'stdio' or 'tcp'"
                )

            server = McpServerConfig(name=name, transport=transport)

            if "command" in item:
                val = item["command"]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: mcp.servers[{i}].command must be a string")
                server.command = val

            if "args" in item:
                val = item["args"]
                if not isinstance(val, list):
                    raise SystemExit(f"Config error: mcp.servers[{i}].args must be an array")
                server.args = [str(arg) for arg in val]

            if "env" in item:
                val = item["env"]
                if not isinstance(val, dict):
                    raise SystemExit(f"Config error: mcp.servers[{i}].env must be a table")
                server.env = {str(k): str(v) for k, v in val.items()}

            if "host" in item:
                val = item["host"]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: mcp.servers[{i}].host must be a string")
                server.host = val

            if "port" in item:
                val = item["port"]
                if not isinstance(val, int):
                    raise SystemExit(f"Config error: mcp.servers[{i}].port must be an integer")
                server.port = val

            config.mcp.servers.append(server)


# 用 MY_AGENT_* 环境变量覆盖 config 中对应字段
def _apply_env(config: Config) -> None:
    host = os.environ.get(f"{_ENV_PREFIX}_HOST")

    if host is not None:
        config.host = host

    port_str = os.environ.get(f"{_ENV_PREFIX}_PORT")
    if port_str is not None:
        try:
            config.port = int(port_str)
        except ValueError:
            raise SystemExit(
                f"Config error: {_ENV_PREFIX}_PORT must be an integer, got: {port_str!r}"
            )

    log_level = os.environ.get(f"{_ENV_PREFIX}_LOG_LEVEL")
    if log_level is not None:
        config.logging.level = log_level

    log_file = os.environ.get(f"{_ENV_PREFIX}_LOG_FILE")
    if log_file is not None:
        config.logging.file = log_file

    log_format = os.environ.get(f"{_ENV_PREFIX}_LOG_FORMAT")
    if log_format is not None:
        config.logging.format = log_format

    max_steps_str = os.environ.get(f"{_ENV_PREFIX}_MAX_STEPS")
    if max_steps_str is not None:
        try:
            val = int(max_steps_str)
            if val <= 0:
                raise SystemExit(
                    f"Config error: {_ENV_PREFIX}_MAX_STEPS must be positive,"
                    f" got: {max_steps_str!r}"
                )
            config.agent.max_steps = val
        except ValueError:
            raise SystemExit(
                f"Config error: {_ENV_PREFIX}_MAX_STEPS must be an integer, got: {max_steps_str!r}"
            )

    default_model = os.environ.get(f"{_ENV_PREFIX}_LLM_DEFAULT_MODEL")
    if default_model is not None:
        config.llm.default_model = default_model

    trace_enabled = os.environ.get(f"{_ENV_PREFIX}_TRACE_ENABLED")

    if trace_enabled is not None:
        config.trace.enabled = trace_enabled.lower() not in ("0", "false", "no")

    trace_file = os.environ.get(f"{_ENV_PREFIX}_TRACE_FILE")

    if trace_file is not None:
        config.trace.file = trace_file

    trace_payload = os.environ.get(f"{_ENV_PREFIX}_TRACE_INCLUDE_LLM_PAYLOAD")

    if trace_payload is not None:
        config.trace.include_llm_payload = trace_payload.lower() not in (
            "0",
            "false",
            "no",
        )
    runs_dir = os.environ.get(f"{_ENV_PREFIX}_RUNS_DIR")
    if runs_dir is not None:
        config.runs_dir = runs_dir

    compact_enabled = os.environ.get(f"{_ENV_PREFIX}_COMPACT_ENABLED")
    if compact_enabled is not None:
        config.compact.enabled = compact_enabled.lower() not in ("0", "false", "no")

    compact_token_threshold = os.environ.get(f"{_ENV_PREFIX}_COMPACT_TOKEN_THRESHOLD")
    if compact_token_threshold is not None:
        try:
            val = int(compact_token_threshold)
            if val <= 0:
                raise SystemExit(
                    f"Config error: {_ENV_PREFIX}_COMPACT_TOKEN_THRESHOLD must be positive,"
                    f" got: {compact_token_threshold!r}"
                )
            config.compact.token_threshold = val
        except ValueError:
            raise SystemExit(
                f"Config error: {_ENV_PREFIX}_COMPACT_TOKEN_THRESHOLD must be an integer,"
                f" got: {compact_token_threshold!r}"
            )

    compact_result_limit = os.environ.get(f"{_ENV_PREFIX}_COMPACT_TOOL_RESULT_LIMIT")
    if compact_result_limit is not None:
        try:
            val = int(compact_result_limit)
            if val <= 0:
                raise SystemExit(
                    f"Config error: {_ENV_PREFIX}_COMPACT_TOOL_RESULT_LIMIT must be positive,"
                    f" got: {compact_result_limit!r}"
                )
            config.compact.tool_result_limit = val
        except ValueError:
            raise SystemExit(
                f"Config error: {_ENV_PREFIX}_COMPACT_TOOL_RESULT_LIMIT must be an integer,"
                f" got: {compact_result_limit!r}"
            )

    compact_result_keep = os.environ.get(f"{_ENV_PREFIX}_COMPACT_TOOL_RESULT_KEEP")
    if compact_result_keep is not None:
        try:
            val = int(compact_result_keep)
            if val <= 0:
                raise SystemExit(
                    f"Config error: {_ENV_PREFIX}_COMPACT_TOOL_RESULT_KEEP must be positive,"
                    f" got: {compact_result_keep!r}"
                )
            config.compact.tool_result_keep = val
        except ValueError:
            raise SystemExit(
                f"Config error: {_ENV_PREFIX}_COMPACT_TOOL_RESULT_KEEP must be an integer,"
                f" got: {compact_result_keep!r}"
            )

    compact_context_ratio = os.environ.get(f"{_ENV_PREFIX}_COMPACT_CONTEXT_RATIO")
    if compact_context_ratio is not None:
        try:
            ratio_val = float(compact_context_ratio)
            if not (0.0 <= ratio_val <= 1.0):
                raise SystemExit(
                    f"Config error: {_ENV_PREFIX}_COMPACT_CONTEXT_RATIO must be between "
                    f"0 and 1, got: {compact_context_ratio!r}"
                )
            config.compact.context_ratio = ratio_val
        except ValueError:
            raise SystemExit(
                f"Config error: {_ENV_PREFIX}_COMPACT_CONTEXT_RATIO must be a number,"
                f" got: {compact_context_ratio!r}"
            )
