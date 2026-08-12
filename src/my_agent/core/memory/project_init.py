from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from my_agent.core.events.bus import EventBus
from my_agent.core.llm.base import LLMProvider
from my_agent.core.llm.types import LlmResponse


@dataclass
class InitResult:
    """`my-agent init` 执行结果，供 CLI 和 TUI 共同展示。"""

    messages: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)
    status: str = "success"


@dataclass
class ScanLimits:
    """扫描边界：避免递归读取整个仓库。"""

    max_file_size: int = 32_768
    max_total_bytes: int = 200_000
    max_files: int = 20
    max_dir_listing: int = 40


@dataclass
class ProjectScan:
    """确定性扫描结果。"""

    root: Path
    entries: list[str] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)
    dir_listings: dict[str, list[str]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    language: str = "unknown"
    framework: str = ""
    commands: list[str] = field(default_factory=list)
    sensitive_paths: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ProjectUnderstanding:
    """项目理解结果，可来自模型或确定性 fallback。"""

    source: str = "fallback"
    project_summary: str = ""
    languages_and_frameworks: str = ""
    repository_structure: str = ""
    key_files: str = ""
    commands: str = ""
    architecture_notes: str = ""
    coding_conventions: str = ""
    testing_notes: str = ""
    generated_or_sensitive_paths: str = ""
    known_unknowns: str = ""
    fact_labels: dict[str, str] = field(default_factory=dict)


# 顶层扫描时直接忽略的目录/文件名（生成物、缓存、虚拟环境等）
_IGNORED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "runs",
        "build",
        "dist",
        ".tox",
        ".nox",
        ".cache",
        ".tmp-npm-cache",
        ".uv-cache",
        ".uv-cache-e2e",
        ".uv-cache-e2e-client",
        ".uv-cache-e2e-core",
        ".superpowers",
        ".agents",
    }
)

# 常见关键文件或目录（存在时优先识别）
_KEY_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "Pipfile",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "README.md",
    "AGENTS.md",
    "TODO.md",
    "CONTRIBUTING.md",
    "Makefile",
    "justfile",
    "src",
    "tests",
    "docs",
    "app",
    "lib",
)

# 值得列出顶层子目录的代码/文档目录
_CODE_DIRS: frozenset[str] = frozenset({"src", "tests", "docs", "app", "lib"})

_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".ico",
        ".mp3",
        ".mp4",
        ".mov",
        ".pdf",
        ".docx",
        ".xlsx",
        ".parquet",
        ".db",
        ".sqlite",
    }
)

_SENSITIVE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".env",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".crt",
        ".cer",
        ".der",
    }
)

_SENSITIVE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\.env",
        r"secret",
        r"private[_\-]?key",
        r"api[_\-]?key",
        r"token",
        r"credential",
        r"certificate",
        r"id_rsa",
        r"id_dsa",
        r"id_ecdsa",
    )
)

_MEMORY_RULES = """## Memory update / merge rules

- Preserve human-edited content: when regenerating context, keep content
  that was manually added outside the managed sections.
- Re-running `my-agent init` must not duplicate paragraphs or sections.
- Project-specific facts go into `.my-agent/context.md`.
- Cross-project preferences go into `~/.my-agent/context.md`.
- On conflict, keep old content and append the new observation — never
  silently delete existing information.
- Never record secrets, API keys, tokens, or personally sensitive information.
- Mark uncertain facts as `tentative` / `needs verification`, not as certain conclusions."""


_MARKER_BEGIN = "<!-- MY-AGENT:BEGIN -->"
_MARKER_END = "<!-- MY-AGENT:END -->"


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_sensitive(name: str) -> bool:
    """根据文件名判断是否为敏感文件。"""
    if any(name.lower().endswith(ext) for ext in _SENSITIVE_EXTENSIONS):
        return True
    return any(pattern.search(name) for pattern in _SENSITIVE_NAME_PATTERNS)


def _is_binary(name: str) -> bool:
    """根据扩展名判断是否为二进制文件。"""
    return any(name.lower().endswith(ext) for ext in _BINARY_EXTENSIONS)


def _should_ignore_top_level(name: str, path: Path) -> bool:
    """判断顶层目录/文件是否属于生成物或缓存。"""
    if name in _IGNORED_TOP_LEVEL:
        return True
    if name.startswith(".") and path.is_dir():
        # 隐藏目录通常也是缓存/配置，保守跳过
        return True
    return False


def _read_text(path: Path, max_bytes: int) -> str | None:
    """以安全方式读取文本，遇到二进制或不可读文件返回 None。"""
    try:
        with path.open("rb") as f:
            sample = f.read(1024)
        if b"\x00" in sample:
            return None
        content = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError, PermissionError):
        return None
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        content = encoded[:max_bytes].decode("utf-8", errors="replace")
    return content


def _list_dir(path: Path, max_items: int) -> list[str]:
    """只列出目录的直接子项，不递归。"""
    try:
        items = sorted(p.name + "/" if p.is_dir() else p.name for p in path.iterdir())
    except (OSError, PermissionError):
        return []
    if len(items) > max_items:
        return items[:max_items] + [f"... ({len(items) - max_items} more)"]
    return items


def _parse_pyproject(text: str) -> dict[str, Any]:
    """解析 pyproject.toml，提取项目元信息。"""
    try:
        import tomllib
    except ImportError:
        return {}
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}
    project = data.get("project", {})
    result: dict[str, Any] = {}
    if "name" in project:
        result["name"] = project["name"]
    if "version" in project:
        result["version"] = project["version"]
    if "requires-python" in project:
        result["requires_python"] = project["requires-python"]
    if "dependencies" in project:
        deps = project["dependencies"]
        result["dependencies"] = deps[:20] if isinstance(deps, list) else []
    if "scripts" in project:
        result["scripts"] = project["scripts"]
    if "tool" in data:
        tools = data["tool"]
        if isinstance(tools, dict):
            if "pytest" in tools:
                result["has_pytest"] = True
            if "ruff" in tools:
                result["has_ruff"] = True
            if "mypy" in tools:
                result["has_mypy"] = True
    return result


def _parse_package_json(text: str) -> dict[str, Any]:
    """解析 package.json，提取项目元信息。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("name", "version", "scripts"):
        if key in data:
            result[key] = data[key]
    deps = data.get("dependencies", {})
    dev_deps = data.get("devDependencies", {})
    if isinstance(deps, dict):
        result["dependencies"] = list(deps.keys())[:20]
    if isinstance(dev_deps, dict):
        result["dev_dependencies"] = list(dev_deps.keys())[:20]
    return result


def _detect_language(scan: ProjectScan) -> str:
    """根据扫描结果推断主要语言。"""
    if "pyproject.toml" in scan.key_files or any(name.endswith(".py") for name in scan.entries):
        return "python"
    if "package.json" in scan.key_files or any(
        name.endswith((".js", ".ts", ".jsx", ".tsx")) for name in scan.entries
    ):
        return "node / javascript"
    if any(name.endswith((".java", ".kt")) for name in scan.entries):
        return "java / kotlin"
    if any(name.endswith(".go") for name in scan.entries):
        return "go"
    if any(name.endswith(".rs") for name in scan.entries):
        return "rust"
    return "unknown"


def _detect_framework(scan: ProjectScan) -> str:
    """根据 manifest 推断框架。"""
    manifest = scan.manifest
    if "pyproject.toml" in scan.key_files:
        deps = manifest.get("pyproject", {}).get("dependencies", [])
        deps_str = " ".join(deps).lower()
        if "fastapi" in deps_str:
            return "fastapi"
        if "flask" in deps_str:
            return "flask"
        if "django" in deps_str:
            return "django"
        if "pytest" in deps_str:
            return "pytest"
    if "package.json" in scan.key_files:
        deps = manifest.get("package_json", {}).get("dependencies", [])
        dev = manifest.get("package_json", {}).get("dev_dependencies", [])
        all_deps = set(deps + dev)
        if "express" in all_deps:
            return "express"
        if "react" in all_deps:
            return "react"
    return ""


def _infer_commands(scan: ProjectScan) -> list[str]:
    """根据项目类型推断开发命令。"""
    commands: list[str] = []
    language = scan.language.lower()
    if "python" in language:
        has_uv = "uv.lock" in scan.key_files or (
            "pyproject.toml" in scan.key_files and scan.manifest.get("pyproject", {}).get("name")
        )
        if has_uv:
            commands.extend(
                [
                    "uv sync --dev",
                    "uv run pytest",
                    "uv run ruff check src tests",
                    "uv run mypy src",
                ]
            )
        else:
            commands.extend(
                [
                    "pip install -e .",
                    "pytest",
                ]
            )
        if scan.manifest.get("pyproject", {}).get("has_ruff"):
            commands.append("ruff check src tests")
        if scan.manifest.get("pyproject", {}).get("has_mypy"):
            commands.append("mypy src")
    elif "node" in language:
        package = scan.manifest.get("package_json", {})
        scripts = package.get("scripts", {})
        commands.append("npm install")
        if "test" in scripts:
            commands.append("npm test")
        elif "tests" in scan.entries or "test" in scan.entries:
            commands.append("npm test")
        if "build" in scripts:
            commands.append("npm run build")
        if "start" in scripts:
            commands.append("npm start")
        if "lint" in scripts:
            commands.append("npm run lint")
    elif "java" in language or "kotlin" in language:
        commands.extend(["./gradlew build", "./gradlew test"])
    elif "go" in language:
        commands.extend(["go build ./...", "go test ./..."])
    elif "rust" in language:
        commands.extend(["cargo build", "cargo test"])
    if not commands:
        commands.append("Add project-specific test/build commands after confirming them.")
    return commands


def scan_project(root: Path, *, limits: ScanLimits | None = None) -> ProjectScan:
    """对项目根目录执行有边界的确定性扫描。"""
    limits = limits or ScanLimits()
    scan = ProjectScan(root=root)
    remaining_bytes = limits.max_total_bytes
    remaining_files = limits.max_files

    try:
        top_level = sorted(root.iterdir())
    except (OSError, PermissionError) as exc:
        scan.notes.append(f"Could not list project root: {exc}")
        return scan

    for path in top_level:
        name = path.name
        if path.is_symlink():
            scan.skipped.append(f"{name}: symlink")
            continue
        if _should_ignore_top_level(name, path):
            scan.skipped.append(f"{name}: ignored")
            continue
        if _is_sensitive(name):
            scan.skipped.append(f"{name}: sensitive")
            scan.sensitive_paths.append(name)
            continue

        if path.is_dir():
            scan.entries.append(name + "/")
            if name in _CODE_DIRS:
                scan.dir_listings[name] = _list_dir(path, limits.max_dir_listing)
            if name in _KEY_FILES:
                scan.key_files.append(name)
            continue

        if not path.is_file():
            continue

        scan.entries.append(name)
        if name in _KEY_FILES:
            scan.key_files.append(name)

        if _is_binary(name):
            scan.skipped.append(f"{name}: binary")
            continue

        try:
            size = path.stat().st_size
        except (OSError, PermissionError):
            scan.skipped.append(f"{name}: stat failed")
            continue

        if size > limits.max_file_size:
            scan.skipped.append(f"{name}: exceeds max file size ({size} bytes)")
            scan.notes.append(f"Skipped {name} due to size limit.")
            continue

        if remaining_files <= 0:
            scan.skipped.append(f"{name}: file budget exhausted")
            continue

        budget = min(size, remaining_bytes)
        if budget <= 0 or budget < size:
            scan.skipped.append(f"{name}: total byte budget exhausted")
            continue

        content = _read_text(path, max_bytes=budget)
        if content is None:
            scan.skipped.append(f"{name}: unreadable or binary")
            continue

        scan.file_contents[name] = content
        remaining_bytes -= len(content.encode("utf-8", errors="replace"))
        remaining_files -= 1

        # 解析 manifest
        if name == "pyproject.toml":
            scan.manifest["pyproject"] = _parse_pyproject(content)
        elif name == "package.json":
            scan.manifest["package_json"] = _parse_package_json(content)
        elif name == "requirements.txt":
            scan.manifest["requirements"] = content.splitlines()[:20]
        elif name == "README.md":
            scan.manifest["readme_excerpt"] = content[:2000]
        elif name == "AGENTS.md":
            scan.manifest["agents_excerpt"] = content[:2000]
        elif name == "TODO.md":
            scan.manifest["todo_excerpt"] = content[:2000]

    scan.language = _detect_language(scan)
    scan.framework = _detect_framework(scan)
    scan.commands = _infer_commands(scan)

    if not scan.entries:
        scan.notes.append("Project root appears empty or all entries were ignored.")

    return scan


def _build_prompt(scan: ProjectScan) -> str:
    """构建提交给 LLM 的项目分析 prompt。"""
    payload = {
        "root": str(scan.root.resolve()),
        "language": scan.language,
        "framework": scan.framework,
        "entries": scan.entries,
        "key_files": scan.key_files,
        "commands": scan.commands,
        "dir_listings": scan.dir_listings,
        "file_contents": scan.file_contents,
        "manifest": scan.manifest,
        "sensitive_paths": scan.sensitive_paths,
        "skipped": scan.skipped,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


_SYSTEM_PROMPT = """You are a project analyst for a coding assistant.
Analyze the repository scan provided below and return a single JSON object
with exactly these fields:

{
  "project_summary": "1-2 sentences describing what the project is",
  "languages_and_frameworks": "primary language and frameworks",
  "repository_structure": "concise summary of top-level layout",
  "key_files": "list of important files and their purpose",
  "commands": "install / test / lint / build commands, semicolon-separated",
  "architecture_notes": "how major modules interact",
  "coding_conventions": "style, linting, or naming conventions visible from files",
  "testing_notes": "how tests are organized and run",
  "generated_or_sensitive_paths": "paths that should not be edited or committed",
  "known_unknowns": "what is not clear from the scan and needs verification",
  "fact_labels": {
    "project_summary": "confirmed|inferred|needs_verification",
    ...
  }
}

Rules:
- Only use the files and facts listed in the scan. Do not invent files.
- Mark each field as "confirmed" if directly supported by the scan,
  "inferred" if reasonably deduced, or "needs_verification" if uncertain.
- Never include secrets, API keys, tokens, credentials, or private values.
- Keep answers concise but specific and actionable for a future coding agent."""


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON，兼容 markdown 代码块。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉首尾的 markdown 代码块标记
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return cast(dict[str, Any], json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    return None


_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "project_summary",
        "languages_and_frameworks",
        "repository_structure",
        "key_files",
        "commands",
        "architecture_notes",
        "coding_conventions",
        "testing_notes",
        "generated_or_sensitive_paths",
        "known_unknowns",
    }
)


def _fallback_understanding(
    scan: ProjectScan,
    *,
    source: str = "fallback",
    notes: list[str] | None = None,
) -> ProjectUnderstanding:
    """当模型不可用时，基于确定性扫描生成基础理解。"""
    labels = {field: "confirmed" for field in _REQUIRED_FIELDS}
    labels["architecture_notes"] = "needs_verification"
    labels["coding_conventions"] = "needs_verification"
    labels["testing_notes"] = "inferred"
    labels["known_unknowns"] = "needs_verification"

    return ProjectUnderstanding(
        source=source,
        project_summary=f"Project at {scan.root.name}: a {scan.language} project."
        if scan.language != "unknown"
        else f"Project at {scan.root.name}: language not detected.",
        languages_and_frameworks=scan.language
        + (f" ({scan.framework})" if scan.framework else ""),
        repository_structure=", ".join(scan.entries) or "(empty)",
        key_files=", ".join(scan.key_files) or "(none)",
        commands="; ".join(scan.commands) or "(unknown)",
        architecture_notes=(
            "Based on top-level scan only. "
            "Verify module relationships before making changes."
        ),
        coding_conventions=(
            "Infer from existing source files and project conventions."
        ),
        testing_notes=(
            "Run tests using the detected commands; "
            "verify test organization manually."
        ),
        generated_or_sensitive_paths=", ".join(scan.sensitive_paths) or "(none)",
        known_unknowns="; ".join(notes)
        if notes
        else "Detailed architecture and conventions not verified by model.",
        fact_labels=labels,
    )


def _understanding_from_dict(
    data: dict[str, Any], scan: ProjectScan, *, source: str
) -> ProjectUnderstanding:
    """把模型返回的 JSON 转换为结构化理解。"""
    labels = data.get("fact_labels", {})
    default_label = "inferred" if source == "model" else "needs_verification"
    return ProjectUnderstanding(
        source=source,
        project_summary=str(data.get("project_summary", "")),
        languages_and_frameworks=str(data.get("languages_and_frameworks", "")),
        repository_structure=str(data.get("repository_structure", "")),
        key_files=str(data.get("key_files", "")),
        commands=str(data.get("commands", "")),
        architecture_notes=str(data.get("architecture_notes", "")),
        coding_conventions=str(data.get("coding_conventions", "")),
        testing_notes=str(data.get("testing_notes", "")),
        generated_or_sensitive_paths=str(data.get("generated_or_sensitive_paths", "")),
        known_unknowns=str(data.get("known_unknowns", "")),
        fact_labels={
            field: str(labels.get(field, default_label)).lower() for field in _REQUIRED_FIELDS
        },
    )


async def understand_project(
    scan: ProjectScan,
    provider: LLMProvider | None,
    *,
    run_id: str = "project-init",
) -> ProjectUnderstanding:
    """使用 provider 或 fallback 生成项目理解。"""
    if provider is None:
        return _fallback_understanding(scan)

    bus = EventBus()
    response = await provider.chat(
        messages=[{"role": "user", "content": _build_prompt(scan)}],
        tool_schemas=[],
        bus=bus,
        run_id=run_id,
        system=_SYSTEM_PROMPT,
    )
    if not isinstance(response, LlmResponse) or not isinstance(response.text, str):
        return _fallback_understanding(
            scan, source="invalid", notes=["Provider returned an unexpected response type."]
        )

    parsed = _extract_json(response.text)
    if parsed is None:
        return _fallback_understanding(
            scan,
            source="invalid",
            notes=["Provider returned invalid JSON; using deterministic fallback."],
        )

    if not _REQUIRED_FIELDS.issubset(parsed.keys()):
        missing = _REQUIRED_FIELDS - parsed.keys()
        return _fallback_understanding(
            scan,
            source="invalid",
            notes=[f"Provider response missing fields: {', '.join(sorted(missing))}."],
        )

    return _understanding_from_dict(parsed, scan, source="model")


def _label_for(section: str, understanding: ProjectUnderstanding) -> str:
    return understanding.fact_labels.get(section, "inferred")


def _section_value(understanding: ProjectUnderstanding, section: str) -> str:
    """安全地读取 ProjectUnderstanding 的字符串字段。"""
    return getattr(understanding, section)  # type: ignore[no-any-return]


_CONTEXT_SECTION_TITLES: dict[str, str] = {
    "project_summary": "Project overview",
    "languages_and_frameworks": "Languages and frameworks",
    "repository_structure": "Repository structure",
    "key_files": "Key files",
    "commands": "Development commands",
    "architecture_notes": "Architecture notes",
    "coding_conventions": "Coding conventions",
    "testing_notes": "Testing notes",
    "generated_or_sensitive_paths": "Generated or sensitive paths",
    "known_unknowns": "Known unknowns / needs verification",
}


def write_context_md(root: Path, understanding: ProjectUnderstanding) -> bool:
    """生成或覆盖 `.my-agent/context.md`。"""
    context_path = root / ".my-agent" / "context.md"
    context_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Project Context",
        "",
        f"Project root: {root.resolve()}",
        f"Project name: {root.name}",
        f"Generated: {_now_utc()}",
        f"Source: {understanding.source}",
        "",
    ]
    for section, title in _CONTEXT_SECTION_TITLES.items():
        label = _label_for(section, understanding)
        value = _section_value(understanding, section)
        lines.append(f"## {title}\n\n({label}) {value}")
        lines.append("")
    lines.extend(
        [
            _MEMORY_RULES,
            "",
            "## Notes for future agent runs",
            "",
            "*(Add project-specific notes, conventions, or preferences here.)*",
            "",
        ]
    )
    context_path.write_text("\n".join(lines), encoding="utf-8")
    return True


_AGENTS_SECTION_TITLES: dict[str, str] = {
    "project_summary": "Project overview",
    "languages_and_frameworks": "Languages and frameworks",
    "repository_structure": "Repository structure",
    "key_files": "Key files",
    "commands": "Development commands",
    "architecture_notes": "Architecture notes",
    "coding_conventions": "Coding conventions",
    "testing_notes": "Testing notes",
    "generated_or_sensitive_paths": "Generated or sensitive paths",
    "known_unknowns": "Known unknowns",
}


def _build_agents_block(understanding: ProjectUnderstanding) -> str:
    lines = [
        _MARKER_BEGIN,
        "## Project Understanding",
        "",
        "Project context is stored in `.my-agent/context.md`.",
        "",
    ]
    for section, title in _AGENTS_SECTION_TITLES.items():
        label = _label_for(section, understanding)
        value = _section_value(understanding, section)
        lines.append(f"**{title}** ({label})  ")
        lines.append(value)
        lines.append("")
    lines.extend(
        [
            "<!-- Managed by `my-agent init`. Do not edit this block directly. -->",
            _MARKER_END,
        ]
    )
    return "\n".join(lines)


def update_agents_md(root: Path, understanding: ProjectUnderstanding) -> tuple[bool, bool]:
    """创建或保守更新 AGENTS.md 的 marker 管理区。"""
    agents_path = root / "AGENTS.md"
    block = _build_agents_block(understanding)
    created = False
    updated = False

    if agents_path.exists():
        original = agents_path.read_text(encoding="utf-8")
        if _MARKER_BEGIN in original and _MARKER_END in original:
            before = original.split(_MARKER_BEGIN)[0]
            after = original.split(_MARKER_END, 1)[1]
            new_content = before + block + after
        else:
            new_content = original.rstrip() + "\n\n" + block + "\n"
        agents_path.write_text(new_content, encoding="utf-8")
        updated = True
    else:
        agents_path.write_text("# Project Instructions\n\n" + block + "\n", encoding="utf-8")
        created = True

    return created, updated


class ProjectInitService:
    """CLI `my-agent init` 和 TUI `/init` 共享的项目初始化服务。"""

    def __init__(
        self,
        root: Path,
        *,
        provider: LLMProvider | None = None,
        limits: ScanLimits | None = None,
        run_id: str = "project-init",
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root
        self.provider = provider
        self.limits = limits or ScanLimits()
        self.run_id = run_id
        self.progress_callback = progress_callback

    def _report(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _files_already_exist(self) -> bool:
        return (self.root / ".my-agent" / "context.md").exists() or (
            self.root / "AGENTS.md"
        ).exists()

    async def run(self) -> InitResult:
        """执行完整 init 流程并返回人类可读结果。"""
        self._report("Scanning project structure...")
        try:
            scan = scan_project(self.root, limits=self.limits)
        except Exception as exc:
            return InitResult(
                status="failed",
                messages=[f"Failed to scan project: {exc}"],
            )

        self._report("Analyzing project...")
        try:
            understanding = await understand_project(scan, self.provider, run_id=self.run_id)
        except Exception as exc:
            understanding = _fallback_understanding(
                scan, notes=[f"Provider analysis failed: {exc}"]
            )

        # 如果模型输出非法且已有文件存在，不覆盖现有文件；
        # provider 不可用的 fallback 仍会生成基础结果。
        if understanding.source == "invalid" and self._files_already_exist():
            return InitResult(
                status="failed",
                messages=[
                    "Provider returned invalid output; existing files preserved.",
                    "Run with a valid DEEPSEEK_API_KEY to regenerate.",
                ],
            )

        self._report("Writing project guidance files...")
        try:
            write_context_md(self.root, understanding)
            agents_created, agents_updated = update_agents_md(self.root, understanding)
        except Exception as exc:
            return InitResult(
                status="failed",
                messages=[f"Failed to write guidance files: {exc}"],
            )

        created: list[str] = []
        updated: list[str] = []
        created.append(".my-agent/context.md")
        if agents_created:
            created.append("AGENTS.md")
        else:
            updated.append("AGENTS.md")

        messages = [
            f"Initialized project at {self.root.resolve()}",
        ]
        if created:
            messages.append(f"  Created: {', '.join(created)}")
        if updated:
            messages.append(f"  Updated: {', '.join(updated)}")
        if understanding.source == "fallback":
            messages.append("  Note: LLM provider unavailable; used deterministic fallback.")
        elif understanding.source == "invalid":
            messages.append(
                "  Note: LLM provider returned invalid output; used deterministic fallback."
            )

        return InitResult(
            status="success",
            messages=messages,
            files_created=created,
            files_updated=updated,
        )


def create_init_provider(model: str) -> LLMProvider | None:
    """如果 API key 存在，则返回一个 provider；否则返回 None。"""
    if os.environ.get("DEEPSEEK_API_KEY"):
        from my_agent.core.llm.provider import DeepSeekProvider

        return DeepSeekProvider(model)
    return None
