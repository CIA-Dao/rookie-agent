from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from my_agent.core.runtime.windows import WindowsRuntime

_SOURCE_SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".vue", ".mjs", ".cjs"}
_IMPORT_RE = re.compile(
    r"(?:\bfrom\s*|\bimport\s*\(|\bimport\s*)[\"']([^\"']+)[\"']"
)


@dataclass
class DeliveryVerification:
    checked: bool
    passed: bool
    failures: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return "; ".join(self.failures)


@dataclass
class DeliveryResult:
    write_status: str
    artifact_status: str
    import_status: str
    build_status: str
    final_status: str
    failures: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "write_status": self.write_status,
            "artifact_status": self.artifact_status,
            "import_status": self.import_status,
            "build_status": self.build_status,
            "final_status": self.final_status,
            "failures": list(self.failures),
            "changed_files": list(self.changed_files),
        }


def aggregate_delivery_result(
    *,
    context_status: str,
    context_reason: str | None,
    verification: DeliveryVerification,
    manifest: Mapping[str, object],
) -> DeliveryResult:
    changed_paths: list[str] = []
    for key in ("created", "modified"):
        paths = manifest.get(key)
        if isinstance(paths, list):
            changed_paths.extend(path for path in paths if isinstance(path, str))
    changed = sorted(
        set(changed_paths)
    )
    failures = list(verification.failures)
    if context_reason and context_reason not in failures:
        failures.append(context_reason)
    checked = verification.checked
    import_failures = [failure for failure in verification.failures if "imports" in failure]
    build_failures = [failure for failure in verification.failures if "build" in failure]
    return DeliveryResult(
        write_status="success" if changed else "not_attempted",
        artifact_status="changed" if changed else "unchanged",
        import_status="failed" if import_failures else ("success" if checked else "skipped"),
        build_status="failed" if build_failures else ("success" if checked else "skipped"),
        final_status=(
            "accepted"
            if context_status == "success" and checked and verification.passed
            else "not_accepted"
        ),
        failures=failures,
        changed_files=changed,
    )


def is_project_delivery_goal(goal: str) -> bool:
    text = goal.casefold()
    project_words = ("项目", "project", "应用", "app", "网站", "web", "vue", "vite")
    action_words = ("创建", "新建", "生成", "搭建", "实现", "create", "build", "scaffold")
    return any(word in text for word in project_words) and any(
        word in text for word in action_words
    )


def _resolve_import(root: Path, source: Path, specifier: str) -> Path | None:
    if not specifier.startswith("."):
        return None
    base = (source.parent / specifier).resolve()
    candidates = [base]
    if base.suffix:
        candidates.extend(base.parent / f"{base.stem}{suffix}" for suffix in _SOURCE_SUFFIXES)
    else:
        candidates.extend(base.with_suffix(suffix) for suffix in _SOURCE_SUFFIXES)
    candidates.extend(base / f"index{suffix}" for suffix in _SOURCE_SUFFIXES)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _check_imports(root: Path) -> list[str]:
    failures: list[str] = []
    for source in root.rglob("*"):
        if not source.is_file() or "node_modules" in source.parts:
            continue
        if source.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for specifier in _IMPORT_RE.findall(text):
            if specifier.startswith(".") and _resolve_import(root, source, specifier) is None:
                relative = source.relative_to(root).as_posix()
                failures.append(f"{relative} imports missing {specifier}")
    return failures


async def _check_web_build(root: Path) -> list[str]:
    package_file = root / "package.json"
    if not package_file.is_file():
        return []
    try:
        package = json.loads(package_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid package.json: {exc}"]
    scripts = package.get("scripts", {})
    if not isinstance(scripts, dict) or not isinstance(scripts.get("build"), str):
        return []
    command = WindowsRuntime(root).normalize_argv(["npm", "run", "build"])
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=60)
    except FileNotFoundError:
        return ["npm is not available; web build was not run"]
    except TimeoutError:
        process.kill()
        await process.communicate()
        return ["npm run build timed out after 60s"]
    if process.returncode:
        text = output.decode("utf-8", errors="replace").strip().splitlines()
        detail = text[-1] if text else f"exit code {process.returncode}"
        return [f"npm run build failed: {detail}"]
    return []


async def verify_project_delivery(root: Path | str, goal: str) -> DeliveryVerification:
    workspace = Path(root)
    if not workspace.is_dir() or not is_project_delivery_goal(goal):
        return DeliveryVerification(checked=False, passed=True)
    failures = _check_imports(workspace)
    failures.extend(await _check_web_build(workspace))
    return DeliveryVerification(checked=True, passed=not failures, failures=failures)
