from __future__ import annotations

import re
from pathlib import Path


class WorkspacePathError(ValueError):
    pass


# P6: workspace security boundary.
# Subclass of WorkspacePathError so existing tool error handlers continue to work.
class WorkspaceSecurityError(WorkspacePathError):
    pass


# Directories denied anywhere under workspace_root.
# Protects My Agent internals, VCS metadata, run/session traces, and common
# cache/virtualenv directories. Project application log directories such as
# `logs/` are intentionally NOT denied here.
DENIED_DIR_NAMES = frozenset(
    {
        ".my-agent",
        ".git",
        "runs",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tmp-npm-cache",
    }
)

# Lowercase stems that indicate a secret/credential/token/certificate/private key.
# Matched as word-bounded substrings of the filename via _SENSITIVE_TOKEN_RE so
# `api_key.txt`, `client-secret.json`, `access_token.yaml`, `id_rsa` all hit,
# while `monkeytoken.py` (no word boundary) does not.
_SENSITIVE_TOKENS = (
    "secret",
    "token",
    "credential",
    "password",
    "passwd",
    "api[_-]?key",
    "access[_-]?key",
    "secret[_-]?key",
    "private[_-]?key",
    "id[_-]?rsa",
    "id[_-]?dsa",
    "id[_-]?ecdsa",
    "id[_-]?ed25519",
)

# Suffixes that always indicate a private key / certificate bundle.
_SENSITIVE_SUFFIXES = (
    ".pem",
    ".key",
    ".crt",
    ".cer",
    ".der",
    ".p12",
    ".pfx",
    ".keystore",
    ".jks",
)

_SENSITIVE_TOKEN_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:" + "|".join(_SENSITIVE_TOKENS) + r")(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

# Human-readable summary used by tests and diagnostics. Not parsed back.
SENSITIVE_FILENAME_PATTERNS = [
    ".env",
    ".env.* (e.g. .env.local, .env.production)",
    "*.pem (e.g. client.pem)",
    "*.key (e.g. secret.key)",
    "*.crt, *.cer, *.der, *.p12, *.pfx, *.keystore, *.jks",
    "id_rsa, id_dsa, id_ecdsa, id_ed25519",
    "filenames containing: secret, token, credential, password, passwd, "
    "api_key, access_key, secret_key, private_key",
]


def workspace_root_or_cwd(workspace_root: Path | str | None = None) -> Path:
    if workspace_root is None or str(workspace_root).strip() == "":
        return Path.cwd().resolve()
    return Path(workspace_root).expanduser().resolve()


def _is_sensitive_filename(path: Path) -> bool:
    # 返回 True 当且仅当文件名表明这是 secret / 证书 / 私钥 / 凭证类文件
    name = path.name.lower()
    if name.startswith(".env"):
        return True
    if name.endswith(_SENSITIVE_SUFFIXES):
        return True
    return bool(_SENSITIVE_TOKEN_RE.search(name))


def is_denied_internal_path(workspace_root: Path | str | None, resolved: Path) -> bool:
    # 判断已 resolve 后的绝对路径是否落在 workspace 内部敏感/敏感目录上；
    # 对 workspace 之外的路径不下结论（由 resolve_workspace_path 单独拦截）。
    root = workspace_root_or_cwd(workspace_root)
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return False
    denied_dir_names = {name.lower() for name in DENIED_DIR_NAMES}
    for part in rel.parts:
        if part.lower() in denied_dir_names:
            return True
    return _is_sensitive_filename(resolved)


def resolve_workspace_path(workspace_root: Path | str | None, user_path: str) -> Path:
    # 解析用户输入路径，强制落在 workspace_root 内部；并额外拒绝内部敏感路径
    root = workspace_root_or_cwd(workspace_root)
    path = Path(user_path).expanduser()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError(f"path outside workspace: {user_path}") from exc
    if is_denied_internal_path(root, resolved):
        raise WorkspaceSecurityError(f"access to internal path denied: {user_path}")
    return resolved
