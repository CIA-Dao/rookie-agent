from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from my_agent.core.tools.base import BaseTool, ToolResult
from my_agent.core.tools.workspace import WorkspacePathError, resolve_workspace_path

_MAX_CHUNK_CHARS = 12_000


@dataclass
class _Transaction:
    target: Path
    temporary: Path
    total_chunks: int
    next_chunk: int
    digest: hashlib._Hash
    expected_sha256: str | None
    expected_source_sha256: str | None


class ChunkedWriteStore:
    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = workspace_root
        self._transactions: dict[str, _Transaction] = {}

    def _target(self, path_str: str) -> Path:
        return resolve_workspace_path(self._workspace_root, path_str)

    @staticmethod
    def _key(path: Path) -> str:
        return str(path).casefold()

    def begin(
        self,
        path_str: str,
        total_chunks: int,
        expected_sha256: str | None,
        expected_source_sha256: str | None = None,
    ) -> ToolResult:
        if total_chunks <= 0:
            return ToolResult(
                content="total_chunks must be a positive integer",
                is_error=True,
                error_type="schema_error",
            )
        if expected_sha256 is not None and len(expected_sha256) != 64:
            return ToolResult(
                content="expected_sha256 must be a 64-character hex digest",
                is_error=True,
                error_type="schema_error",
            )
        if expected_source_sha256 is not None and len(expected_source_sha256) != 64:
            return ToolResult(
                content="expected_source_sha256 must be a 64-character hex digest",
                is_error=True,
                error_type="schema_error",
            )
        try:
            target = self._target(path_str)
        except WorkspacePathError as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")
        key = self._key(target)
        if key in self._transactions:
            return ToolResult(
                content=f"chunked write already active for {path_str}",
                is_error=True,
                error_type="runtime_error",
            )
        if expected_source_sha256 is not None:
            if not target.exists():
                return ToolResult(
                    content=f"source changed before write: target does not exist for {path_str}",
                    is_error=True,
                    error_type="runtime_error",
                )
            try:
                source_digest = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError as exc:
                return ToolResult(
                    content=f"cannot verify source before write: {exc}",
                    is_error=True,
                    error_type="runtime_error",
                )
            if source_digest != expected_source_sha256:
                return ToolResult(
                    content=(
                        f"source changed before write: expected {expected_source_sha256}, "
                        f"got {source_digest}"
                    ),
                    is_error=True,
                    error_type="runtime_error",
                )
        temporary = target.with_name(target.name + ".my-agent-partial")
        try:
            temporary.unlink(missing_ok=True)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.touch()
        except OSError as exc:
            return ToolResult(
                content=f"cannot start chunked write: {exc}",
                is_error=True,
                error_type="runtime_error",
            )
        self._transactions[key] = _Transaction(
            target=target,
            temporary=temporary,
            total_chunks=total_chunks,
            next_chunk=0,
            digest=hashlib.sha256(),
            expected_sha256=expected_sha256,
            expected_source_sha256=expected_source_sha256,
        )
        return ToolResult(
            content=f"started chunked write for {path_str}: {total_chunks} chunks"
        )

    def chunk(self, path_str: str, chunk_index: int, content: str) -> ToolResult:
        if len(content) > _MAX_CHUNK_CHARS:
            return ToolResult(
                content=f"chunk too large: {len(content)} characters (limit {_MAX_CHUNK_CHARS})",
                is_error=True,
                error_type="schema_error",
            )
        try:
            target = self._target(path_str)
        except WorkspacePathError as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")
        transaction = self._transactions.get(self._key(target))
        if transaction is None:
            return ToolResult(
                content=f"no active chunked write for {path_str}",
                is_error=True,
                error_type="runtime_error",
            )
        if chunk_index != transaction.next_chunk:
            return ToolResult(
                content=(
                    f"expected chunk_index {transaction.next_chunk}, got {chunk_index}"
                ),
                is_error=True,
                error_type="schema_error",
            )
        encoded = content.encode("utf-8")
        try:
            with transaction.temporary.open("ab") as handle:
                handle.write(encoded)
        except OSError as exc:
            return ToolResult(
                content=f"chunk write error: {exc}",
                is_error=True,
                error_type="runtime_error",
            )
        transaction.digest.update(encoded)
        transaction.next_chunk += 1
        return ToolResult(
            content=(
                f"wrote chunk {chunk_index + 1}/{transaction.total_chunks} "
                f"({len(encoded)} bytes) for {path_str}"
            )
        )

    def commit(self, path_str: str) -> ToolResult:
        try:
            target = self._target(path_str)
        except WorkspacePathError as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")
        key = self._key(target)
        transaction = self._transactions.get(key)
        if transaction is None:
            return ToolResult(
                content=f"no active chunked write for {path_str}",
                is_error=True,
                error_type="runtime_error",
            )
        if transaction.next_chunk != transaction.total_chunks:
            return ToolResult(
                content=(
                    f"cannot commit incomplete write: received "
                    f"{transaction.next_chunk}/{transaction.total_chunks} chunks"
                ),
                is_error=True,
                error_type="runtime_error",
            )
        digest = transaction.digest.hexdigest()
        if transaction.expected_sha256 and digest != transaction.expected_sha256:
            return ToolResult(
                content=f"sha256 mismatch: expected {transaction.expected_sha256}, got {digest}",
                is_error=True,
                error_type="runtime_error",
            )
        try:
            os.replace(transaction.temporary, transaction.target)
        except OSError as exc:
            return ToolResult(
                content=f"chunked commit error: {exc}",
                is_error=True,
                error_type="runtime_error",
            )
        del self._transactions[key]
        return ToolResult(content=f"committed chunked write to {path_str} sha256={digest}")


class WriteFileBeginTool(BaseTool):
    name = "write_file_begin"
    description = (
        "Start a transactional chunked file write. Use for large files; "
        "write_file remains preferable for small files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative target file path."},
            "total_chunks": {"type": "integer", "description": "Expected chunk count."},
            "expected_sha256": {
                "type": "string",
                "description": "Optional SHA-256 of the complete UTF-8 file.",
            },
            "expected_source_sha256": {
                "type": "string",
                "description": "Optional SHA-256 observed while reading the original target.",
            },
        },
        "required": ["path", "total_chunks"],
    }

    def __init__(self, store: ChunkedWriteStore) -> None:
        self._store = store

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return self._store.begin(
            str(params.get("path") or ""),
            int(str(params.get("total_chunks") or 0)),
            str(params["expected_sha256"]) if params.get("expected_sha256") else None,
            str(params["expected_source_sha256"])
            if params.get("expected_source_sha256")
            else None,
        )


class WriteFileChunkTool(BaseTool):
    name = "write_file_chunk"
    description = (
        f"Append one ordered UTF-8 chunk (maximum {_MAX_CHUNK_CHARS} characters) "
        "to an active transactional file write."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "chunk_index": {"type": "integer", "description": "Zero-based chunk index."},
            "content": {"type": "string"},
        },
        "required": ["path", "chunk_index", "content"],
    }

    def __init__(self, store: ChunkedWriteStore) -> None:
        self._store = store

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return self._store.chunk(
            str(params.get("path") or ""),
            int(str(params.get("chunk_index") or 0)),
            str(params.get("content") or ""),
        )


class WriteFileCommitTool(BaseTool):
    name = "write_file_commit"
    description = "Atomically commit a completed transactional chunked file write."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, store: ChunkedWriteStore) -> None:
        self._store = store

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return self._store.commit(str(params.get("path") or ""))
