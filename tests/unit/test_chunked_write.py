import hashlib
from pathlib import Path

from my_agent.core.tools.builtin.chunked_write import (
    ChunkedWriteStore,
    WriteFileBeginTool,
    WriteFileChunkTool,
    WriteFileCommitTool,
)


async def test_chunked_write_commits_exact_content_atomically(tmp_path: Path) -> None:
    store = ChunkedWriteStore(tmp_path)
    begin = WriteFileBeginTool(store)
    chunk = WriteFileChunkTool(store)
    commit = WriteFileCommitTool(store)
    content = "第一块\n第二块\n第三块\n"
    digest = hashlib.sha256(content.encode()).hexdigest()

    assert not (tmp_path / "src" / "game.js").exists()
    assert not (
        await begin.invoke(
            {"path": "src/game.js", "total_chunks": 3, "expected_sha256": digest}
        )
    ).is_error
    assert not (
        await chunk.invoke({"path": "src/game.js", "chunk_index": 0, "content": "第一块\n"})
    ).is_error
    assert not (
        await chunk.invoke({"path": "src/game.js", "chunk_index": 1, "content": "第二块\n"})
    ).is_error
    assert not (
        await chunk.invoke({"path": "src/game.js", "chunk_index": 2, "content": "第三块\n"})
    ).is_error
    assert not (tmp_path / "src" / "game.js").exists()

    result = await commit.invoke({"path": "src/game.js"})

    assert not result.is_error
    assert (tmp_path / "src" / "game.js").read_text(encoding="utf-8") == content
    assert not (tmp_path / "src" / "game.js.my-agent-partial").exists()


async def test_chunked_write_rejects_out_of_order_and_incomplete_commit(tmp_path: Path) -> None:
    store = ChunkedWriteStore(tmp_path)
    begin = WriteFileBeginTool(store)
    chunk = WriteFileChunkTool(store)
    commit = WriteFileCommitTool(store)

    assert not (await begin.invoke({"path": "game.js", "total_chunks": 2})).is_error
    out_of_order = await chunk.invoke(
        {"path": "game.js", "chunk_index": 1, "content": "second"}
    )
    assert out_of_order.is_error
    assert "expected chunk_index 0" in out_of_order.content
    incomplete = await commit.invoke({"path": "game.js"})
    assert incomplete.is_error
    assert "0/2" in incomplete.content
    assert not (tmp_path / "game.js").exists()


async def test_chunked_write_rejects_large_chunk_and_path_traversal(tmp_path: Path) -> None:
    store = ChunkedWriteStore(tmp_path)
    begin = WriteFileBeginTool(store)
    chunk = WriteFileChunkTool(store)

    outside = await begin.invoke({"path": "../escape.txt", "total_chunks": 1})
    assert outside.is_error
    assert not (tmp_path.parent / "escape.txt").exists()

    assert not (await begin.invoke({"path": "small.txt", "total_chunks": 1})).is_error
    large = await chunk.invoke(
        {"path": "small.txt", "chunk_index": 0, "content": "x" * 12_001}
    )
    assert large.is_error
    assert "chunk too large" in large.content


async def test_chunked_write_rejects_changed_source_identity(tmp_path: Path) -> None:
    target = tmp_path / "engine.js"
    target.write_bytes(b"original")
    source_digest = hashlib.sha256(b"original").hexdigest()
    target.write_bytes(b"changed")

    result = await WriteFileBeginTool(ChunkedWriteStore(tmp_path)).invoke(
        {
            "path": "engine.js",
            "total_chunks": 1,
            "expected_source_sha256": source_digest,
        }
    )

    assert result.is_error
    assert "source changed" in result.content
    assert target.read_bytes() == b"changed"
