from __future__ import annotations

import asyncio

import pytest

from my_agent.core.workspace.leases import WorkspaceInUseError, WorkspaceLeaseRegistry


@pytest.mark.asyncio
async def test_workspace_lease_rejects_overlapping_workspaces(tmp_path) -> None:
    registry = WorkspaceLeaseRegistry()
    parent = tmp_path / "project"
    child = parent / "nested"

    first = await registry.acquire(
        str(parent),
        owner_session_id="sess-1",
        owner_connection_id="conn-1",
    )

    with pytest.raises(WorkspaceInUseError) as exc_info:
        await registry.acquire(
            str(child),
            owner_session_id="sess-2",
            owner_connection_id="conn-2",
        )

    assert exc_info.value.active is first
    assert exc_info.value.active.owner_session_id == "sess-1"


@pytest.mark.asyncio
async def test_workspace_lease_release_allows_new_owner(tmp_path) -> None:
    registry = WorkspaceLeaseRegistry()
    workspace = str(tmp_path / "project")

    await registry.acquire(
        workspace,
        owner_session_id="sess-1",
        owner_connection_id="conn-1",
    )
    assert await registry.release_session("sess-1")

    replacement = await registry.acquire(
        workspace,
        owner_session_id="sess-2",
        owner_connection_id="conn-2",
    )
    assert replacement.owner_session_id == "sess-2"


@pytest.mark.asyncio
async def test_workspace_lease_release_connection_is_idempotent(tmp_path) -> None:
    registry = WorkspaceLeaseRegistry()
    await registry.acquire(
        str(tmp_path / "project"),
        owner_session_id="sess-1",
        owner_connection_id="conn-1",
    )

    assert await registry.release_connection("conn-1") == 1
    assert await registry.release_connection("conn-1") == 0


@pytest.mark.asyncio
async def test_workspace_lease_allows_unrelated_workspaces(tmp_path) -> None:
    registry = WorkspaceLeaseRegistry()

    first = await registry.acquire(
        str(tmp_path / "tank"),
        owner_session_id="sess-1",
        owner_connection_id="conn-1",
    )
    second = await registry.acquire(
        str(tmp_path / "tetris"),
        owner_session_id="sess-2",
        owner_connection_id="conn-2",
    )

    assert first.owner_session_id == "sess-1"
    assert second.owner_session_id == "sess-2"


@pytest.mark.asyncio
async def test_workspace_lease_admission_is_atomic_for_concurrent_requests(tmp_path) -> None:
    registry = WorkspaceLeaseRegistry()
    workspace = str(tmp_path / "project")

    async def acquire(session_id: str) -> str:
        try:
            await registry.acquire(
                workspace,
                owner_session_id=session_id,
                owner_connection_id=f"conn-{session_id}",
            )
        except WorkspaceInUseError:
            return "rejected"
        return "accepted"

    results = await asyncio.gather(acquire("sess-1"), acquire("sess-2"))

    assert sorted(results) == ["accepted", "rejected"]


@pytest.mark.asyncio
async def test_workspace_lease_heartbeat_refreshes_and_reaps_stale_owner(tmp_path) -> None:
    clock = [0.0]
    registry = WorkspaceLeaseRegistry(
        monotonic_clock=lambda: clock[0],
        stale_after_seconds=10.0,
    )
    workspace = str(tmp_path / "project")
    await registry.acquire(
        workspace,
        owner_session_id="sess-1",
        owner_connection_id="conn-1",
    )

    clock[0] = 9.0
    await registry.heartbeat("sess-1")
    clock[0] = 18.0
    await registry.heartbeat("sess-1")
    clock[0] = 29.0
    assert await registry.reap_stale() == ["sess-1"]

    replacement = await registry.acquire(
        workspace,
        owner_session_id="sess-2",
        owner_connection_id="conn-2",
    )
    assert replacement.owner_session_id == "sess-2"
