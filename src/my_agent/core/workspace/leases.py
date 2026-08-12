from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic

from my_agent.core.workspace.identity import (
    CanonicalWorkspace,
    canonicalize_workspace,
    workspaces_overlap,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class TuiWorkspaceLease:
    lease_id: str
    canonical_workspace: CanonicalWorkspace
    owner_session_id: str
    owner_connection_id: str
    owner_client_type: str
    created_at: str
    last_heartbeat_monotonic: float
    state: str = "active"


class WorkspaceInUseError(RuntimeError):
    def __init__(
        self,
        requested: CanonicalWorkspace,
        active: TuiWorkspaceLease,
    ) -> None:
        self.requested = requested
        self.active = active
        super().__init__(
            "workspace_in_use: requested workspace overlaps active workspace "
            f"{active.canonical_workspace.path}"
        )


class WorkspaceLeaseStaleError(RuntimeError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"session_stale: workspace lease for {session_id} expired")


class WorkspaceLeaseRegistry:
    """Core-local, atomic ownership registry for active TUI workspaces."""

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], str] = _now,
        stale_after_seconds: float = 120.0,
    ) -> None:
        self._lock = asyncio.Lock()
        self._leases: dict[str, TuiWorkspaceLease] = {}
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._stale_after_seconds = stale_after_seconds

    async def acquire(
        self,
        workspace_root: str,
        *,
        owner_session_id: str,
        owner_connection_id: str,
        owner_client_type: str = "tui",
    ) -> TuiWorkspaceLease:
        requested = canonicalize_workspace(workspace_root)
        async with self._lock:
            self._reap_stale_unlocked(self._monotonic_clock())
            for active in self._leases.values():
                if active.state == "active" and workspaces_overlap(
                    requested, active.canonical_workspace
                ):
                    raise WorkspaceInUseError(requested, active)

            lease = TuiWorkspaceLease(
                lease_id=f"lease-{uuid.uuid4().hex[:12]}",
                canonical_workspace=requested,
                owner_session_id=owner_session_id,
                owner_connection_id=owner_connection_id,
                owner_client_type=owner_client_type,
                created_at=self._wall_clock(),
                last_heartbeat_monotonic=self._monotonic_clock(),
            )
            self._leases[lease.lease_id] = lease
            return lease

    async def heartbeat(self, session_id: str) -> TuiWorkspaceLease:
        async with self._lock:
            now = self._monotonic_clock()
            lease = next(
                (
                    candidate
                    for candidate in self._leases.values()
                    if candidate.owner_session_id == session_id
                ),
                None,
            )
            if lease is None or lease.state != "active":
                raise WorkspaceLeaseStaleError(session_id)
            if now - lease.last_heartbeat_monotonic >= self._stale_after_seconds:
                self._release(lease)
                raise WorkspaceLeaseStaleError(session_id)
            lease.last_heartbeat_monotonic = now
            return lease

    async def reap_stale(self) -> list[str]:
        async with self._lock:
            return self._reap_stale_unlocked(self._monotonic_clock())

    async def release_session(self, session_id: str) -> bool:
        async with self._lock:
            return self._release_matching(
                lambda lease: lease.owner_session_id == session_id
            )

    async def bind_session(self, lease_id: str, session_id: str) -> bool:
        async with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.state != "active":
                return False
            lease.owner_session_id = session_id
            return True

    async def release_lease(self, lease_id: str) -> bool:
        async with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            self._release(lease)
            return True

    async def release_connection(self, connection_id: str) -> int:
        async with self._lock:
            released = 0
            for lease in list(self._leases.values()):
                if lease.owner_connection_id == connection_id:
                    self._release(lease)
                    released += 1
            return released

    async def active_leases(self) -> list[TuiWorkspaceLease]:
        async with self._lock:
            return [lease for lease in self._leases.values() if lease.state == "active"]

    def _release_matching(self, predicate: Callable[[TuiWorkspaceLease], bool]) -> bool:
        for lease in list(self._leases.values()):
            if predicate(lease):
                self._release(lease)
                return True
        return False

    def _release(self, lease: TuiWorkspaceLease) -> None:
        lease.state = "released"
        self._leases.pop(lease.lease_id, None)

    def _reap_stale_unlocked(self, now: float) -> list[str]:
        stale_sessions: list[str] = []
        for lease in list(self._leases.values()):
            if now - lease.last_heartbeat_monotonic >= self._stale_after_seconds:
                stale_sessions.append(lease.owner_session_id)
                self._release(lease)
        return stale_sessions
