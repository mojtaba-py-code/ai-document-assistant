"""The authenticated caller, as the server (never the client) understands it."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from docassist.authz.permissions import Permission, permissions_for
from docassist.core.enums import Classification, Role
from docassist.core.errors import PermissionDenied
from docassist.db.session import DbContext


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: uuid.UUID
    org_id: uuid.UUID | None
    role: Role
    clearance: Classification
    session_id: uuid.UUID
    email: str = ""
    department_ids: frozenset[uuid.UUID] = field(default_factory=frozenset)
    managed_department_ids: frozenset[uuid.UUID] = field(default_factory=frozenset)
    ip_prefix: str | None = None

    @property
    def is_platform_admin(self) -> bool:
        return self.role is Role.PLATFORM_ADMIN

    @property
    def permissions(self) -> frozenset[Permission]:
        return permissions_for(self.role)

    def has(self, permission: Permission) -> bool:
        return permission in self.permissions

    def require(self, permission: Permission) -> None:
        if not self.has(permission):
            raise PermissionDenied(internal_detail=f"{self.role} lacks {permission}")

    def require_org(self) -> uuid.UUID:
        if self.org_id is None:
            raise PermissionDenied(internal_detail="tenant endpoint called without organization")
        return self.org_id

    @property
    def db_context(self) -> DbContext:
        return DbContext(org_id=self.org_id, user_id=self.user_id, platform=self.is_platform_admin)

    def access_fingerprint(self) -> str:
        """Stable digest of everything that determines document visibility (cache keys)."""
        import hashlib

        parts = [
            str(self.org_id),
            str(self.user_id),
            self.role.value,
            self.clearance.value,
            ",".join(sorted(str(d) for d in self.department_ids)),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
