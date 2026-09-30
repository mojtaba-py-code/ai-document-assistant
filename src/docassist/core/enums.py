"""Domain vocabularies shared by every layer (DB constraints, API schemas, policy)."""

from __future__ import annotations

from enum import StrEnum


class Classification(StrEnum):
    """Security classification of a document, ordered from least to most sensitive."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"

    @property
    def rank(self) -> int:
        return _CLASSIFICATION_RANK[self]

    @classmethod
    def at_most(cls, ceiling: Classification) -> list[Classification]:
        return [c for c in cls if c.rank <= ceiling.rank]

    @classmethod
    def highest(cls, values: list[Classification]) -> Classification:
        return max(values, key=lambda c: c.rank) if values else cls.PUBLIC


_CLASSIFICATION_RANK = {
    Classification.PUBLIC: 0,
    Classification.INTERNAL: 1,
    Classification.CONFIDENTIAL: 2,
    Classification.RESTRICTED: 3,
}


class Role(StrEnum):
    PLATFORM_ADMIN = "platform_admin"
    ORGANIZATION_ADMIN = "organization_admin"
    DEPARTMENT_MANAGER = "department_manager"
    EMPLOYEE = "employee"
    AUDITOR = "auditor"


class UserStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class OrganizationStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class VersionStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class DocumentType(StrEnum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    POLICY = "policy"
    HR = "hr"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    REPORT = "report"
    OTHER = "other"


class GranteeType(StrEnum):
    USER = "user"
    DEPARTMENT = "department"
    ROLE = "role"


class GrantPermission(StrEnum):
    READ = "read"
    MANAGE = "manage"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD = "dead"
    CANCELLED = "cancelled"


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    DENIED = "denied"
    FAILURE = "failure"


class ExportStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    EXPIRED = "expired"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
