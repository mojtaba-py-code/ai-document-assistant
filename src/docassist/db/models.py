"""ORM models.

Tenant-isolation rules encoded in the schema (defence in depth on top of RLS):

* every tenant-owned row carries ``organization_id``;
* parents expose ``UNIQUE (organization_id, id)`` and children reference them through
  **composite foreign keys** ``(organization_id, parent_id)`` - the database itself refuses a
  row that links a document in organisation A to a department/user/chunk of organisation B;
* deletes are ``RESTRICT`` by default (no accidental cascades) except for pure child data
  (chunks, embeddings, grants) that must disappear with the version/document it belongs to.

Row-level security policies, roles, grants and triggers live in the Alembic migration.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from docassist.core.enums import (
    AuditOutcome,
    Classification,
    DocumentStatus,
    DocumentType,
    ExportStatus,
    GranteeType,
    GrantPermission,
    JobStatus,
    MessageRole,
    OrganizationStatus,
    Role,
    UserStatus,
    VersionStatus,
)
from docassist.db.base import Base, created_at_col, enum_check, updated_at_col, uuid_pk

UUIDT = UUID(as_uuid=True)
TS = DateTime(timezone=True)


def _values(enum: Any) -> list[str]:
    return [member.value for member in enum]


# --------------------------------------------------------------------------- #
# Tenancy & identity
# --------------------------------------------------------------------------- #
class Organization(Base):
    __tablename__ = "organizations"
    __table_args__ = (
        CheckConstraint(enum_check("status", _values(OrganizationStatus)), name="status"),
        CheckConstraint(r"slug ~ '^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$'", name="slug_format"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    slug: Mapped[str] = mapped_column(String(63), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=OrganizationStatus.ACTIVE
    )
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()


class Department(Base):
    __tablename__ = "departments"
    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        UniqueConstraint("organization_id", "slug"),
        CheckConstraint(r"slug ~ '^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$'", name="slug_format"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUIDT, ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(63), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = created_at_col()


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        CheckConstraint(enum_check("role", _values(Role)), name="role"),
        CheckConstraint(enum_check("clearance", _values(Classification)), name="clearance"),
        CheckConstraint(enum_check("status", _values(UserStatus)), name="status"),
        CheckConstraint(
            "(role = 'platform_admin') = (organization_id IS NULL)",
            name="platform_admin_has_no_org",
        ),
        CheckConstraint("email = lower(email)", name="email_lowercase"),
        Index("ix_users_org_role", "organization_id", "role"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUIDT, ForeignKey("organizations.id", ondelete="RESTRICT")
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    clearance: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=UserStatus.ACTIVE)
    failed_login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lockout_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(TS)
    last_login_at: Mapped[datetime | None] = mapped_column(TS)
    password_changed_at: Mapped[datetime | None] = mapped_column(TS)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mfa_secret_enc: Mapped[bytes | None] = mapped_column(LargeBinary)
    mfa_last_used_step: Mapped[int | None] = mapped_column(BigInteger)
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()


class UserDepartment(Base):
    __tablename__ = "user_departments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "user_id"],
            ["users.organization_id", "users.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["organization_id", "department_id"],
            ["departments.organization_id", "departments.id"],
            ondelete="CASCADE",
        ),
        Index("ix_user_departments_department", "organization_id", "department_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(UUIDT, primary_key=True)
    department_id: Mapped[uuid.UUID] = mapped_column(UUIDT, primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    is_manager: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = created_at_col()


class AuthSession(Base):
    """A login session. Access JWTs carry its id (``sid``) and die with it."""

    __tablename__ = "auth_sessions"
    __table_args__ = (Index("ix_auth_sessions_user", "user_id", "revoked_at"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDT, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    created_at: Mapped[datetime] = created_at_col()
    expires_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(TS)
    revoked_at: Mapped[datetime | None] = mapped_column(TS)
    revoke_reason: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(256))
    ip_prefix: Mapped[str | None] = mapped_column(String(64))
    mfa_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RefreshToken(Base):
    """Single-use refresh token (stored as a peppered HMAC, never in clear)."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUIDT, ForeignKey("auth_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), unique=True, nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    expires_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(TS)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDT, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), unique=True, nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    expires_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(TS)


class MfaChallenge(Base):
    """Second step of an MFA login: bound to the user, short-lived, attempt-limited."""

    __tablename__ = "mfa_challenges"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDT, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), unique=True, nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    expires_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    used_at: Mapped[datetime | None] = mapped_column(TS)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "department_id"],
            ["departments.organization_id", "departments.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["organization_id", "owner_id"],
            ["users.organization_id", "users.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            enum_check("classification", _values(Classification)), name="classification"
        ),
        CheckConstraint(enum_check("status", _values(DocumentStatus)), name="status"),
        CheckConstraint(enum_check("doc_type", _values(DocumentType)), name="doc_type"),
        CheckConstraint(
            "suggested_classification IS NULL OR "
            + enum_check("suggested_classification", _values(Classification)),
            name="suggested_classification",
        ),
        CheckConstraint("cardinality(tags) <= 20", name="tag_limit"),
        Index("ix_documents_org_status", "organization_id", "status", "classification"),
        Index("ix_documents_org_department", "organization_id", "department_id"),
        Index("ix_documents_org_owner", "organization_id", "owner_id"),
        Index("ix_documents_org_created", "organization_id", "created_at"),
        Index("ix_documents_tags", "tags", postgresql_using="gin"),
        ForeignKeyConstraint(
            ["organization_id", "current_version_id"],
            ["document_versions.organization_id", "document_versions.id"],
            name="fk_documents_current_version",
            ondelete="SET NULL (current_version_id)",
            use_alter=True,
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUIDT, ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    classification: Mapped[str] = mapped_column(String(16), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(16), nullable=False, default=DocumentType.OTHER)
    doc_type_source: Mapped[str] = mapped_column(String(16), nullable=False, default="auto")
    doc_type_confidence: Mapped[float | None] = mapped_column(Float)
    allowed_roles: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)), nullable=False, server_default=text("'{}'")
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), nullable=False, server_default=text("'{}'")
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DocumentStatus.PROCESSING
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    version_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    suggested_classification: Mapped[str | None] = mapped_column(String(16))
    sensitivity_signals: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)), nullable=False, server_default=text("'{}'")
    )
    legal_hold: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retention_until: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()
    deleted_at: Mapped[datetime | None] = mapped_column(TS)
    deleted_by: Mapped[uuid.UUID | None] = mapped_column(UUIDT)


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        UniqueConstraint("document_id", "version_number"),
        ForeignKeyConstraint(
            ["organization_id", "document_id"],
            ["documents.organization_id", "documents.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(enum_check("status", _values(VersionStatus)), name="status"),
        CheckConstraint("size_bytes >= 0", name="size_non_negative"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_hex"),
        Index("ix_document_versions_org_sha", "organization_id", "sha256"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    extension: Mapped[str] = mapped_column(String(10), nullable=False)
    declared_mime: Mapped[str | None] = mapped_column(String(127))
    detected_mime: Mapped[str] = mapped_column(String(127), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=VersionStatus.UPLOADED)
    page_count: Mapped[int | None] = mapped_column(Integer)
    char_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(String(16))
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    security_findings: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    semantic_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    needs_ocr: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_detail: Mapped[str | None] = mapped_column(String(500))
    change_note: Mapped[str | None] = mapped_column(String(500))
    created_by: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    processed_at: Mapped[datetime | None] = mapped_column(TS)


class DocumentGrant(Base):
    __tablename__ = "document_grants"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "document_id"],
            ["documents.organization_id", "documents.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["organization_id", "grantee_user_id"],
            ["users.organization_id", "users.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["organization_id", "grantee_department_id"],
            ["departments.organization_id", "departments.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(enum_check("grantee_type", _values(GranteeType)), name="grantee_type"),
        CheckConstraint(enum_check("permission", _values(GrantPermission)), name="permission"),
        CheckConstraint(
            "(grantee_type = 'user' AND grantee_user_id IS NOT NULL "
            "AND grantee_department_id IS NULL AND grantee_role IS NULL) OR "
            "(grantee_type = 'department' AND grantee_department_id IS NOT NULL "
            "AND grantee_user_id IS NULL AND grantee_role IS NULL) OR "
            "(grantee_type = 'role' AND grantee_role IS NOT NULL "
            "AND grantee_user_id IS NULL AND grantee_department_id IS NULL)",
            name="grantee_shape",
        ),
        CheckConstraint(
            "grantee_role IS NULL OR grantee_role <> 'platform_admin'", name="no_platform_grants"
        ),
        Index("ix_document_grants_document", "organization_id", "document_id"),
        Index("ix_document_grants_user", "organization_id", "grantee_user_id"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    grantee_type: Mapped[str] = mapped_column(String(16), nullable=False)
    grantee_user_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    grantee_department_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    grantee_role: Mapped[str | None] = mapped_column(String(32))
    permission: Mapped[str] = mapped_column(
        String(16), nullable=False, default=GrantPermission.READ
    )
    granted_by: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    expires_at: Mapped[datetime | None] = mapped_column(TS)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("version_id", "chunk_index"),
        UniqueConstraint("organization_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "version_id"],
            ["document_versions.organization_id", "document_versions.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["organization_id", "document_id"],
            ["documents.organization_id", "documents.id"],
            ondelete="CASCADE",
        ),
        Index("ix_document_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_document_chunks_doc", "organization_id", "document_id", "version_id"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(300))
    heading_path: Mapped[list[str]] = mapped_column(
        ARRAY(String(300)), nullable=False, server_default=text("'{}'")
    )
    block_types: Mapped[list[str]] = mapped_column(
        ARRAY(String(16)), nullable=False, server_default=text("'{}'")
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    injection_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    injection_flags: Mapped[list[str]] = mapped_column(
        ARRAY(String(48)), nullable=False, server_default=text("'{}'")
    )
    pii_types: Mapped[list[str]] = mapped_column(
        ARRAY(String(24)), nullable=False, server_default=text("'{}'")
    )
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(section, '')), 'A') || "
            "setweight(to_tsvector('english', content), 'B') || "
            "setweight(to_tsvector('simple', content), 'C')",
            persisted=True,
        ),
    )
    created_at: Mapped[datetime] = created_at_col()


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "chunk_id"],
            ["document_chunks.organization_id", "document_chunks.id"],
            ondelete="CASCADE",
        ),
        Index("ix_chunk_embeddings_doc", "organization_id", "document_id"),
        Index(
            "ix_chunk_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(UUIDT, primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = created_at_col()


class ExtractedField(Base):
    __tablename__ = "extracted_fields"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "version_id"],
            ["document_versions.organization_id", "document_versions.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("method IN ('rules', 'llm')", name="method"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        Index("ix_extracted_fields_lookup", "organization_id", "field", "value_date"),
        Index("ix_extracted_fields_version", "organization_id", "version_id"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    field: Mapped[str] = mapped_column(String(64), nullable=False)
    value_text: Mapped[str | None] = mapped_column(String(1000))
    value_date: Mapped[date | None] = mapped_column(Date)
    value_number: Mapped[Decimal | None] = mapped_column(Numeric(20, 4))
    currency: Mapped[str | None] = mapped_column(String(3))
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(8), nullable=False)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    page: Mapped[int | None] = mapped_column(Integer)
    evidence: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = created_at_col()


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "user_id"],
            ["users.organization_id", "users.id"],
            ondelete="CASCADE",
        ),
        Index("ix_conversations_user", "organization_id", "user_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "conversation_id"],
            ["conversations.organization_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(enum_check("role", _values(MessageRole)), name="role"),
        Index("ix_messages_conversation", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    warnings: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = created_at_col()


# --------------------------------------------------------------------------- #
# Jobs, audit, usage, exports
# --------------------------------------------------------------------------- #
class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(enum_check("status", _values(JobStatus)), name="status"),
        CheckConstraint("attempts >= 0 AND max_attempts >= 1", name="attempts"),
        Index(
            "ix_jobs_claimable",
            "priority",
            "run_after",
            postgresql_where=text("status = 'queued'"),
        ),
        Index("ix_jobs_running_lease", "locked_until", postgresql_where=text("status = 'running'")),
        Index("ix_jobs_org_created", "organization_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUIDT, ForeignKey("organizations.id", ondelete="RESTRICT")
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.QUEUED)
    priority: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=100)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    run_after: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=text("now()"))
    locked_by: Mapped[str | None] = mapped_column(String(128))
    locked_until: Mapped[datetime | None] = mapped_column(TS)
    last_error_code: Mapped[str | None] = mapped_column(String(64))
    last_error: Mapped[str | None] = mapped_column(String(500))
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    request_id: Mapped[str | None] = mapped_column(String(64))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()
    started_at: Mapped[datetime | None] = mapped_column(TS)
    finished_at: Mapped[datetime | None] = mapped_column(TS)


class AuditEvent(Base):
    """Append-only, HMAC hash-chained audit record (UPDATE/DELETE blocked by trigger)."""

    __tablename__ = "audit_events"
    __table_args__ = (
        CheckConstraint(enum_check("outcome", _values(AuditOutcome)), name="outcome"),
        Index("ix_audit_events_org_time", "organization_id", "occurred_at"),
        Index("ix_audit_events_unsealed", "id", postgresql_where=text("hash IS NULL")),
        Index("ix_audit_events_actor", "organization_id", "actor_user_id", "occurred_at"),
        Index("ix_audit_events_action", "organization_id", "action", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    occurred_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=text("now()"))
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    actor_role: Mapped[str | None] = mapped_column(String(32))
    actor_ip_prefix: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(32))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    seal_seq: Mapped[int | None] = mapped_column(BigInteger)
    prev_hash: Mapped[bytes | None] = mapped_column(LargeBinary(32))
    hash: Mapped[bytes | None] = mapped_column(LargeBinary(32))
    sealed_at: Mapped[datetime | None] = mapped_column(TS)


class AuditChainHead(Base):
    __tablename__ = "audit_chain_heads"

    chain_key: Mapped[uuid.UUID] = mapped_column(UUIDT, primary_key=True)
    last_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    last_hash: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    anchor_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    anchor_hash: Mapped[bytes | None] = mapped_column(LargeBinary(32))
    updated_at: Mapped[datetime] = updated_at_col()


class LlmUsage(Base):
    __tablename__ = "llm_usage"
    __table_args__ = (Index("ix_llm_usage_org_time", "organization_id", "created_at"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUIDT, ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUIDT)
    request_id: Mapped[str | None] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, default=Decimal(0))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = created_at_col()


class Export(Base):
    __tablename__ = "exports"
    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "user_id"],
            ["users.organization_id", "users.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(enum_check("status", _values(ExportStatus)), name="status"),
        CheckConstraint("format IN ('csv', 'json')", name="format"),
        Index("ix_exports_user", "organization_id", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUIDT, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    format: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=ExportStatus.PENDING)
    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    row_count: Mapped[int | None] = mapped_column(Integer)
    storage_key: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = created_at_col()
    ready_at: Mapped[datetime | None] = mapped_column(TS)
    expires_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    download_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_downloaded_at: Mapped[datetime | None] = mapped_column(TS)


TENANT_TABLES: tuple[str, ...] = (
    "departments",
    "user_departments",
    "documents",
    "document_versions",
    "document_grants",
    "document_chunks",
    "chunk_embeddings",
    "extracted_fields",
    "llm_usage",
    "exports",
)
"""Tables whose rows always belong to exactly one organisation (pure tenant RLS)."""

NULLABLE_TENANT_TABLES: tuple[str, ...] = (
    "users",
    "auth_sessions",
    "refresh_tokens",
    "password_reset_tokens",
    "mfa_challenges",
)
"""Identity tables where ``organization_id IS NULL`` marks a platform-operator row."""

PRIVATE_TABLES: tuple[str, ...] = ("conversations", "messages")
"""Tenant tables additionally restricted to the owning user."""
