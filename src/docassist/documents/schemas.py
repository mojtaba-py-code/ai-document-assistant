"""API schemas of the documents area.

Inputs forbid unknown fields (``extra="forbid"``) so a typo or an injected attribute is an
error instead of being silently ignored. Outputs are plain data transfer objects built by
:class:`~docassist.documents.service.DocumentService`; every string in them has been
sanitised for display.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated, Any, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from docassist.core.enums import (
    Classification,
    DocumentStatus,
    DocumentType,
    GranteeType,
    GrantPermission,
    Role,
    VersionStatus,
)

MAX_TAGS = 20
MAX_TAG_CHARS = 64
NULLABLE_UPDATE_FIELDS = frozenset({"department_id", "retention_until"})
PERMISSION_FIELDS = frozenset({"classification", "department_id", "allowed_roles", "legal_hold"})
"""Update fields that change who may see or manage a document (audited separately)."""

TagList = Annotated[list[Annotated[str, Field(max_length=200)]], Field(max_length=MAX_TAGS)]
RoleList = Annotated[list[Role], Field(max_length=len(Role))]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class _Output(BaseModel):
    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class UploadForm(_Input):
    """Metadata fields of the multipart upload (``tags``/``allowed_roles`` pre-split)."""

    title: str | None = Field(default=None, max_length=300)
    classification: Classification
    department_id: uuid.UUID | None = None
    doc_type: DocumentType | None = None
    tags: TagList = Field(default_factory=list)
    allowed_roles: RoleList = Field(default_factory=list)
    allow_duplicate: bool = False


class VersionForm(_Input):
    """Metadata fields of the multipart new-version upload."""

    change_note: str | None = Field(default=None, max_length=500)
    allow_duplicate: bool = False


class UrlImportRequest(_Input):
    url: str = Field(min_length=8, max_length=2048)
    title: str | None = Field(default=None, max_length=300)
    classification: Classification
    department_id: uuid.UUID | None = None
    doc_type: DocumentType | None = None
    tags: TagList = Field(default_factory=list)
    allowed_roles: RoleList = Field(default_factory=list)
    allow_duplicate: bool = False


class DocumentUpdate(_Input):
    """Partial update. Only fields present in the request body are applied;
    ``department_id`` and ``retention_until`` may be set to ``null`` to clear them."""

    title: str | None = Field(default=None, min_length=1, max_length=300)
    tags: TagList | None = None
    doc_type: DocumentType | None = None
    classification: Classification | None = None
    department_id: uuid.UUID | None = None
    allowed_roles: RoleList | None = None
    retention_until: date | None = None
    legal_hold: bool | None = None

    @model_validator(mode="after")
    def _no_null_for_required(self) -> Self:
        for name in self.model_fields_set - NULLABLE_UPDATE_FIELDS:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class GrantCreate(_Input):
    grantee_type: GranteeType
    user_id: uuid.UUID | None = None
    department_id: uuid.UUID | None = None
    role: Role | None = None
    permission: GrantPermission = GrantPermission.READ
    expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        expected = {
            GranteeType.USER: "user_id",
            GranteeType.DEPARTMENT: "department_id",
            GranteeType.ROLE: "role",
        }[self.grantee_type]
        present = {
            name for name in ("user_id", "department_id", "role") if getattr(self, name) is not None
        }
        if present != {expected}:
            raise ValueError(f"grantee_type {self.grantee_type.value} requires exactly {expected}")
        if self.role is Role.PLATFORM_ADMIN:
            raise ValueError("platform_admin cannot be granted document access")
        return self


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
class FindingInfo(_Output):
    code: str
    severity: str
    detail: str


class UploadResult(_Output):
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    status: DocumentStatus
    version_status: VersionStatus
    job_id: uuid.UUID | None
    detected_mime: str
    size_bytes: int
    sha256: str
    findings: list[str]
    """Finding codes only (details are for document managers)."""
    duplicate_of: uuid.UUID | None = None
    """Set when an identical file was accepted because ``allow_duplicate`` was true."""


class DocumentSummary(_Output):
    id: uuid.UUID
    title: str
    classification: Classification
    doc_type: DocumentType
    doc_type_source: str
    status: DocumentStatus
    department_id: uuid.UUID | None
    owner_id: uuid.UUID
    tags: list[str]
    allowed_roles: list[str]
    version_count: int
    current_version_id: uuid.UUID | None
    suggested_classification: Classification | None
    legal_hold: bool
    created_at: datetime
    updated_at: datetime
    can_read: bool
    can_manage: bool


class DocumentPage(_Output):
    items: list[DocumentSummary]
    next_cursor: str | None


class VersionInfo(_Output):
    id: uuid.UUID
    version_number: int
    status: VersionStatus
    is_current: bool
    original_filename: str
    extension: str
    detected_mime: str
    size_bytes: int
    sha256: str
    page_count: int | None
    chunk_count: int | None
    needs_ocr: bool
    semantic_indexed: bool
    error_code: str | None
    change_note: str | None
    created_by: uuid.UUID
    created_at: datetime
    processed_at: datetime | None
    findings: list[FindingInfo] | None = None
    """Scanner findings - only returned to principals who can manage the document."""


class GrantInfo(_Output):
    id: uuid.UUID
    grantee_type: GranteeType
    grantee_user_id: uuid.UUID | None
    grantee_department_id: uuid.UUID | None
    grantee_role: str | None
    permission: GrantPermission
    granted_by: uuid.UUID
    created_at: datetime
    expires_at: datetime | None
    active: bool


class IngestionInfo(_Output):
    version_number: int
    status: VersionStatus
    error_code: str | None


class DocumentDetail(DocumentSummary):
    retention_until: date | None
    sensitivity_signals: list[str]
    metadata: dict[str, Any]
    """Metadata extracted from the current version (sanitised) - readers only."""
    ingestion: IngestionInfo | None
    versions: list[VersionInfo]
    grants: list[GrantInfo] | None
    """Only returned to principals who can manage the document."""


class ContentChunk(_Output):
    chunk_index: int
    page_start: int | None
    page_end: int | None
    section: str | None
    heading_path: list[str]
    text: str
    injection_flags: list[str]
    suspicious: bool


class ContentPage(_Output):
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    page: int | None
    page_count: int | None
    items: list[ContentChunk]
    next_cursor: str | None
