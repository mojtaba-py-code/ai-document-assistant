"""Document service: uploads, versions, listing, metadata, grants, deletion, download, preview.

Security invariants (see also :mod:`docassist.authz.policy`):

* **Policy in SQL.** Every query that returns documents or chunks embeds the access policy
  in its ``WHERE`` clause (:func:`visible_clause`, ``readable_clause``,
  ``manageable_clause``) - nothing is fetched first and filtered afterwards. Documents a
  caller cannot see raise ``NotFound`` (no existence leak); visible documents the caller may
  not act on raise ``PermissionDenied``, and the denial is audited.
* **RBAC then ABAC.** Each method first checks the role permission (``document:read`` or
  ``document:upload``), then the per-document rule. Managing a document (metadata,
  permissions, versions, deletion) follows the policy's ``can_manage``: organisation admins,
  department managers of the document's department, owners and holders of an active
  ``manage`` grant - within their clearance.
* **Quarantine.** Quarantined documents are visible only to principals who can manage them,
  are never queued for parsing, and their files are downloadable only by managers who pass
  ``acknowledge_risk`` *and* would be entitled to read the content.
* **Transactional outbox.** An upload is spooled (size-capped while streaming), validated,
  scanned, stored encrypted and then registered in one transaction together with its
  ingestion job and audit event; the stored blob is deleted again if that transaction fails.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import itertools
import json
import math
import re
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final, Protocol, cast

from sqlalchemy import ColumnElement, and_, case, delete, func, or_, select, tuple_, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor, AuditLogger
from docassist.authz.permissions import Permission
from docassist.authz.policy import (
    DocumentFacts,
    can_read,
    listable_clause,
    manageable_clause,
    readable_clause,
)
from docassist.authz.principal import Principal
from docassist.cache.ratelimit import RateLimiter
from docassist.core.config import Settings
from docassist.core.context import utcnow
from docassist.core.enums import (
    AuditOutcome,
    Classification,
    DocumentStatus,
    DocumentType,
    GranteeType,
    GrantPermission,
    JobStatus,
    Role,
    VersionStatus,
)
from docassist.core.errors import (
    Conflict,
    NotFound,
    PayloadTooLarge,
    PermissionDenied,
    RejectedContent,
    ServiceUnavailable,
    UnsupportedMediaType,
    ValidationFailed,
)
from docassist.core.ids import uuid7
from docassist.core.logging import get_logger
from docassist.core.text import clean_line_text, sanitize_text
from docassist.db.models import (
    ChunkEmbedding,
    Department,
    Document,
    DocumentChunk,
    DocumentGrant,
    DocumentVersion,
    ExtractedField,
    Job,
    User,
)
from docassist.db.session import Database
from docassist.documents.scanning import (
    REJECT_CODES,
    ActiveContentScanner,
    Finding,
    MalwareScanner,
    NullScanner,
    Severity,
    Verdict,
    decide,
)
from docassist.documents.schemas import (
    MAX_TAG_CHARS,
    MAX_TAGS,
    PERMISSION_FIELDS,
    ContentChunk,
    ContentPage,
    DocumentDetail,
    DocumentPage,
    DocumentSummary,
    DocumentUpdate,
    FindingInfo,
    GrantCreate,
    GrantInfo,
    IngestionInfo,
    UploadResult,
    VersionInfo,
)
from docassist.documents.storage import (
    ObjectStorage,
    StorageError,
    new_object_key,
    storage_context,
)
from docassist.documents.url_import import UrlImporter
from docassist.documents.validation import (
    DEFAULT_MEMORY_SPOOL_BYTES,
    TEXT_EXTENSIONS,
    DetectedType,
    SpooledUpload,
    ZipLimits,
    extension_of,
    sanitize_filename,
    spool_upload,
    validate_content,
)
from docassist.jobs.queue import enqueue
from docassist.observability import metrics
from docassist.security.crypto import DecryptionError
from docassist.security.ssrf import EgressPolicy

log = get_logger(__name__)

INGEST_JOB: Final = "ingest_version"
VECTOR_SYNC_DOCUMENT_JOB: Final = "vector_sync_document"
VECTOR_DELETE_DOCUMENT_JOB: Final = "vector_delete_document"
VECTOR_PAYLOAD_FIELDS: Final = frozenset(
    {"classification", "department_id", "allowed_roles", "doc_type", "tags", "status", "owner_id"}
)
"""Document fields copied into external vector-store payloads (a change triggers a sync)."""
DEFAULT_PAGE_SIZE: Final = 25
MAX_PAGE_SIZE: Final = 100
DEFAULT_CONTENT_PAGE_SIZE: Final = 20
TENANT_ROLES: Final = frozenset(r.value for r in Role if r is not Role.PLATFORM_ADMIN)
"""Roles that may appear in ``allowed_roles`` and role grants."""

_MAX_QUERY_CHARS = 200
_MAX_CURSOR_CHARS = 256
_MAX_TAG_INPUTS = 200
_UPLOAD_CHUNK = 256 * 1024
_EARLIEST_RETENTION = date(2000, 1, 1)
_LATEST_RETENTION = date(2200, 12, 31)
_METADATA_KEY = re.compile(r"[a-z][a-z0-9_]{0,39}")
_REJECTION_MESSAGES: Final = {
    "pdf_encrypted": (
        "Encrypted or password-protected PDFs cannot be scanned. "
        "Remove the protection and upload the file again."
    ),
}


class DocumentServiceDeps(Protocol):
    """What the service needs from the composition root (the ``Container``)."""

    settings: Settings
    db: Database
    audit: AuditLogger
    limiter: RateLimiter
    storage: ObjectStorage
    egress: EgressPolicy
    overrides: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DownloadPayload:
    filename: str
    content_type: str
    size: int
    version_number: int
    quarantined: bool
    chunks: Iterator[bytes]
    """Decrypted plaintext; the first segment has already been authenticated."""


@dataclass(frozen=True, slots=True)
class _Loaded:
    document: Document
    can_read: bool
    can_manage: bool


@dataclass(frozen=True, slots=True)
class _Metadata:
    title: str | None
    classification: Classification
    department_id: uuid.UUID | None
    doc_type: DocumentType | None
    tags: list[str]
    allowed_roles: list[str]


@dataclass(frozen=True, slots=True)
class _Inspection:
    detected: DetectedType
    findings: list[Finding]
    verdict: Verdict
    signature: str | None


@dataclass(frozen=True, slots=True)
class _Source:
    kind: str
    host: str | None = None


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def visible_clause(principal: Principal, now: datetime) -> ColumnElement[bool]:
    """Documents the principal may know about: listable, and - when quarantined - manageable."""
    return and_(
        listable_clause(principal, now),
        or_(
            Document.status != DocumentStatus.QUARANTINED.value,
            manageable_clause(principal, now),
        ),
    )


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Sanitised, lower-cased, de-duplicated tags (order kept); at most 20 of 64 chars."""
    raw = list(itertools.islice(tags, _MAX_TAG_INPUTS + 1))
    if len(raw) > _MAX_TAG_INPUTS:
        raise ValidationFailed(f"A document can have at most {MAX_TAGS} tags.")
    out: list[str] = []
    for value in raw:
        tag = clean_line_text(str(value), MAX_TAG_CHARS).lower().strip()
        if tag and tag not in out:
            out.append(tag)
    if len(out) > MAX_TAGS:
        raise ValidationFailed(f"A document can have at most {MAX_TAGS} tags.")
    return out


def normalize_roles(roles: Iterable[str | Role]) -> list[str]:
    """Validated ``allowed_roles``: organisation roles only (never ``platform_admin``)."""
    values = sorted({(r.value if isinstance(r, Role) else str(r)).strip().lower() for r in roles})
    if any(v not in TENANT_ROLES for v in values):
        raise ValidationFailed("allowed_roles may only contain organisation roles.")
    return values


def _clean_optional(value: str | None, limit: int) -> str | None:
    """``clean_line_text`` for optional fields; empty results become ``None``."""
    if not value:
        return None
    return clean_line_text(value, limit) or None


def _like_pattern(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> dict[str, Any]:
    invalid = ValidationFailed("The pagination cursor is invalid.")
    if not cursor or len(cursor) > _MAX_CURSOR_CHARS:
        raise invalid
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise invalid from exc
    if not isinstance(data, dict):
        raise invalid
    return data


def _list_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    data = _decode_cursor(cursor)
    try:
        created = datetime.fromisoformat(str(data["t"]))
        ident = uuid.UUID(str(data["i"]))
    except (KeyError, ValueError) as exc:
        raise ValidationFailed("The pagination cursor is invalid.") from exc
    if created.tzinfo is None:
        raise ValidationFailed("The pagination cursor is invalid.")
    return created, ident


def _content_cursor(cursor: str) -> int:
    value = _decode_cursor(cursor).get("c")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationFailed("The pagination cursor is invalid.")
    return value


def _sanitize_metadata(raw: Any) -> dict[str, Any]:
    """Only simple, sanitised values under simple keys survive (it came from the file)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in itertools.islice(raw.items(), 40):
        if not isinstance(key, str) or not _METADATA_KEY.fullmatch(key):
            continue
        if isinstance(value, bool | int) or (isinstance(value, float) and math.isfinite(value)):
            out[key] = value
        elif isinstance(value, str):
            cleaned = clean_line_text(value, 300)
            if cleaned:
                out[key] = cleaned
        elif isinstance(value, list):
            items = [clean_line_text(v, 120) for v in value[:50] if isinstance(v, str)]
            out[key] = [item for item in items if item]
    return out


def _findings_out(raw: Any) -> list[FindingInfo]:
    if not isinstance(raw, list):
        return []
    return [
        FindingInfo(
            code=clean_line_text(str(item.get("code", "")), 64),
            severity=clean_line_text(str(item.get("severity", "")), 16),
            detail=clean_line_text(str(item.get("detail", "")), 300),
        )
        for item in raw[:100]
        if isinstance(item, dict)
    ]


def _grant_out(grant: DocumentGrant, now: datetime) -> GrantInfo:
    return GrantInfo(
        id=grant.id,
        grantee_type=GranteeType(grant.grantee_type),
        grantee_user_id=grant.grantee_user_id,
        grantee_department_id=grant.grantee_department_id,
        grantee_role=grant.grantee_role,
        permission=GrantPermission(grant.permission),
        granted_by=grant.granted_by,
        created_at=grant.created_at,
        expires_at=grant.expires_at,
        active=grant.expires_at is None or grant.expires_at > now,
    )


def _summary_fields(doc: Document, can_read_doc: bool, can_manage_doc: bool) -> dict[str, Any]:
    return {
        "id": doc.id,
        "title": clean_line_text(doc.title, 300),
        "classification": Classification(doc.classification),
        "doc_type": DocumentType(doc.doc_type),
        "doc_type_source": doc.doc_type_source,
        "status": DocumentStatus(doc.status),
        "department_id": doc.department_id,
        "owner_id": doc.owner_id,
        "tags": list(doc.tags or ()),
        "allowed_roles": list(doc.allowed_roles or ()),
        "version_count": doc.version_count,
        "current_version_id": doc.current_version_id,
        "suggested_classification": (
            Classification(doc.suggested_classification) if doc.suggested_classification else None
        ),
        "legal_hold": doc.legal_hold,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "can_read": can_read_doc,
        "can_manage": can_manage_doc,
    }


def _title_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return clean_line_text(stem.replace("_", " "), 300) or "Untitled document"


def _permission_snapshot(doc: Document) -> dict[str, Any]:
    return {
        "classification": doc.classification,
        "department_id": str(doc.department_id) if doc.department_id else None,
        "allowed_roles": sorted(doc.allowed_roles or ()),
        "legal_hold": doc.legal_hold,
    }


async def _chunked(data: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(data), _UPLOAD_CHUNK):
        yield data[start : start + _UPLOAD_CHUNK]


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class DocumentService:
    def __init__(
        self,
        deps: DocumentServiceDeps,
        *,
        settings: Settings | None = None,
        malware_scanner: MalwareScanner | None = None,
        active_scanner: ActiveContentScanner | None = None,
        url_importer: UrlImporter | None = None,
        memory_spool_bytes: int = DEFAULT_MEMORY_SPOOL_BYTES,
    ) -> None:
        self._settings = settings or deps.settings
        self._db = deps.db
        self._audit = deps.audit
        self._limiter = deps.limiter
        self._storage = deps.storage
        self._malware: MalwareScanner = malware_scanner or NullScanner()
        self._active = active_scanner or ActiveContentScanner()
        self._importer = url_importer or UrlImporter(
            self._settings.upload,
            deps.egress,
            transport=deps.overrides.get("url_import_transport"),
        )
        self._memory_spool_bytes = memory_spool_bytes

    @property
    def malware_scanner(self) -> MalwareScanner:
        return self._malware

    # ================================================================ uploads
    async def upload(
        self,
        principal: Principal,
        *,
        stream: AsyncIterator[bytes],
        filename: str,
        declared_mime: str | None,
        title: str | None = None,
        classification: Classification | str,
        department_id: uuid.UUID | None = None,
        doc_type: DocumentType | str | None = None,
        tags: Iterable[str] = (),
        allowed_roles: Iterable[str | Role] = (),
        allow_duplicate: bool = False,
    ) -> UploadResult:
        """Create a document (version 1) from an uploaded file.

        Requires ``document:upload``; rate limited per user. Returns the new ids, the status
        (``processing``, or ``quarantined`` when scanners flagged the file), the queued
        ingestion job and the finding codes.
        """
        org_id = self._require(principal, Permission.DOCUMENT_UPLOAD)
        await self._limiter.enforce(
            "upload", str(principal.user_id), self._settings.rate_limit.upload_per_user
        )
        meta = await self._metadata(
            principal,
            title=title,
            classification=classification,
            department_id=department_id,
            doc_type=doc_type,
            tags=tags,
            allowed_roles=allowed_roles,
        )
        return await self._create(
            principal,
            org_id,
            meta=meta,
            stream=stream,
            filename=filename,
            declared_mime=declared_mime,
            allow_duplicate=allow_duplicate,
            source=_Source("upload"),
        )

    async def import_url(
        self,
        principal: Principal,
        url: str,
        *,
        title: str | None = None,
        classification: Classification | str,
        department_id: uuid.UUID | None = None,
        doc_type: DocumentType | str | None = None,
        tags: Iterable[str] = (),
        allowed_roles: Iterable[str | Role] = (),
        allow_duplicate: bool = False,
    ) -> UploadResult:
        """Fetch a document from an allowlisted URL, then treat it exactly like an upload.

        ``FeatureDisabled`` unless ``upload.url_import_enabled``; the URL is checked against
        the import allowlist *and* the egress policy before any network access.
        """
        org_id = self._require(principal, Permission.DOCUMENT_UPLOAD)
        try:
            host = self._importer.check(url)
        except ValidationFailed as exc:
            await self._audit.record_detached(
                Actor.of(principal),
                "document.import_refused",
                outcome=AuditOutcome.DENIED,
                details={"reason": exc.internal_detail or exc.code},
            )
            raise
        await self._limiter.enforce(
            "upload", str(principal.user_id), self._settings.rate_limit.upload_per_user
        )
        meta = await self._metadata(
            principal,
            title=title,
            classification=classification,
            department_id=department_id,
            doc_type=doc_type,
            tags=tags,
            allowed_roles=allowed_roles,
        )
        try:
            fetched = await self._importer.fetch(url)
        except (ValidationFailed, PayloadTooLarge) as exc:
            await self._audit.record_detached(
                Actor.of(principal),
                "document.import_refused",
                outcome=AuditOutcome.DENIED,
                details={"host": host, "reason": exc.internal_detail or exc.code},
            )
            raise
        return await self._create(
            principal,
            org_id,
            meta=meta,
            stream=_chunked(fetched.content),
            filename=fetched.filename,
            declared_mime=fetched.content_type,
            allow_duplicate=allow_duplicate,
            source=_Source("url", fetched.host),
        )

    async def add_version(
        self,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        stream: AsyncIterator[bytes],
        filename: str,
        declared_mime: str | None = None,
        change_note: str | None = None,
        allow_duplicate: bool = False,
    ) -> UploadResult:
        """Upload a new version of a document the principal can manage.

        The document keeps serving its current (indexed) version until the new one has been
        indexed; the version number is allocated under a row lock on the document.
        """
        org_id = self._require(principal, Permission.DOCUMENT_UPLOAD)
        await self._limiter.enforce(
            "upload", str(principal.user_id), self._settings.rate_limit.upload_per_user
        )
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, utcnow())
        if not loaded.can_manage:
            raise await self._deny(principal, document_id, "add_version")
        note = _clean_optional(change_note, 500)
        safe_name, extension = await self._filename(principal, filename)
        with await self._spool(principal, stream, extension) as spool:
            inspection = await self._inspect(principal, spool, extension)
            duplicate_of = await self._duplicate_check(principal, spool.sha256, allow_duplicate)
            version_id = uuid7()
            key = new_object_key(org_id)
            await self._storage.put_stream(
                key, spool.aiter_chunks(), storage_context(org_id, "version", version_id)
            )
            try:
                return await self._insert_version(
                    principal,
                    org_id,
                    document_id,
                    version_id=version_id,
                    key=key,
                    spool=spool,
                    inspection=inspection,
                    safe_name=safe_name,
                    extension=extension,
                    declared_mime=declared_mime,
                    change_note=note,
                    duplicate_of=duplicate_of,
                )
            except BaseException:
                await self._discard_blob(key)
                raise

    # ================================================================== reads
    async def list_documents(
        self,
        principal: Principal,
        *,
        q: str | None = None,
        doc_type: DocumentType | None = None,
        classification: Classification | None = None,
        department_id: uuid.UUID | None = None,
        status: DocumentStatus | None = None,
        tag: str | None = None,
        owner: str | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> DocumentPage:
        """Documents the principal may know about, newest first (keyset pagination)."""
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValidationFailed(f"limit must be between 1 and {MAX_PAGE_SIZE}.")
        if status is DocumentStatus.DELETED:
            raise ValidationFailed("Deleted documents cannot be listed.")
        if owner is not None and owner != "me":
            raise ValidationFailed("owner only supports the value 'me'.")
        now = utcnow()
        can_read_col = case((readable_clause(principal, now), True), else_=False)
        can_manage_col = case((manageable_clause(principal, now), True), else_=False)
        stmt = select(
            Document, can_read_col.label("can_read"), can_manage_col.label("can_manage")
        ).where(Document.organization_id == org_id, visible_clause(principal, now))
        if q:
            text = clean_line_text(q, _MAX_QUERY_CHARS)
            if text:
                stmt = stmt.where(Document.title.ilike(_like_pattern(text), escape="\\"))
        if doc_type is not None:
            stmt = stmt.where(Document.doc_type == DocumentType(doc_type).value)
        if classification is not None:
            stmt = stmt.where(Document.classification == Classification(classification).value)
        if department_id is not None:
            stmt = stmt.where(Document.department_id == department_id)
        if status is not None:
            stmt = stmt.where(Document.status == DocumentStatus(status).value)
        if tag:
            wanted = clean_line_text(tag, MAX_TAG_CHARS).lower()
            stmt = stmt.where(Document.tags.contains([wanted]))
        if owner == "me":
            stmt = stmt.where(Document.owner_id == principal.user_id)
        if cursor:
            created, ident = _list_cursor(cursor)
            stmt = stmt.where(tuple_(Document.created_at, Document.id) < tuple_(created, ident))
        stmt = stmt.order_by(Document.created_at.desc(), Document.id.desc()).limit(limit + 1)
        async with self._db.session(principal.db_context) as session:
            rows = (await session.execute(stmt)).all()
        items = [
            DocumentSummary(**_summary_fields(doc, bool(r), bool(m))) for doc, r, m in rows[:limit]
        ]
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1][0]
            next_cursor = _encode_cursor({"t": last.created_at.isoformat(), "i": str(last.id)})
        return DocumentPage(items=items, next_cursor=next_cursor)

    async def get(self, principal: Principal, document_id: uuid.UUID) -> DocumentDetail:
        """Metadata, versions, ingestion state; grants and findings for managers only;
        extracted file metadata for readers only."""
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now)
            return await self._detail(session, loaded, now)

    async def content(
        self,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        version_number: int | None = None,
        page: int | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_CONTENT_PAGE_SIZE,
    ) -> ContentPage:
        """Indexed text of a version (preview pane), chunk by chunk; requires read access."""
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValidationFailed(f"limit must be between 1 and {MAX_PAGE_SIZE}.")
        if page is not None and page < 1:
            raise ValidationFailed("page must be positive.")
        after = _content_cursor(cursor) if cursor else None
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now)
            if not loaded.can_read:
                raise await self._deny(principal, document_id, "view")
            doc = loaded.document
            version = await self._pick_version(session, doc, version_number)
            if version.status != VersionStatus.INDEXED.value:
                raise Conflict(
                    "This version has not been indexed yet.",
                    extra={"reason": "version_not_indexed"},
                )
            chunk = DocumentChunk
            stmt = (
                select(chunk)
                .join(
                    Document,
                    and_(
                        Document.id == chunk.document_id,
                        Document.organization_id == chunk.organization_id,
                    ),
                )
                .where(
                    readable_clause(principal, now),
                    chunk.organization_id == org_id,
                    chunk.document_id == doc.id,
                    chunk.version_id == version.id,
                )
            )
            if page is not None:
                stmt = stmt.where(
                    chunk.page_start <= page,
                    func.coalesce(chunk.page_end, chunk.page_start) >= page,
                )
            if after is not None:
                stmt = stmt.where(chunk.chunk_index > after)
            stmt = stmt.order_by(chunk.chunk_index).limit(limit + 1)
            rows = list((await session.execute(stmt)).scalars().all())
            self._audit.record(
                session,
                Actor.of(principal),
                "document.view",
                resource_type="document",
                resource_id=doc.id,
                details={"version_number": version.version_number, "page": page},
            )
            await session.commit()
        warn = self._settings.retrieval.injection_warn_threshold
        items = [
            ContentChunk(
                chunk_index=c.chunk_index,
                page_start=c.page_start,
                page_end=c.page_end,
                section=_clean_optional(c.section, 300),
                heading_path=[clean_line_text(h, 300) for h in (c.heading_path or ())],
                text=sanitize_text(c.content)[0],
                injection_flags=list(c.injection_flags or ()),
                suspicious=c.injection_score >= warn,
            )
            for c in rows[:limit]
        ]
        next_cursor = (
            _encode_cursor({"c": rows[limit - 1].chunk_index}) if len(rows) > limit else None
        )
        return ContentPage(
            document_id=doc.id,
            version_id=version.id,
            version_number=version.version_number,
            page=page,
            page_count=version.page_count,
            items=items,
            next_cursor=next_cursor,
        )

    async def download(
        self,
        principal: Principal,
        document_id: uuid.UUID,
        *,
        version_number: int | None = None,
        acknowledge_risk: bool = False,
    ) -> DownloadPayload:
        """Decrypted original file of a version (default: the current one, else the latest).

        Requires read access. Quarantined versions additionally require the ability to
        manage the document, ``acknowledge_risk=True`` and content entitlement.
        """
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now)
            doc = loaded.document
            version = await self._pick_version(session, doc, version_number)
            quarantined = version.status == VersionStatus.QUARANTINED.value
            if quarantined:
                await self._authorize_quarantined_download(
                    session, principal, loaded, now, acknowledge_risk=acknowledge_risk
                )
            elif not loaded.can_read:
                raise await self._deny(principal, document_id, "download")
            chunks = await self._open_blob(org_id, version)
            self._audit.record(
                session,
                Actor.of(principal),
                "document.download",
                resource_type="document",
                resource_id=doc.id,
                details={
                    "version_number": version.version_number,
                    "size_bytes": version.size_bytes,
                    "quarantined": quarantined,
                    "acknowledged_risk": acknowledge_risk if quarantined else False,
                },
            )
            await session.commit()
        if quarantined:
            metrics.SECURITY_EVENTS.labels(kind="quarantined_download").inc()
        first, rest = chunks
        return DownloadPayload(
            filename=sanitize_filename(version.original_filename),
            content_type=self._served_type(version, first),
            size=version.size_bytes,
            version_number=version.version_number,
            quarantined=quarantined,
            chunks=itertools.chain([first], rest),
        )

    # ============================================================ management
    async def update(
        self, principal: Principal, document_id: uuid.UUID, patch: DocumentUpdate
    ) -> DocumentDetail:
        """Apply a partial update (see :class:`DocumentUpdate`); requires manage rights.

        Classification may not exceed the actor's clearance (organisation admins excepted);
        moving to a department requires managing that department (or being an organisation
        admin); ``legal_hold`` is for organisation admins only. Changes to who can see the
        document are audited as ``document.permissions_changed`` with before/after values.
        """
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        fields = patch.model_fields_set
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now, lock=True)
            if not loaded.can_manage:
                raise await self._deny(principal, document_id, "update")
            doc = loaded.document
            before = _permission_snapshot(doc)
            changed = await self._apply_patch(
                session, principal, org_id=org_id, doc=doc, patch=patch, fields=fields
            )
            if not changed:
                return await self._detail(session, loaded, now)
            await session.flush()
            after = _permission_snapshot(doc)
            actor = Actor.of(principal)
            permission_changes = [f for f in changed if f in PERMISSION_FIELDS]
            other_changes = [f for f in changed if f not in PERMISSION_FIELDS]
            if permission_changes:
                self._audit.record(
                    session,
                    actor,
                    "document.permissions_changed",
                    resource_type="document",
                    resource_id=doc.id,
                    details={
                        "fields": permission_changes,
                        "before": {k: before[k] for k in permission_changes},
                        "after": {k: after[k] for k in permission_changes},
                    },
                )
            if other_changes:
                self._audit.record(
                    session,
                    actor,
                    "document.updated",
                    resource_type="document",
                    resource_id=doc.id,
                    details={"fields": other_changes},
                )
            if VECTOR_PAYLOAD_FIELDS.intersection(changed):
                await self._vector_job(session, VECTOR_SYNC_DOCUMENT_JOB, org_id, doc.id, principal)
            try:
                refreshed = await self._load(session, principal, org_id, document_id, now)
            except NotFound:  # the actor just moved the document out of their own reach
                refreshed = _Loaded(doc, can_read=False, can_manage=False)
            detail = await self._detail(session, refreshed, now)
            await session.commit()
        return detail

    async def delete(self, principal: Principal, document_id: uuid.UUID) -> None:
        """Soft-delete a document and, in the same transaction, hard-delete its chunks,
        embeddings and extracted fields (it vanishes from retrieval immediately) and cancel
        its queued ingestion jobs. Refused while the document is under legal hold. Stored
        files are purged later by the retention worker."""
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now, lock=True)
            if not loaded.can_manage:
                raise await self._deny(principal, document_id, "delete")
            doc = loaded.document
            if doc.legal_hold:
                await self._audit.record_detached(
                    Actor.of(principal),
                    "document.delete_refused",
                    outcome=AuditOutcome.DENIED,
                    resource_type="document",
                    resource_id=document_id,
                    details={"reason": "legal_hold"},
                )
                raise Conflict(
                    "This document is under legal hold and cannot be deleted.",
                    extra={"reason": "legal_hold"},
                )
            version_ids = [
                str(v)
                for v in (
                    await session.execute(
                        select(DocumentVersion.id).where(
                            DocumentVersion.organization_id == org_id,
                            DocumentVersion.document_id == doc.id,
                        )
                    )
                )
                .scalars()
                .all()
            ]
            embeddings = await self._delete_rows(
                session,
                delete(ChunkEmbedding).where(
                    ChunkEmbedding.organization_id == org_id,
                    ChunkEmbedding.document_id == doc.id,
                ),
            )
            chunks = await self._delete_rows(
                session,
                delete(DocumentChunk).where(
                    DocumentChunk.organization_id == org_id, DocumentChunk.document_id == doc.id
                ),
            )
            fields = await self._delete_rows(
                session,
                delete(ExtractedField).where(
                    ExtractedField.organization_id == org_id, ExtractedField.document_id == doc.id
                ),
            )
            cancelled = 0
            if version_ids:
                cancelled = await self._delete_rows(
                    session,
                    update(Job)
                    .where(
                        Job.organization_id == org_id,
                        Job.kind == INGEST_JOB,
                        Job.status == JobStatus.QUEUED.value,
                        Job.payload.op("->>")("version_id").in_(version_ids),
                    )
                    .values(
                        status=JobStatus.CANCELLED.value,
                        finished_at=func.now(),
                        updated_at=func.now(),
                        last_error_code="document_deleted",
                    ),
                )
            doc.status = DocumentStatus.DELETED.value
            doc.deleted_at = now
            doc.deleted_by = principal.user_id
            self._audit.record(
                session,
                Actor.of(principal),
                "document.deleted",
                resource_type="document",
                resource_id=doc.id,
                details={
                    "versions": len(version_ids),
                    "chunks_deleted": chunks,
                    "embeddings_deleted": embeddings,
                    "fields_deleted": fields,
                    "jobs_cancelled": cancelled,
                },
            )
            # status is part of the payloads (sync) and the points must go (delete)
            await self._vector_job(session, VECTOR_SYNC_DOCUMENT_JOB, org_id, doc.id, principal)
            await self._vector_job(session, VECTOR_DELETE_DOCUMENT_JOB, org_id, doc.id, principal)
            await session.commit()

    # ================================================================ grants
    async def list_grants(self, principal: Principal, document_id: uuid.UUID) -> list[GrantInfo]:
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now)
            if not loaded.can_manage:
                raise await self._deny(principal, document_id, "list_grants")
            grants = await self._grants(session, org_id, document_id)
        return [_grant_out(g, now) for g in grants]

    async def add_grant(
        self, principal: Principal, document_id: uuid.UUID, grant: GrantCreate
    ) -> GrantInfo:
        """Grant a user, department or role read/manage access (manage rights required).

        The grantee must belong to the same organisation; ``platform_admin`` can never be a
        grantee; ``expires_at`` must lie in the future; an identical active grant is a
        ``Conflict``. Audited as ``document.grant_added``.
        """
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        now = utcnow()
        if grant.expires_at is not None and grant.expires_at <= now:
            raise ValidationFailed("expires_at must be in the future.")
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now, lock=True)
            if not loaded.can_manage:
                raise await self._deny(principal, document_id, "add_grant")
            await self._check_grantee(session, org_id, grant)
            role = grant.role.value if grant.role is not None else None
            existing = (
                await session.execute(
                    select(DocumentGrant.id).where(
                        DocumentGrant.organization_id == org_id,
                        DocumentGrant.document_id == document_id,
                        DocumentGrant.grantee_type == grant.grantee_type.value,
                        DocumentGrant.permission == grant.permission.value,
                        DocumentGrant.grantee_user_id.is_not_distinct_from(grant.user_id),
                        DocumentGrant.grantee_department_id.is_not_distinct_from(
                            grant.department_id
                        ),
                        DocumentGrant.grantee_role.is_not_distinct_from(role),
                        or_(DocumentGrant.expires_at.is_(None), DocumentGrant.expires_at > now),
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                raise Conflict(
                    "An identical active grant already exists.",
                    extra={"grant_id": str(existing)},
                )
            row = DocumentGrant(
                id=uuid7(),
                organization_id=org_id,
                document_id=document_id,
                grantee_type=grant.grantee_type.value,
                grantee_user_id=grant.user_id,
                grantee_department_id=grant.department_id,
                grantee_role=role,
                permission=grant.permission.value,
                granted_by=principal.user_id,
                expires_at=grant.expires_at,
            )
            session.add(row)
            await session.flush()
            self._audit.record(
                session,
                Actor.of(principal),
                "document.grant_added",
                resource_type="document",
                resource_id=document_id,
                details={
                    "grant_id": str(row.id),
                    "grantee_type": row.grantee_type,
                    "grantee": str(grant.user_id or grant.department_id or role),
                    "permission": row.permission,
                    "expires_at": row.expires_at.isoformat() if row.expires_at else None,
                    "self_grant": grant.user_id == principal.user_id,
                },
            )
            await self._vector_job(
                session, VECTOR_SYNC_DOCUMENT_JOB, org_id, document_id, principal
            )
            await session.commit()
            await session.refresh(row)
        return _grant_out(row, now)

    async def revoke_grant(
        self, principal: Principal, document_id: uuid.UUID, grant_id: uuid.UUID
    ) -> None:
        org_id = self._require(principal, Permission.DOCUMENT_READ)
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            loaded = await self._load(session, principal, org_id, document_id, now, lock=True)
            if not loaded.can_manage:
                raise await self._deny(principal, document_id, "revoke_grant")
            grant = (
                await session.execute(
                    select(DocumentGrant).where(
                        DocumentGrant.id == grant_id,
                        DocumentGrant.organization_id == org_id,
                        DocumentGrant.document_id == document_id,
                    )
                )
            ).scalar_one_or_none()
            if grant is None:
                raise NotFound("Grant not found.")
            details = {
                "grant_id": str(grant.id),
                "grantee_type": grant.grantee_type,
                "grantee": str(
                    grant.grantee_user_id or grant.grantee_department_id or grant.grantee_role
                ),
                "permission": grant.permission,
            }
            await session.delete(grant)
            self._audit.record(
                session,
                Actor.of(principal),
                "document.grant_revoked",
                resource_type="document",
                resource_id=document_id,
                details=details,
            )
            await self._vector_job(
                session, VECTOR_SYNC_DOCUMENT_JOB, org_id, document_id, principal
            )
            await session.commit()

    # ========================================================= upload pipeline
    async def _create(
        self,
        principal: Principal,
        org_id: uuid.UUID,
        *,
        meta: _Metadata,
        stream: AsyncIterator[bytes],
        filename: str,
        declared_mime: str | None,
        allow_duplicate: bool,
        source: _Source,
    ) -> UploadResult:
        safe_name, extension = await self._filename(principal, filename)
        with await self._spool(principal, stream, extension) as spool:
            inspection = await self._inspect(principal, spool, extension)
            duplicate_of = await self._duplicate_check(principal, spool.sha256, allow_duplicate)
            document_id, version_id = uuid7(), uuid7()
            key = new_object_key(org_id)
            await self._storage.put_stream(
                key, spool.aiter_chunks(), storage_context(org_id, "version", version_id)
            )
            try:
                return await self._insert_document(
                    principal,
                    org_id,
                    meta=meta,
                    document_id=document_id,
                    version_id=version_id,
                    key=key,
                    spool=spool,
                    inspection=inspection,
                    safe_name=safe_name,
                    extension=extension,
                    declared_mime=declared_mime,
                    duplicate_of=duplicate_of,
                    source=source,
                )
            except BaseException:
                await self._discard_blob(key)
                raise

    async def _filename(self, principal: Principal, filename: str) -> tuple[str, str]:
        safe_name = sanitize_filename(filename)
        extension = extension_of(safe_name)
        if extension not in self._settings.upload.allowed_extensions:
            await self._record_rejection(principal, "extension_not_allowed", extension)
            raise UnsupportedMediaType(
                "This file type is not supported.",
                internal_detail=f"extension {extension!r} not allowed",
                extra={"reason": "extension_not_allowed"},
            )
        return safe_name, extension

    async def _spool(
        self, principal: Principal, stream: AsyncIterator[bytes], extension: str
    ) -> SpooledUpload:
        try:
            return await spool_upload(
                stream,
                max_bytes=self._settings.upload.max_upload_bytes,
                temp_dir=self._settings.storage.temp_dir,
                memory_limit=self._memory_spool_bytes,
            )
        except PayloadTooLarge:
            await self._record_rejection(principal, "too_large", extension)
            raise

    async def _inspect(
        self, principal: Principal, spool: SpooledUpload, extension: str
    ) -> _Inspection:
        upload = self._settings.upload
        limits = ZipLimits(
            max_entries=upload.max_zip_entries,
            max_uncompressed_bytes=upload.max_zip_uncompressed_bytes,
            max_ratio=upload.max_zip_ratio,
        )
        try:
            validated = await asyncio.to_thread(validate_content, spool, extension, limits)
        except (UnsupportedMediaType, RejectedContent) as exc:
            await self._record_rejection(
                principal, str(exc.extra.get("reason", exc.code)), extension
            )
            raise
        detected = validated.detected
        findings = await asyncio.to_thread(
            self._active.scan,
            spool,
            detected.family,
            extension=detected.extension,
            text_encoding=detected.text_encoding,
        )
        try:
            malware = await self._malware.scan(spool)
        except ServiceUnavailable:
            await self._audit.record_detached(
                Actor.of(principal),
                "document.upload_failed",
                outcome=AuditOutcome.FAILURE,
                details={"reason": "malware_scanner_unavailable", "extension": extension},
            )
            raise
        if not malware.clean:
            findings.append(
                Finding(
                    "malware_detected",
                    Severity.CRITICAL,
                    f"Malware scanner reported {malware.signature or 'a threat'}",
                )
            )
        elif not malware.scanned:
            findings.append(
                Finding(
                    "malware_scan_skipped",
                    Severity.INFO,
                    "No malware scanner is configured; only built-in heuristics ran",
                )
            )
        verdict = decide(findings, reject_active_content=upload.reject_active_content)
        if verdict is Verdict.REJECT:
            reason = sorted(f.code for f in findings if f.code in REJECT_CODES)[0]
            await self._record_rejection(principal, reason, extension)
            raise RejectedContent(
                _REJECTION_MESSAGES.get(reason), internal_detail=reason, extra={"reason": reason}
            )
        return _Inspection(detected, findings, verdict, malware.signature)

    async def _duplicate_check(
        self, principal: Principal, sha256: str, allow_duplicate: bool
    ) -> uuid.UUID | None:
        """An existing document *the principal can see* with a version of identical content.

        Invisible documents never count, so the check cannot be used to probe for files.
        """
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            found = (
                await session.execute(
                    select(Document.id)
                    .join(
                        DocumentVersion,
                        and_(
                            DocumentVersion.document_id == Document.id,
                            DocumentVersion.organization_id == Document.organization_id,
                        ),
                    )
                    .where(
                        DocumentVersion.organization_id == principal.org_id,
                        DocumentVersion.sha256 == sha256,
                        visible_clause(principal, now),
                    )
                    .order_by(Document.created_at, Document.id)
                    .limit(1)
                )
            ).scalar_one_or_none()
        if found is not None and not allow_duplicate:
            raise Conflict(
                "An identical file already exists.",
                internal_detail="duplicate upload",
                extra={"duplicate_of": str(found)},
            )
        return found

    async def _insert_document(
        self,
        principal: Principal,
        org_id: uuid.UUID,
        *,
        meta: _Metadata,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        key: str,
        spool: SpooledUpload,
        inspection: _Inspection,
        safe_name: str,
        extension: str,
        declared_mime: str | None,
        duplicate_of: uuid.UUID | None,
        source: _Source,
    ) -> UploadResult:
        quarantined = inspection.verdict is Verdict.QUARANTINE
        doc_status = DocumentStatus.QUARANTINED if quarantined else DocumentStatus.PROCESSING
        version_status = VersionStatus.QUARANTINED if quarantined else VersionStatus.UPLOADED
        async with self._db.session(principal.db_context) as session:
            session.add(
                Document(
                    id=document_id,
                    organization_id=org_id,
                    department_id=meta.department_id,
                    owner_id=principal.user_id,
                    title=meta.title or _title_from_filename(safe_name),
                    classification=meta.classification.value,
                    doc_type=(meta.doc_type or DocumentType.OTHER).value,
                    doc_type_source="user" if meta.doc_type else "auto",
                    allowed_roles=meta.allowed_roles,
                    tags=meta.tags,
                    status=doc_status.value,
                    version_count=1,
                )
            )
            try:
                await session.flush()
            except IntegrityError as exc:
                raise ValidationFailed(
                    "The department does not exist.", internal_detail="document insert failed"
                ) from exc
            session.add(
                self._version_row(
                    org_id,
                    document_id,
                    version_id,
                    number=1,
                    key=key,
                    spool=spool,
                    inspection=inspection,
                    safe_name=safe_name,
                    extension=extension,
                    declared_mime=declared_mime,
                    change_note=None,
                    status=version_status,
                    created_by=principal.user_id,
                )
            )
            await session.flush()
            job_id = (
                None
                if quarantined
                else await self._enqueue_ingest(session, org_id, version_id, principal)
            )
            self._audit.record(
                session,
                Actor.of(principal),
                "document.quarantined" if quarantined else "document.upload",
                resource_type="document",
                resource_id=document_id,
                details=self._upload_details(
                    version_id=version_id,
                    number=1,
                    spool=spool,
                    extension=extension,
                    inspection=inspection,
                    duplicate_of=duplicate_of,
                    source=source,
                    classification=meta.classification,
                ),
            )
            await session.commit()
        self._count_outcome(inspection)
        return UploadResult(
            document_id=document_id,
            version_id=version_id,
            version_number=1,
            status=doc_status,
            version_status=version_status,
            job_id=job_id,
            detected_mime=inspection.detected.mime,
            size_bytes=spool.size,
            sha256=spool.sha256,
            findings=[f.code for f in inspection.findings],
            duplicate_of=duplicate_of,
        )

    async def _insert_version(
        self,
        principal: Principal,
        org_id: uuid.UUID,
        document_id: uuid.UUID,
        *,
        version_id: uuid.UUID,
        key: str,
        spool: SpooledUpload,
        inspection: _Inspection,
        safe_name: str,
        extension: str,
        declared_mime: str | None,
        change_note: str | None,
        duplicate_of: uuid.UUID | None,
    ) -> UploadResult:
        quarantined = inspection.verdict is Verdict.QUARANTINE
        version_status = VersionStatus.QUARANTINED if quarantined else VersionStatus.UPLOADED
        now = utcnow()
        async with self._db.session(principal.db_context) as session:
            # Re-check under the row lock: rights may have changed while the file streamed in.
            loaded = await self._load(session, principal, org_id, document_id, now, lock=True)
            if not loaded.can_manage:
                raise await self._deny(principal, document_id, "add_version")
            doc = loaded.document
            highest = (
                await session.execute(
                    select(func.max(DocumentVersion.version_number)).where(
                        DocumentVersion.organization_id == org_id,
                        DocumentVersion.document_id == document_id,
                    )
                )
            ).scalar_one_or_none()
            number = int(highest or 0) + 1
            session.add(
                self._version_row(
                    org_id,
                    document_id,
                    version_id,
                    number=number,
                    key=key,
                    spool=spool,
                    inspection=inspection,
                    safe_name=safe_name,
                    extension=extension,
                    declared_mime=declared_mime,
                    change_note=change_note,
                    status=version_status,
                    created_by=principal.user_id,
                )
            )
            doc.version_count = number
            previous_status = doc.status
            if doc.status != DocumentStatus.READY.value:
                doc.status = (
                    DocumentStatus.QUARANTINED if quarantined else DocumentStatus.PROCESSING
                ).value
            await session.flush()
            if doc.status != previous_status:
                await self._vector_job(
                    session, VECTOR_SYNC_DOCUMENT_JOB, org_id, document_id, principal
                )
            job_id = (
                None
                if quarantined
                else await self._enqueue_ingest(session, org_id, version_id, principal)
            )
            self._audit.record(
                session,
                Actor.of(principal),
                "document.quarantined" if quarantined else "document.version_added",
                resource_type="document",
                resource_id=document_id,
                details=self._upload_details(
                    version_id=version_id,
                    number=number,
                    spool=spool,
                    extension=extension,
                    inspection=inspection,
                    duplicate_of=duplicate_of,
                    source=_Source("upload"),
                    classification=Classification(doc.classification),
                ),
            )
            status = DocumentStatus(doc.status)
            await session.commit()
        self._count_outcome(inspection)
        return UploadResult(
            document_id=document_id,
            version_id=version_id,
            version_number=number,
            status=status,
            version_status=version_status,
            job_id=job_id,
            detected_mime=inspection.detected.mime,
            size_bytes=spool.size,
            sha256=spool.sha256,
            findings=[f.code for f in inspection.findings],
            duplicate_of=duplicate_of,
        )

    @staticmethod
    def _version_row(
        org_id: uuid.UUID,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        *,
        number: int,
        key: str,
        spool: SpooledUpload,
        inspection: _Inspection,
        safe_name: str,
        extension: str,
        declared_mime: str | None,
        change_note: str | None,
        status: VersionStatus,
        created_by: uuid.UUID,
    ) -> DocumentVersion:
        declared = _clean_optional(declared_mime, 127)
        return DocumentVersion(
            id=version_id,
            organization_id=org_id,
            document_id=document_id,
            version_number=number,
            storage_key=key,
            original_filename=safe_name,
            extension=extension,
            declared_mime=declared.lower() if declared else None,
            detected_mime=inspection.detected.mime,
            size_bytes=spool.size,
            sha256=spool.sha256,
            status=status.value,
            security_findings=[f.as_dict() for f in inspection.findings],
            change_note=change_note,
            created_by=created_by,
        )

    @staticmethod
    async def _enqueue_ingest(
        session: AsyncSession, org_id: uuid.UUID, version_id: uuid.UUID, principal: Principal
    ) -> uuid.UUID | None:
        return await enqueue(
            session,
            kind=INGEST_JOB,
            organization_id=org_id,
            payload={"version_id": str(version_id)},
            idempotency_key=f"ingest:{version_id}",
            created_by=principal.user_id,
        )

    @staticmethod
    def _upload_details(
        *,
        version_id: uuid.UUID,
        number: int,
        spool: SpooledUpload,
        extension: str,
        inspection: _Inspection,
        duplicate_of: uuid.UUID | None,
        source: _Source,
        classification: Classification,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "version_id": str(version_id),
            "version_number": number,
            "size_bytes": spool.size,
            "sha256": spool.sha256,
            "extension": extension,
            "classification": classification.value,
            "source": source.kind,
            "findings": [f.code for f in inspection.findings],
        }
        if source.host:
            details["source_host"] = source.host
        if duplicate_of is not None:
            details["duplicate_of"] = str(duplicate_of)
        if inspection.signature:
            details["malware_signature"] = inspection.signature
        return details

    @staticmethod
    def _count_outcome(inspection: _Inspection) -> None:
        if inspection.verdict is Verdict.QUARANTINE:
            metrics.SECURITY_EVENTS.labels(kind="upload_quarantined").inc()

    async def _record_rejection(self, principal: Principal, reason: str, extension: str) -> None:
        metrics.SECURITY_EVENTS.labels(kind="upload_rejected").inc()
        await self._audit.record_detached(
            Actor.of(principal),
            "document.upload_rejected",
            outcome=AuditOutcome.DENIED,
            details={"reason": reason, "extension": extension},
        )

    async def _discard_blob(self, key: str) -> None:
        try:
            await self._storage.delete(key)
        except (OSError, StorageError):
            log.exception("orphan_blob_delete_failed")

    # ============================================================== metadata
    async def _metadata(
        self,
        principal: Principal,
        *,
        title: str | None,
        classification: Classification | str,
        department_id: uuid.UUID | None,
        doc_type: DocumentType | str | None,
        tags: Iterable[str],
        allowed_roles: Iterable[str | Role],
    ) -> _Metadata:
        try:
            level = Classification(classification)
            kind = DocumentType(doc_type) if doc_type else None
        except ValueError as exc:
            raise ValidationFailed("Unknown classification or document type.") from exc
        if level.rank > principal.clearance.rank:
            raise await self._deny(
                principal,
                None,
                "classify_above_clearance",
                message="You cannot classify a document above your own clearance.",
            )
        if department_id is not None:
            if principal.role is Role.ORGANIZATION_ADMIN:
                async with self._db.session(principal.db_context) as session:
                    await self._check_department(session, principal.require_org(), department_id)
            elif department_id not in principal.department_ids:
                raise await self._deny(
                    principal,
                    None,
                    "file_in_foreign_department",
                    message="You can only file documents in your own departments.",
                )
        cleaned_title = clean_line_text(title, 300) if title else ""
        return _Metadata(
            title=cleaned_title or None,
            classification=level,
            department_id=department_id,
            doc_type=kind,
            tags=normalize_tags(tags),
            allowed_roles=normalize_roles(allowed_roles),
        )

    @staticmethod
    async def _check_department(
        session: AsyncSession, org_id: uuid.UUID, department_id: uuid.UUID
    ) -> None:
        found = (
            await session.execute(
                select(Department.id).where(
                    Department.id == department_id, Department.organization_id == org_id
                )
            )
        ).scalar_one_or_none()
        if found is None:
            raise ValidationFailed("The department does not exist.")

    async def _check_grantee(
        self, session: AsyncSession, org_id: uuid.UUID, grant: GrantCreate
    ) -> None:
        if grant.grantee_type is GranteeType.USER:
            found = (
                await session.execute(
                    select(User.id).where(User.id == grant.user_id, User.organization_id == org_id)
                )
            ).scalar_one_or_none()
            if found is None:
                raise ValidationFailed("The user does not exist in this organisation.")
        elif grant.grantee_type is GranteeType.DEPARTMENT:
            if grant.department_id is None:
                raise ValidationFailed("department_id is required.")
            await self._check_department(session, org_id, grant.department_id)
        elif grant.role is None or grant.role.value not in TENANT_ROLES:
            raise ValidationFailed("Only organisation roles can be granted access.")

    async def _apply_patch(
        self,
        session: AsyncSession,
        principal: Principal,
        *,
        org_id: uuid.UUID,
        doc: Document,
        patch: DocumentUpdate,
        fields: set[str],
    ) -> list[str]:
        changed: list[str] = []
        is_admin = principal.role is Role.ORGANIZATION_ADMIN
        if "title" in fields and patch.title is not None:
            title = clean_line_text(patch.title, 300)
            if not title:
                raise ValidationFailed("The title cannot be empty.")
            if title != doc.title:
                doc.title = title
                changed.append("title")
        if "tags" in fields and patch.tags is not None:
            tags = normalize_tags(patch.tags)
            if tags != list(doc.tags or ()):
                doc.tags = tags
                changed.append("tags")
        if (
            "doc_type" in fields
            and patch.doc_type is not None
            and (patch.doc_type.value != doc.doc_type or doc.doc_type_source != "user")
        ):
            doc.doc_type = patch.doc_type.value
            doc.doc_type_source = "user"
            doc.doc_type_confidence = None
            changed.append("doc_type")
        if "classification" in fields and patch.classification is not None:
            level = patch.classification
            if not is_admin and level.rank > principal.clearance.rank:
                raise await self._deny(
                    principal,
                    doc.id,
                    "classify_above_clearance",
                    message="You cannot classify a document above your own clearance.",
                )
            if level.value != doc.classification:
                doc.classification = level.value
                changed.append("classification")
            suggested = doc.suggested_classification
            if suggested and Classification(suggested).rank <= level.rank:
                doc.suggested_classification = None
        if "department_id" in fields:
            target = patch.department_id
            if target is not None:
                if not is_admin and target not in principal.managed_department_ids:
                    raise await self._deny(
                        principal,
                        doc.id,
                        "move_to_unmanaged_department",
                        message="You can only move documents into departments you manage.",
                    )
                await self._check_department(session, org_id, target)
            if target != doc.department_id:
                doc.department_id = target
                changed.append("department_id")
        if "allowed_roles" in fields and patch.allowed_roles is not None:
            roles = normalize_roles(patch.allowed_roles)
            if roles != sorted(doc.allowed_roles or ()):
                doc.allowed_roles = roles
                changed.append("allowed_roles")
        if "retention_until" in fields:
            until = patch.retention_until
            if until is not None and not _EARLIEST_RETENTION <= until <= _LATEST_RETENTION:
                raise ValidationFailed("retention_until is out of range.")
            if until != doc.retention_until:
                doc.retention_until = until
                changed.append("retention_until")
        if "legal_hold" in fields and patch.legal_hold is not None:
            if not is_admin:
                raise await self._deny(
                    principal,
                    doc.id,
                    "legal_hold",
                    message="Only organisation administrators can change legal holds.",
                )
            if patch.legal_hold != doc.legal_hold:
                doc.legal_hold = patch.legal_hold
                changed.append("legal_hold")
        return changed

    # ================================================================ loading
    @staticmethod
    def _require(principal: Principal, permission: Permission) -> uuid.UUID:
        principal.require(permission)
        return principal.require_org()

    @staticmethod
    async def _load(
        session: AsyncSession,
        principal: Principal,
        org_id: uuid.UUID,
        document_id: uuid.UUID,
        now: datetime,
        *,
        lock: bool = False,
    ) -> _Loaded:
        """Fetch one document *through* the visibility policy; ``NotFound`` if invisible."""
        stmt = select(
            Document,
            case((readable_clause(principal, now), True), else_=False).label("can_read"),
            case((manageable_clause(principal, now), True), else_=False).label("can_manage"),
        ).where(
            Document.id == document_id,
            Document.organization_id == org_id,
            visible_clause(principal, now),
        )
        if lock:
            stmt = stmt.with_for_update(of=Document)
        row = (await session.execute(stmt)).first()
        if row is None:
            raise NotFound("Document not found.")
        return _Loaded(row[0], bool(row[1]), bool(row[2]))

    @staticmethod
    async def _pick_version(
        session: AsyncSession, doc: Document, version_number: int | None
    ) -> DocumentVersion:
        stmt = select(DocumentVersion).where(
            DocumentVersion.organization_id == doc.organization_id,
            DocumentVersion.document_id == doc.id,
        )
        if version_number is not None:
            stmt = stmt.where(DocumentVersion.version_number == version_number)
        elif doc.current_version_id is not None:
            stmt = stmt.where(DocumentVersion.id == doc.current_version_id)
        else:
            stmt = stmt.order_by(DocumentVersion.version_number.desc()).limit(1)
        version = (await session.execute(stmt)).scalars().first()
        if version is None:
            raise NotFound("Version not found.")
        return version

    @staticmethod
    async def _grants(
        session: AsyncSession, org_id: uuid.UUID, document_id: uuid.UUID
    ) -> list[DocumentGrant]:
        return list(
            (
                await session.execute(
                    select(DocumentGrant)
                    .where(
                        DocumentGrant.organization_id == org_id,
                        DocumentGrant.document_id == document_id,
                    )
                    .order_by(DocumentGrant.created_at, DocumentGrant.id)
                )
            )
            .scalars()
            .all()
        )

    async def _detail(
        self, session: AsyncSession, loaded: _Loaded, now: datetime
    ) -> DocumentDetail:
        doc = loaded.document
        versions = list(
            (
                await session.execute(
                    select(DocumentVersion)
                    .where(
                        DocumentVersion.organization_id == doc.organization_id,
                        DocumentVersion.document_id == doc.id,
                    )
                    .order_by(DocumentVersion.version_number.desc())
                )
            )
            .scalars()
            .all()
        )
        grants = (
            [_grant_out(g, now) for g in await self._grants(session, doc.organization_id, doc.id)]
            if loaded.can_manage
            else None
        )
        current = next((v for v in versions if v.id == doc.current_version_id), None)
        latest = versions[0] if versions else None
        return DocumentDetail(
            **_summary_fields(doc, loaded.can_read, loaded.can_manage),
            retention_until=doc.retention_until,
            sensitivity_signals=list(doc.sensitivity_signals or ()),
            metadata=_sanitize_metadata(current.doc_metadata)
            if current is not None and loaded.can_read
            else {},
            ingestion=IngestionInfo(
                version_number=latest.version_number,
                status=VersionStatus(latest.status),
                error_code=_clean_optional(latest.error_code, 64),
            )
            if latest is not None
            else None,
            versions=[
                VersionInfo(
                    id=v.id,
                    version_number=v.version_number,
                    status=VersionStatus(v.status),
                    is_current=v.id == doc.current_version_id,
                    original_filename=sanitize_filename(v.original_filename),
                    extension=v.extension,
                    detected_mime=v.detected_mime,
                    size_bytes=v.size_bytes,
                    sha256=v.sha256,
                    page_count=v.page_count,
                    chunk_count=v.chunk_count,
                    needs_ocr=v.needs_ocr,
                    semantic_indexed=v.semantic_indexed,
                    error_code=_clean_optional(v.error_code, 64),
                    change_note=_clean_optional(v.change_note, 500),
                    created_by=v.created_by,
                    created_at=v.created_at,
                    processed_at=v.processed_at,
                    findings=_findings_out(v.security_findings) if loaded.can_manage else None,
                )
                for v in versions
            ],
            grants=grants,
        )

    # ============================================================== downloads
    async def _authorize_quarantined_download(
        self,
        session: AsyncSession,
        principal: Principal,
        loaded: _Loaded,
        now: datetime,
        *,
        acknowledge_risk: bool,
    ) -> None:
        doc = loaded.document
        if not loaded.can_manage:
            raise await self._deny(principal, doc.id, "download_quarantined")
        if not acknowledge_risk:
            raise PermissionDenied(
                "This file is quarantined. Downloading it requires acknowledge_risk=true.",
                internal_detail="quarantined download without acknowledgement",
                extra={"reason": "acknowledge_risk_required"},
            )
        # Managing is not reading (org admins manage RESTRICTED files they may not read):
        # the manager must also be entitled to the content, judged as if it were ready.
        grants = await self._grants(session, doc.organization_id, doc.id)
        facts = dataclasses.replace(
            DocumentFacts.from_row(doc, grants), status=DocumentStatus.READY
        )
        if not can_read(principal, facts, now):
            raise await self._deny(principal, doc.id, "download_quarantined")

    async def _open_blob(
        self, org_id: uuid.UUID, version: DocumentVersion
    ) -> tuple[bytes, Iterator[bytes]]:
        """Open and authenticate the first segment now, so tampering or a missing object
        becomes an error response instead of a truncated download."""
        iterator = self._storage.iter_plaintext(
            version.storage_key, storage_context(org_id, "version", version.id)
        )
        try:
            first = await asyncio.to_thread(next, iterator, b"")
        except (StorageError, DecryptionError, OSError) as exc:
            metrics.SECURITY_EVENTS.labels(kind="storage_integrity_failure").inc()
            log.exception("document_blob_unreadable", error_type=type(exc).__name__)
            raise ServiceUnavailable(
                "The file could not be retrieved.", internal_detail=type(exc).__name__
            ) from exc
        return first, iterator

    @staticmethod
    def _served_type(version: DocumentVersion, first: bytes) -> str:
        mime = version.detected_mime
        if version.extension in TEXT_EXTENSIONS:
            utf16 = first.startswith((b"\xff\xfe", b"\xfe\xff"))
            return f"{mime}; charset={'utf-16' if utf16 else 'utf-8'}"
        return mime

    # ================================================================== misc
    async def _deny(
        self,
        principal: Principal,
        document_id: uuid.UUID | None,
        operation: str,
        *,
        message: str | None = None,
    ) -> PermissionDenied:
        await self._audit.record_detached(
            Actor.of(principal),
            "document.access_denied",
            outcome=AuditOutcome.DENIED,
            resource_type="document",
            resource_id=document_id,
            details={"operation": operation},
        )
        return PermissionDenied(message, internal_detail=f"document {operation} denied")

    async def _vector_job(
        self,
        session: AsyncSession,
        kind: str,
        org_id: uuid.UUID,
        document_id: uuid.UUID,
        principal: Principal,
    ) -> None:
        """Keep an external vector store in step (Qdrant backend only), in the caller's
        transaction: ``vector_sync_document`` whenever a field carried in the point payloads
        changes (classification, department, allowed roles, grants, type, tags, status),
        ``vector_delete_document`` when the document is deleted."""
        if self._settings.retrieval.backend != "qdrant":
            return
        key = (
            f"vdelete:{document_id}"
            if kind == VECTOR_DELETE_DOCUMENT_JOB
            else f"vsync:{document_id}:{uuid7()}"
        )
        await enqueue(
            session,
            kind=kind,
            organization_id=org_id,
            payload={"document_id": str(document_id)},
            idempotency_key=key,
            created_by=principal.user_id,
        )

    @staticmethod
    async def _delete_rows(session: AsyncSession, statement: Any) -> int:
        result = cast("CursorResult[Any]", await session.execute(statement))
        return int(result.rowcount or 0)


__all__ = [
    "INGEST_JOB",
    "VECTOR_DELETE_DOCUMENT_JOB",
    "VECTOR_SYNC_DOCUMENT_JOB",
    "DocumentService",
    "DocumentServiceDeps",
    "DownloadPayload",
    "normalize_roles",
    "normalize_tags",
    "visible_clause",
]
