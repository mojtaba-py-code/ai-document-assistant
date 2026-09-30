"""Secure exports of extracted fields, deadlines, search results and document reports.

Lifecycle
    ``create`` (permission ``export:create``, rate limit ``export_per_user``) validates the
    parameters, then in ONE transaction inserts the ``exports`` row (``pending``), enqueues
    the ``export_generate`` job and audits ``export.created``. The worker job rebuilds the
    creator's principal from the database *at generation time* (a disabled user or a
    revoked permission stops the export), collects rows with the readable clause in every
    query, renders CSV/JSON, encrypts the blob bound to ``storage_context(org, "export",
    id)`` and marks the row ``ready`` with ``expires_at = now + retention.export_ttl_hours``
    (audited as ``export.generated`` with the row count).

Row cap
    At most :data:`EXPORT_MAX_ROWS` rows; an organisation may lower it with
    ``organizations.settings["exports"]["max_rows"]``. Larger results are truncated to the
    cap and flagged (``truncated`` in the API, the JSON body and the audit event).

Downloads
    Only the creating user, only while ``ready`` and not expired (410 afterwards):
    * ``GET /exports/{id}/download`` with the user's bearer token;
    * a signed link from ``POST /exports/{id}/link``: an HMAC (``TokenService.sign``,
      purpose ``"export"``) over ``export_id|org_id|user_id|expires|download_count`` that
      lives at most :data:`LINK_TTL_SECONDS`. Redeeming it atomically increments the
      export's ``download_count`` *only if it still equals the value signed into the
      link*, so every link works once and any download consumes all outstanding links.
    Every download is audited as ``export.downloaded`` with the row count.

CSV safety
    UTF-8 with BOM, every field quoted, invisible/control characters removed, and any text
    cell starting with ``= + - @`` TAB or CR (also after leading spaces) prefixed with ``'``
    so spreadsheet applications never evaluate it as a formula. Typed numbers (int,
    Decimal, float) are written as plain numbers and need no prefix.
"""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import and_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor
from docassist.authz.permissions import Permission, permissions_for
from docassist.authz.policy import readable_clause
from docassist.authz.principal import Principal
from docassist.core.context import utcnow
from docassist.core.enums import (
    AuditOutcome,
    Classification,
    ExportStatus,
    OrganizationStatus,
    Role,
    UserStatus,
)
from docassist.core.errors import (
    AppError,
    Conflict,
    NotFound,
    PermissionDenied,
    ServiceUnavailable,
    ValidationFailed,
)
from docassist.core.ids import uuid7
from docassist.core.logging import get_logger
from docassist.core.text import clean_line_text, sanitize_text
from docassist.db.models import (
    Document,
    DocumentVersion,
    Export,
    ExtractedField,
    Organization,
    User,
    UserDepartment,
)
from docassist.db.session import Database, DbContext
from docassist.documents.storage import StorageError, new_object_key, storage_context
from docassist.intelligence.access import field_row, load_readable_document
from docassist.intelligence.deadlines import local_today, query_deadlines, resolve_fields, window
from docassist.intelligence.schemas import (
    DeadlinesExportParams,
    DocumentReport,
    DocumentReportExportParams,
    ExportCreateRequest,
    ExportLink,
    ExportList,
    ExportOut,
    ExtractedFieldsExportParams,
    SearchResultsExportParams,
)
from docassist.intelligence.textops import format_decimal, parse_timezone
from docassist.jobs.queue import JobError, PermanentJobError, enqueue
from docassist.observability import metrics
from docassist.search.types import SearchFilters
from docassist.security.crypto import DecryptionError
from docassist.security.tokens import TokenService

if TYPE_CHECKING:
    from docassist.api.container import Container
    from docassist.intelligence.reports import ReportBuilder
    from docassist.search.service import SearchService

log = get_logger(__name__)

EXPORT_MAX_ROWS = 10_000
MAX_EXPORT_BYTES = 25 * 1024 * 1024
MAX_CELL_CHARS = 32_000
LINK_TTL_SECONDS = 300
LINK_PURPOSE = "export"
JOB_KIND = "export_generate"
EXPORT_RATE_BUCKET = "export_user"
LINK_RATE_BUCKET = "export_link_ip"
META_KEY = "_meta"
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_NO_SESSION = uuid.UUID(int=0)

PARAM_MODELS: dict[str, type[BaseModel]] = {
    "extracted_fields": ExtractedFieldsExportParams,
    "deadlines": DeadlinesExportParams,
    "search_results": SearchResultsExportParams,
    "document_report": DocumentReportExportParams,
}
COLUMNS: dict[str, tuple[str, ...]] = {
    "extracted_fields": (
        "document_id", "document_title", "doc_type", "version_number", "field", "value",
        "value_date", "value_number", "currency", "confidence", "method", "page", "evidence",
    ),
    "deadlines": (
        "document_id", "document_title", "doc_type", "field", "date", "days_left", "value",
        "confidence", "method", "page", "evidence",
    ),
    "search_results": (
        "rank", "document_id", "document_title", "version_number", "page_start", "page_end",
        "section", "score", "snippet",
    ),
    "document_report": ("section", "item", "value", "page", "detail"),
}  # fmt: skip
MEDIA_TYPES = {"csv": "text/csv; charset=utf-8", "json": "application/json"}


class Gone(AppError):
    status_code = 410
    code = "gone"
    title = "Gone"
    default_message = "The resource is no longer available."


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def csv_safe_cell(value: object) -> str:
    """One CSV cell: typed numbers verbatim, text sanitised and formula-neutralised."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return format_decimal(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    text, _ = sanitize_text(str(value))
    text = text[:MAX_CELL_CHARS]
    if text.startswith(FORMULA_PREFIXES) or text.lstrip(" ").startswith(FORMULA_PREFIXES):
        return "'" + text
    return text


def render_csv(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([csv_safe_cell(v) for v in row])
    return buffer.getvalue().encode("utf-8-sig")


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return format_decimal(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, str):
        return sanitize_text(value)[0][:MAX_CELL_CHARS]
    return value


def render_json(
    meta: dict[str, Any],
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    extra: dict[str, Any] | None = None,
) -> bytes:
    body: dict[str, Any] = {
        "export": meta,
        "columns": list(columns),
        "rows": [{c: _json_value(v) for c, v in zip(columns, row, strict=True)} for row in rows],
    }
    if extra:
        body.update(extra)
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


# --------------------------------------------------------------------------- #
# Signed links
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"^v1\.([0-9a-f]{32})\.([0-9a-f]{32})\.([0-9a-f]{32})\.(\d{1,12})\.(\d{1,9})\.([0-9a-f]{64})$"
)


@dataclass(frozen=True, slots=True)
class LinkToken:
    export_id: uuid.UUID
    org_id: uuid.UUID
    user_id: uuid.UUID
    expires: int
    counter: int


def _link_payload(
    export_id: uuid.UUID, org_id: uuid.UUID, user_id: uuid.UUID, expires: int, counter: int
) -> str:
    return f"{export_id}|{org_id}|{user_id}|{expires}|{counter}"


def build_link_token(
    tokens: TokenService,
    *,
    export_id: uuid.UUID,
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    expires: int,
    counter: int,
) -> str:
    signature = tokens.sign(
        LINK_PURPOSE, _link_payload(export_id, org_id, user_id, expires, counter)
    )
    return f"v1.{export_id.hex}.{org_id.hex}.{user_id.hex}.{expires}.{counter}.{signature}"


def parse_link_token(tokens: TokenService, token: str) -> LinkToken | None:
    """The token's claims if it is well-formed and its signature verifies, else ``None``."""
    if not token or len(token) > 512:
        return None
    m = _TOKEN_RE.fullmatch(token)
    if m is None:
        return None
    export_id, org_id, user_id = (uuid.UUID(hex=m[i]) for i in (1, 2, 3))
    expires, counter = int(m[4]), int(m[5])
    payload = _link_payload(export_id, org_id, user_id, expires, counter)
    if not tokens.verify_signature(LINK_PURPOSE, payload, m[6]):
        return None
    return LinkToken(export_id, org_id, user_id, expires, counter)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ExportFile:
    filename: str
    media_type: str
    data: bytes
    row_count: int


def effective_status(row: Export, now: datetime) -> str:
    if (
        row.status in (ExportStatus.PENDING.value, ExportStatus.READY.value)
        and row.expires_at <= now
    ):
        return ExportStatus.EXPIRED.value
    return row.status


def to_out(row: Export, now: datetime) -> ExportOut:
    params = dict(row.params or {})
    meta = params.pop(META_KEY, None)
    return ExportOut(
        id=row.id,
        kind=row.kind,
        format=row.format,
        status=effective_status(row, now),
        params=params,
        row_count=row.row_count,
        size_bytes=row.size_bytes,
        truncated=bool(isinstance(meta, dict) and meta.get("truncated")),
        error_code=row.error_code,
        created_at=row.created_at,
        ready_at=row.ready_at,
        expires_at=row.expires_at,
        download_count=row.download_count,
        last_downloaded_at=row.last_downloaded_at,
    )


def ensure_downloadable(row: Export, now: datetime) -> None:
    status = effective_status(row, now)
    if status == ExportStatus.EXPIRED.value:
        raise Gone("The export has expired.")
    if status == ExportStatus.FAILED.value:
        raise Conflict("The export failed.", extra={"error_code": row.error_code})
    if status != ExportStatus.READY.value or not row.storage_key:
        raise Conflict("The export is not ready yet.")


def export_filename(row: Export) -> str:
    stamp = (row.ready_at or row.created_at).strftime("%Y%m%d-%H%M%S")
    return f"export-{row.kind.replace('_', '-')}-{stamp}.{row.format}"


def _validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {
            "loc": [str(p) for p in err.get("loc", ())][:6],
            "msg": str(err.get("msg", "invalid"))[:200],
        }
        for err in exc.errors()[:20]
    ]


async def principal_for(
    session: AsyncSession, org_id: uuid.UUID, user_id: uuid.UUID
) -> Principal | None:
    """The user's CURRENT principal (role, clearance, departments), or ``None`` if the user
    or the organisation is no longer active."""
    user = (
        await session.execute(
            select(User).where(User.id == user_id, User.organization_id == org_id)
        )
    ).scalar_one_or_none()
    if user is None or user.status != UserStatus.ACTIVE.value:
        return None
    org_status = (
        await session.execute(select(Organization.status).where(Organization.id == org_id))
    ).scalar_one_or_none()
    if org_status != OrganizationStatus.ACTIVE.value:
        return None
    memberships = (
        await session.execute(
            select(UserDepartment.department_id, UserDepartment.is_manager).where(
                UserDepartment.user_id == user_id, UserDepartment.organization_id == org_id
            )
        )
    ).all()
    return Principal(
        user_id=user.id,
        org_id=org_id,
        role=Role(user.role),
        clearance=Classification(user.clearance),
        session_id=_NO_SESSION,
        email=user.email,
        department_ids=frozenset(m.department_id for m in memberships),
        managed_department_ids=frozenset(m.department_id for m in memberships if m.is_manager),
    )


@dataclass(frozen=True, slots=True)
class Collected:
    columns: tuple[str, ...]
    rows: Sequence[Sequence[object]]
    truncated: bool
    extra: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _Pending:
    org_id: uuid.UUID
    export_id: uuid.UUID
    user_id: uuid.UUID
    kind: str
    fmt: str
    params: dict[str, Any]


class _Failed(Exception):
    """Generation failed permanently with a sanitized error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _admissible(
    principal: Principal | None, kind: str, params: dict[str, Any], *, expired: bool
) -> tuple[Principal, BaseModel]:
    """Principal and validated parameters, or :class:`_Failed` when the export may not be
    generated (expired, creator inactive, permission revoked, invalid parameters)."""
    if expired:
        raise _Failed("expired")
    if principal is None:
        raise _Failed("user_inactive")
    if not principal.has(Permission.EXPORT_CREATE):
        raise _Failed("permission_revoked")
    model = PARAM_MODELS.get(kind)
    if model is None:
        raise _Failed("invalid_params")
    try:
        return principal, model.model_validate(params)
    except ValidationError as exc:
        raise _Failed("invalid_params") from exc


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class ExportService:
    def __init__(self, container: Container, reports: ReportBuilder) -> None:
        self._c = container
        self._reports = reports

    # ------------------------------------------------------------ API side
    async def create(self, principal: Principal, request: ExportCreateRequest) -> ExportOut:
        org_id = principal.require_org()
        principal.require(Permission.EXPORT_CREATE)
        params = self._validate_params(request.kind, request.params)
        settings = self._c.settings
        await self._c.limiter.enforce(
            EXPORT_RATE_BUCKET, str(principal.user_id), settings.rate_limit.export_per_user
        )
        now = utcnow()
        export_id = uuid7()
        async with self._c.db.transaction(principal.db_context) as session:
            if isinstance(params, DocumentReportExportParams):
                await load_readable_document(session, principal, params.document_id, now)
            row = Export(
                id=export_id,
                organization_id=org_id,
                user_id=principal.user_id,
                kind=request.kind,
                format=request.format,
                status=ExportStatus.PENDING.value,
                params=params.model_dump(mode="json"),
                download_count=0,
                created_at=now,
                expires_at=now + timedelta(hours=settings.retention.export_ttl_hours),
            )
            session.add(row)
            await session.flush()
            await enqueue(
                session,
                kind=JOB_KIND,
                organization_id=org_id,
                payload={"export_id": str(export_id)},
                idempotency_key=f"export:{export_id}",
                max_attempts=settings.worker.max_attempts,
                created_by=principal.user_id,
            )
            self._c.audit.record(
                session,
                Actor.of(principal),
                "export.created",
                resource_type="export",
                resource_id=export_id,
                details={"kind": request.kind, "format": request.format},
            )
            return to_out(row, now)

    def _validate_params(self, kind: str, raw: dict[str, Any]) -> BaseModel:
        model = PARAM_MODELS[kind]
        try:
            params = model.model_validate(raw)
        except ValidationError as exc:
            raise ValidationFailed(
                "Invalid export parameters.", extra={"errors": _validation_errors(exc)}
            ) from exc
        if isinstance(params, DeadlinesExportParams):
            resolve_fields(params.fields)
            parse_timezone(params.timezone)
        return params

    async def list_own(self, principal: Principal, *, limit: int = 50) -> ExportList:
        """The caller's exports, newest first (never anybody else's)."""
        org_id = principal.require_org()
        principal.require(Permission.EXPORT_CREATE)
        now = utcnow()
        async with self._c.db.session(principal.db_context) as session:
            rows = (
                await session.execute(
                    select(Export)
                    .where(Export.organization_id == org_id, Export.user_id == principal.user_id)
                    .order_by(Export.created_at.desc(), Export.id.desc())
                    .limit(max(1, min(limit, 100)))
                )
            ).scalars()
            return ExportList(items=[to_out(row, now) for row in rows])

    async def get(self, principal: Principal, export_id: uuid.UUID) -> ExportOut:
        principal.require(Permission.EXPORT_CREATE)
        async with self._c.db.session(principal.db_context) as session:
            row = await self._own(session, principal, export_id)
            return to_out(row, utcnow())

    async def _own(
        self,
        session: AsyncSession,
        principal: Principal,
        export_id: uuid.UUID,
        *,
        lock: bool = False,
    ) -> Export:
        """The caller's own export; anybody else's (same org or not) is reported missing."""
        org_id = principal.require_org()
        stmt = select(Export).where(
            Export.id == export_id,
            Export.organization_id == org_id,
            Export.user_id == principal.user_id,
        )
        if lock:
            stmt = stmt.with_for_update()
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise NotFound(internal_detail="export not visible")
        return row

    async def download(self, principal: Principal, export_id: uuid.UUID) -> ExportFile:
        principal.require(Permission.EXPORT_CREATE)
        now = utcnow()
        async with self._c.db.session(principal.db_context) as session:
            row = await self._own(session, principal, export_id, lock=True)
            ensure_downloadable(row, now)
            data = await self._read_blob(row)
            row.download_count += 1
            row.last_downloaded_at = now
            self._c.audit.record(
                session,
                Actor.of(principal),
                "export.downloaded",
                resource_type="export",
                resource_id=row.id,
                details={"row_count": row.row_count, "kind": row.kind, "via": "session"},
            )
            result = ExportFile(
                export_filename(row), MEDIA_TYPES[row.format], data, row.row_count or 0
            )
            await session.commit()
        return result

    async def create_link(self, principal: Principal, export_id: uuid.UUID) -> ExportLink:
        org_id = principal.require_org()
        principal.require(Permission.EXPORT_CREATE)
        now = utcnow()
        async with self._c.db.transaction(principal.db_context) as session:
            row = await self._own(session, principal, export_id)
            ensure_downloadable(row, now)
            expires_at = min(now + timedelta(seconds=LINK_TTL_SECONDS), row.expires_at)
            expires = int(expires_at.timestamp())
            token = build_link_token(
                self._c.tokens,
                export_id=row.id,
                org_id=org_id,
                user_id=principal.user_id,
                expires=expires,
                counter=row.download_count,
            )
            self._c.audit.record(
                session,
                Actor.of(principal),
                "export.link_created",
                resource_type="export",
                resource_id=row.id,
                details={"expires_in_seconds": expires - int(now.timestamp())},
            )
        base = self._c.settings.public_base_url.rstrip("/")
        return ExportLink(
            url=f"{base}/api/v1/exports/download?token={token}",
            expires_at=datetime.fromtimestamp(expires, UTC),
        )

    async def redeem_link(self, token: str, *, ip_prefix: str | None = None) -> ExportFile:
        """Download through a signed link: valid signature, not expired, unused."""
        claims = parse_link_token(self._c.tokens, token)
        if claims is None:
            await self._reject(None, None, ip_prefix, "invalid")
            raise NotFound()
        now = utcnow()
        now_ts = int(now.timestamp())
        if claims.expires <= now_ts:
            await self._reject(
                claims.org_id, claims.user_id, ip_prefix, "expired", claims.export_id
            )
            raise Gone("This download link has expired.")
        if claims.expires > now_ts + LINK_TTL_SECONDS + 60:
            await self._reject(
                claims.org_id, claims.user_id, ip_prefix, "lifetime", claims.export_id
            )
            raise NotFound()
        ctx = DbContext(org_id=claims.org_id, user_id=claims.user_id)
        async with self._c.db.session(ctx) as session:
            user = (
                await session.execute(
                    select(User).where(
                        User.id == claims.user_id, User.organization_id == claims.org_id
                    )
                )
            ).scalar_one_or_none()
            org_status = (
                await session.execute(
                    select(Organization.status).where(Organization.id == claims.org_id)
                )
            ).scalar_one_or_none()
            allowed = (
                user is not None
                and user.status == UserStatus.ACTIVE.value
                and org_status == OrganizationStatus.ACTIVE.value
                and Permission.EXPORT_CREATE in permissions_for(Role(user.role))
            )
            row = None
            if allowed:
                row = (
                    await session.execute(
                        select(Export)
                        .where(
                            Export.id == claims.export_id,
                            Export.organization_id == claims.org_id,
                            Export.user_id == claims.user_id,
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()
            if row is None or user is None:
                await session.rollback()
                await self._reject(
                    claims.org_id, claims.user_id, ip_prefix, "not_found", claims.export_id
                )
                raise NotFound()
            try:
                ensure_downloadable(row, now)
            except AppError:
                await session.rollback()
                await self._reject(
                    claims.org_id, claims.user_id, ip_prefix, "unavailable", claims.export_id
                )
                raise
            if row.download_count != claims.counter:
                await session.rollback()
                await self._reject(
                    claims.org_id, claims.user_id, ip_prefix, "reused", claims.export_id
                )
                raise Gone("This download link has already been used.")
            data = await self._read_blob(row)
            row.download_count += 1
            row.last_downloaded_at = now
            self._c.audit.record(
                session,
                Actor(user.id, user.role, claims.org_id, ip_prefix),
                "export.downloaded",
                resource_type="export",
                resource_id=row.id,
                details={"row_count": row.row_count, "kind": row.kind, "via": "link"},
            )
            result = ExportFile(
                export_filename(row), MEDIA_TYPES[row.format], data, row.row_count or 0
            )
            await session.commit()
        return result

    async def _reject(
        self,
        org_id: uuid.UUID | None,
        user_id: uuid.UUID | None,
        ip_prefix: str | None,
        reason: str,
        export_id: uuid.UUID | None = None,
    ) -> None:
        await self._c.audit.record_detached(
            Actor(user_id, None, org_id, ip_prefix),
            "export.link_rejected",
            outcome=AuditOutcome.DENIED,
            resource_type="export",
            resource_id=export_id,
            details={"reason": reason},
        )

    async def _read_blob(self, row: Export) -> bytes:
        if not row.storage_key:
            raise Conflict("The export is not ready yet.")
        context = storage_context(row.organization_id, "export", row.id)
        try:
            return await self._c.storage.read_bytes(row.storage_key, context, MAX_EXPORT_BYTES)
        except DecryptionError as exc:
            metrics.SECURITY_EVENTS.labels(kind="export_integrity").inc()
            log.exception("export_blob_integrity_failure", export_id=str(row.id))
            raise ServiceUnavailable(
                "The export file is unavailable.", internal_detail="decryption failed"
            ) from exc
        except (StorageError, OSError) as exc:
            raise ServiceUnavailable(
                "The export file is unavailable.", internal_detail=type(exc).__name__
            ) from exc

    # --------------------------------------------------------- worker side
    async def generate(
        self, org_id: uuid.UUID, export_id: uuid.UUID, *, worker_db: Database, final_attempt: bool
    ) -> dict[str, Any]:
        """Job body of ``export_generate``: build, encrypt, store and publish one export."""
        now = utcnow()
        async with worker_db.session(DbContext.system_for_org(org_id)) as session:
            row = (
                await session.execute(
                    select(Export).where(Export.id == export_id, Export.organization_id == org_id)
                )
            ).scalar_one_or_none()
            if row is None:
                raise PermanentJobError("export row not found", code="export_missing")
            if row.status != ExportStatus.PENDING.value:
                return {"export_id": str(export_id), "skipped": row.status}
            user_id, kind, fmt, params = row.user_id, row.kind, row.format, dict(row.params or {})
            expired = row.expires_at <= now
            principal = None if expired else await principal_for(session, org_id, user_id)
        try:
            actor, model = _admissible(principal, kind, params, expired=expired)
            cap = await self._row_cap(worker_db, org_id)
            collected = await self._collect_as(worker_db, actor, model, cap=cap, now=now)
            data = self._render(export_id, kind, fmt, now, collected)
        except _Failed as failure:
            await self._fail(worker_db, org_id, export_id, failure.code)
            raise PermanentJobError(f"export failed: {failure.code}", code=failure.code) from None
        except (SQLAlchemyError, ServiceUnavailable, OSError) as exc:
            if final_attempt:
                await self._fail(worker_db, org_id, export_id, "generation_failed")
            raise JobError(
                f"export generation failed: {type(exc).__name__}", code="export_retry"
            ) from exc
        pending = _Pending(org_id, export_id, user_id, kind, fmt, params)
        return await self._publish(
            worker_db, pending, data, collected, cap=cap, final_attempt=final_attempt
        )

    async def _collect_as(
        self,
        worker_db: Database,
        principal: Principal,
        params: BaseModel,
        *,
        cap: int,
        now: datetime,
    ) -> Collected:
        ctx = DbContext(org_id=principal.org_id, user_id=principal.user_id)
        try:
            async with worker_db.session(ctx) as session:
                return await self._collect(session, principal, params, cap=cap, now=now)
        except (NotFound, PermissionDenied, Conflict) as exc:
            raise _Failed("source_unavailable") from exc
        except ValidationFailed as exc:
            raise _Failed("invalid_params") from exc

    @staticmethod
    def _render(
        export_id: uuid.UUID, kind: str, fmt: str, now: datetime, collected: Collected
    ) -> bytes:
        if fmt == "csv":
            data = render_csv(collected.columns, collected.rows)
        else:
            meta = {
                "id": str(export_id),
                "kind": kind,
                "generated_at": now.isoformat(),
                "row_count": len(collected.rows),
                "truncated": collected.truncated,
            }
            data = render_json(meta, collected.columns, collected.rows, collected.extra)
        if len(data) > MAX_EXPORT_BYTES:
            raise _Failed("too_large")
        return data

    async def _publish(
        self,
        worker_db: Database,
        job: _Pending,
        data: bytes,
        collected: Collected,
        *,
        cap: int,
        final_attempt: bool,
    ) -> dict[str, Any]:
        org_id, export_id = job.org_id, job.export_id
        key = new_object_key(org_id)
        try:
            await self._c.storage.put_bytes(key, data, storage_context(org_id, "export", export_id))
        except (StorageError, OSError) as exc:
            if final_attempt:
                await self._fail(worker_db, org_id, export_id, "storage_failed")
            raise JobError("export storage failed", code="export_retry") from exc
        ready_at = utcnow()
        meta = {"truncated": collected.truncated, "row_cap": cap}
        try:
            async with worker_db.transaction(DbContext(org_id=org_id)) as session:
                published = (
                    await session.execute(
                        update(Export)
                        .where(
                            Export.id == export_id,
                            Export.organization_id == org_id,
                            Export.status == ExportStatus.PENDING.value,
                        )
                        .values(
                            status=ExportStatus.READY.value,
                            storage_key=key,
                            size_bytes=len(data),
                            row_count=len(collected.rows),
                            ready_at=ready_at,
                            expires_at=ready_at
                            + timedelta(hours=self._c.settings.retention.export_ttl_hours),
                            error_code=None,
                            params={**job.params, META_KEY: meta},
                        )
                        .returning(Export.id)
                    )
                ).scalar_one_or_none()
                if published is not None:
                    self._c.audit.record(
                        session,
                        Actor.system(org_id),
                        "export.generated",
                        resource_type="export",
                        resource_id=export_id,
                        details={
                            "user_id": str(job.user_id),
                            "kind": job.kind,
                            "format": job.fmt,
                            "row_count": len(collected.rows),
                            "truncated": collected.truncated,
                            "size_bytes": len(data),
                        },
                    )
        except SQLAlchemyError as exc:
            await self._delete_quietly(key)
            if final_attempt:
                await self._fail(worker_db, org_id, export_id, "publish_failed")
            raise JobError("export publish failed", code="export_retry") from exc
        except BaseException:
            await self._delete_quietly(key)
            raise
        if published is None:
            await self._delete_quietly(key)
            return {"export_id": str(export_id), "skipped": "state_changed"}
        return {
            "export_id": str(export_id),
            "row_count": len(collected.rows),
            "truncated": collected.truncated,
        }

    async def _delete_quietly(self, key: str) -> None:
        try:
            await self._c.storage.delete(key)
        except (StorageError, OSError):
            log.warning("export_blob_cleanup_failed")

    async def _fail(
        self, worker_db: Database, org_id: uuid.UUID, export_id: uuid.UUID, code: str
    ) -> None:
        async with worker_db.transaction(DbContext(org_id=org_id)) as session:
            changed = (
                await session.execute(
                    update(Export)
                    .where(
                        Export.id == export_id,
                        Export.organization_id == org_id,
                        Export.status == ExportStatus.PENDING.value,
                    )
                    .values(status=ExportStatus.FAILED.value, error_code=code[:64])
                    .returning(Export.id)
                )
            ).scalar_one_or_none()
            if changed is not None:
                self._c.audit.record(
                    session,
                    Actor.system(org_id),
                    "export.failed",
                    outcome=AuditOutcome.FAILURE,
                    resource_type="export",
                    resource_id=export_id,
                    details={"error_code": code},
                )

    async def _row_cap(self, worker_db: Database, org_id: uuid.UUID) -> int:
        async with worker_db.session(DbContext.system_for_org(org_id)) as session:
            settings: Any = (
                await session.execute(
                    select(Organization.settings).where(Organization.id == org_id)
                )
            ).scalar_one_or_none()
        exports: Any = settings.get("exports") if isinstance(settings, dict) else None
        configured: Any = exports.get("max_rows") if isinstance(exports, dict) else None
        if type(configured) is int and configured > 0:  # bool is excluded on purpose
            return min(configured, EXPORT_MAX_ROWS)
        return EXPORT_MAX_ROWS

    # ----------------------------------------------------------- collection
    async def _collect(
        self,
        session: AsyncSession,
        principal: Principal,
        params: BaseModel,
        *,
        cap: int,
        now: datetime,
    ) -> Collected:
        if isinstance(params, ExtractedFieldsExportParams):
            return await self._extracted_fields(session, principal, params, cap=cap, now=now)
        if isinstance(params, DeadlinesExportParams):
            zone = parse_timezone(params.timezone)
            today = local_today(now, zone)
            start, end = window(today, params.within_days, params.overdue_days)
            items, truncated = await query_deadlines(
                session,
                principal,
                now=now,
                start=start,
                end=end,
                today=today,
                fields=resolve_fields(params.fields),
                doc_type=params.doc_type.value if params.doc_type else None,
                limit=cap,
            )
            rows: list[tuple[object, ...]] = [
                (
                    i.document_id, i.document_title, i.doc_type, i.field, i.date, i.days_left,
                    i.value_text, i.confidence, i.method, i.page, i.evidence,
                )
                for i in items
            ]  # fmt: skip
            return Collected(COLUMNS["deadlines"], rows, truncated)
        if isinstance(params, SearchResultsExportParams):
            return await self._search_results(principal, params, cap)
        if isinstance(params, DocumentReportExportParams):
            report = await self._reports.build(session, principal, params.document_id, now=now)
            flat = report_rows(report)
            return Collected(
                COLUMNS["document_report"],
                flat[:cap],
                len(flat) > cap,
                {"report": report.model_dump(mode="json")},
            )
        raise _Failed("invalid_params")  # pragma: no cover - PARAM_MODELS is exhaustive

    async def _extracted_fields(
        self,
        session: AsyncSession,
        principal: Principal,
        params: ExtractedFieldsExportParams,
        *,
        cap: int,
        now: datetime,
    ) -> Collected:
        ef = ExtractedField
        stmt = (
            select(ef, Document.title, Document.doc_type, DocumentVersion.version_number)
            .join(
                Document,
                and_(Document.id == ef.document_id, Document.organization_id == ef.organization_id),
            )
            .join(
                DocumentVersion,
                and_(
                    DocumentVersion.id == ef.version_id,
                    DocumentVersion.organization_id == ef.organization_id,
                ),
            )
            .where(readable_clause(principal, now), ef.version_id == Document.current_version_id)
            .order_by(Document.title, Document.id, ef.field, ef.confidence.desc(), ef.id)
            .limit(cap + 1)
        )
        if params.document_ids:
            stmt = stmt.where(ef.document_id.in_(params.document_ids))
        if params.fields:
            stmt = stmt.where(ef.field.in_(params.fields))
        if params.doc_type is not None:
            stmt = stmt.where(Document.doc_type == params.doc_type.value)
        if params.methods:
            stmt = stmt.where(ef.method.in_(params.methods))
        result = (await session.execute(stmt)).all()
        rows: list[tuple[object, ...]] = []
        for field, title, doc_type, version_number in result[:cap]:
            fr = field_row(field)
            rows.append(
                (
                    fr.document_id, clean_line_text(title, 300), doc_type, version_number, fr.field,
                    fr.display(), fr.value_date, fr.value_number, fr.currency,
                    round(fr.confidence, 4), fr.method, fr.page,
                    clean_line_text(fr.evidence, 500) if fr.evidence else None,
                )
            )  # fmt: skip
        return Collected(COLUMNS["extracted_fields"], rows, len(result) > cap)

    async def _search_results(
        self, principal: Principal, params: SearchResultsExportParams, cap: int
    ) -> Collected:
        """Rows from the search service, executed AS the export's creator: the service
        applies the readable clause, its own rate limit and its ``search.query`` audit."""
        search: SearchService | None = self._c.__dict__.get("search")
        if search is None:
            raise _Failed("search_unavailable")
        filters = SearchFilters(
            doc_types=tuple(t.value for t in params.doc_types),
            document_ids=tuple(params.document_ids),
            tags=tuple(params.tags),
        )
        response = await search.search(
            principal,
            query=params.query,
            mode=params.mode,
            filters=filters,
            limit=min(params.limit, cap),
        )
        rows: list[tuple[object, ...]] = [
            (
                rank,
                item.document_id,
                clean_line_text(item.document_title, 300),
                item.version_number,
                item.page_start,
                item.page_end,
                clean_line_text(item.section, 300) if item.section else None,
                round(float(item.score), 4),
                clean_line_text(item.snippet, 400),
            )
            for rank, item in enumerate(response.results[:cap], start=1)
        ]
        return Collected(COLUMNS["search_results"], rows, False)


def report_rows(report: DocumentReport) -> list[tuple[object, ...]]:
    """Flatten a document report into ``(section, item, value, page, detail)`` rows."""
    doc = report.document
    rows: list[tuple[object, ...]] = [
        ("document", "id", doc.id, None, None),
        ("document", "title", doc.title, None, None),
        ("document", "doc_type", doc.doc_type, None, None),
        ("document", "classification", doc.classification, None, None),
        ("document", "version_number", doc.version_number, None, None),
        ("document", "page_count", doc.page_count, None, None),
        ("summary", "method", report.summary.method, None, None),
        ("summary", "text", report.summary.summary, None, None),
    ]
    for index, point in enumerate(report.summary.key_points, start=1):
        page = point.citations[0].page_start if point.citations else None
        rows.append(("key_point", index, point.text, page, None))
    rows.extend(("field", v.field, v.value, v.page, v.evidence) for v in report.key_fields)
    rows.extend(
        ("deadline", d.field, d.date, d.page, f"{d.days_left} days left") for d in report.deadlines
    )
    rows.extend(("risk", f.code, f.title, f.page, f.detail) for f in report.risk_flags)
    return rows
