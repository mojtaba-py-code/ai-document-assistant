"""Request/response schemas and value rules for the administration area.

Inputs are strict (``extra="forbid"``): an unexpected field is a client bug or a probe, never
silently ignored. Every free-text field is cleaned with :func:`clean_line_text` (no
invisible/control characters, collapsed whitespace, length-capped) before it can reach the
database or an audit record.

``OrgSettings`` is the typed view of ``organizations.settings``. Tenants may only make the
deployment's AI data policy *stricter* (a lower external-classification ceiling, a smaller
token budget), only lower the export row cap, and only keep governance records *longer*
than the deployment minimums - the field bounds and :func:`validate_org_settings` enforce
this whenever settings change, and the ``effective_*`` helpers clamp stored values again at
read time, so lowering a deployment ceiling later still wins over an older, higher tenant
value. Other areas read the stored JSON directly: ``llm.monthly_token_budget`` and
``llm.external_max_classification`` (LLM budget guard and data policy),
``exports.max_rows`` (export service).
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import unicodedata
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from docassist.core.enums import (
    AuditOutcome,
    Classification,
    JobStatus,
    OrganizationStatus,
    Role,
    UserStatus,
)
from docassist.core.text import clean_line_text

if TYPE_CHECKING:
    from docassist.core.config import Settings

SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_EMAIL_LOCAL_RE = re.compile(r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*$")
_EMAIL_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ACTION_FILTER_RE = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*(?:\.\*)?$")
_RESOURCE_TYPE_RE = re.compile(r"^[a-z0-9_]{1,32}$")
_JOB_KIND_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")

MAX_NAME_LENGTH = 200
MAX_DEPARTMENT_NAME = 120
MAX_DESCRIPTION = 500
MAX_DEPARTMENTS_PER_USER = 50
MAX_TOKEN_BUDGET = 10**12
EXPORT_ROW_CAP = 10_000
"""Platform hard cap on exported rows (``intelligence.exports.EXPORT_MAX_ROWS``)."""
_SECTIONS = ("llm", "retention", "exports")


# --------------------------------------------------------------------------- #
# Value rules (also used by the CLI and the demo seeder)
# --------------------------------------------------------------------------- #
def normalize_slug(value: str) -> str:
    """Validate an organisation/department slug (the database enforces the same pattern)."""
    slug = value.strip().lower()
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(
            "slug must be 1-63 characters of a-z, 0-9 and '-', starting and ending with a "
            "letter or digit"
        )
    return slug


def slugify(name: str) -> str:
    """Derive a slug from a display name (ASCII-folded, hyphen-separated, max 63 chars)."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:63].strip("-")
    return slug or "department"


def normalize_email(value: str) -> str:
    """Lower-case and validate an email address (ASCII only - no homoglyph look-alikes)."""
    email = value.strip().lower()
    if len(email) > 320 or email.count("@") != 1:
        raise ValueError("invalid email address")
    local, domain = email.split("@")
    labels = domain.split(".")
    if (
        not 1 <= len(local) <= 64
        or not _EMAIL_LOCAL_RE.fullmatch(local)
        or len(labels) < 2
        or not all(_EMAIL_LABEL_RE.fullmatch(label) for label in labels)
        or labels[-1].isdigit()
    ):
        raise ValueError("invalid email address")
    return email


def clean_name(value: str, max_length: int = MAX_NAME_LENGTH) -> str:
    """Single-line display name: invisible characters removed, whitespace collapsed."""
    cleaned = clean_line_text(value, max_length)
    if not cleaned:
        raise ValueError("must not be empty")
    return cleaned


def escape_like(value: str) -> str:
    r"""Escape ``%``, ``_`` and ``\`` so user input is matched literally by (I)LIKE."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def parse_month(value: str) -> tuple[int, int]:
    """Parse ``YYYY-MM`` (years 2000-2999)."""
    match = _MONTH_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("month must look like YYYY-MM")
    year, month = int(match.group(1)), int(match.group(2))
    if not 2000 <= year <= 2999 or not 1 <= month <= 12:
        raise ValueError("month is out of range")
    return year, month


def validate_action_filter(value: str) -> str:
    """Audit action filter: an exact action (``admin.user_created``) or a prefix (``admin.*``)."""
    action = value.strip()
    if len(action) > 64 or not _ACTION_FILTER_RE.fullmatch(action):
        raise ValueError("action filter must look like 'area.verb' or 'area.*'")
    return action


def validate_resource_type(value: str) -> str:
    if not _RESOURCE_TYPE_RE.fullmatch(value):
        raise ValueError("invalid resource type")
    return value


def validate_job_kind(value: str) -> str:
    if not _JOB_KIND_RE.fullmatch(value):
        raise ValueError("invalid job kind")
    return value


# --------------------------------------------------------------------------- #
# Opaque keyset cursors
# --------------------------------------------------------------------------- #
class CursorError(ValueError):
    """The cursor is malformed (it is client-supplied and therefore untrusted)."""


def encode_cursor(values: dict[str, str | int]) -> str:
    raw = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str, keys: tuple[str, ...]) -> dict[str, Any]:
    """Decode a cursor produced by :func:`encode_cursor`; exactly ``keys`` must be present."""
    if not cursor or len(cursor) > 512:
        raise CursorError("invalid cursor")
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise CursorError("invalid cursor") from exc
    if not isinstance(data, dict) or set(data) != set(keys):
        raise CursorError("invalid cursor")
    return data


def time_id_cursor(created_at: datetime, row_id: uuid.UUID) -> str:
    return encode_cursor({"t": created_at.isoformat(), "id": str(row_id)})


def parse_time_id_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    data = decode_cursor(cursor, ("t", "id"))
    try:
        moment = datetime.fromisoformat(str(data["t"]))
        row_id = uuid.UUID(str(data["id"]))
    except ValueError as exc:
        raise CursorError("invalid cursor") from exc
    if moment.tzinfo is None:
        raise CursorError("invalid cursor")
    return moment, row_id


def int_cursor(value: int) -> str:
    return encode_cursor({"id": value})


def parse_int_cursor(cursor: str) -> int:
    data = decode_cursor(cursor, ("id",))
    value = data["id"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CursorError("invalid cursor")
    return value


# --------------------------------------------------------------------------- #
# Base models
# --------------------------------------------------------------------------- #
class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _Output(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Organisation settings
# --------------------------------------------------------------------------- #
class OrgLlmSettings(_Input):
    """Per-organisation AI data policy. ``None`` = inherit the deployment value."""

    external_max_classification: Classification | None = None
    monthly_token_budget: int | None = Field(default=None, ge=1, le=MAX_TOKEN_BUDGET)


class OrgRetentionSettings(_Input):
    """Retention overrides in days. ``None`` = inherit the deployment value.

    ``audit_days`` and ``llm_usage_days`` are governance records: a tenant may keep them
    longer than the deployment, never shorter (checked in :func:`validate_org_settings`).
    """

    conversation_days: int | None = Field(default=None, ge=1, le=3_650)
    job_days: int | None = Field(default=None, ge=1, le=3_650)
    deleted_document_purge_days: int | None = Field(default=None, ge=0, le=3_650)
    audit_days: int | None = Field(default=None, ge=30, le=36_500)
    llm_usage_days: int | None = Field(default=None, ge=30, le=3_650)


class OrgExportSettings(_Input):
    """Export limits. ``max_rows`` may only lower the platform cap."""

    max_rows: int | None = Field(default=None, ge=1, le=EXPORT_ROW_CAP)


class OrgSettings(_Input):
    """Typed view of ``organizations.settings`` (stored with ``None`` values omitted)."""

    llm: OrgLlmSettings = Field(default_factory=OrgLlmSettings)
    retention: OrgRetentionSettings = Field(default_factory=OrgRetentionSettings)
    exports: OrgExportSettings = Field(default_factory=OrgExportSettings)

    @classmethod
    def from_stored(cls, raw: dict[str, Any] | None) -> OrgSettings:
        """Parse the JSONB column leniently: unknown keys are ignored, invalid data -> defaults."""
        raw = raw if isinstance(raw, dict) else {}
        sections: dict[str, Any] = {}
        models: tuple[tuple[str, type[BaseModel]], ...] = (
            ("llm", OrgLlmSettings),
            ("retention", OrgRetentionSettings),
            ("exports", OrgExportSettings),
        )
        for name, model in models:
            section = raw.get(name)
            if not isinstance(section, dict):
                continue
            known = {k: v for k, v in section.items() if k in model.model_fields}
            try:
                sections[name] = model.model_validate(known)
            except ValidationError:
                sections[name] = model()
        return cls.model_validate(sections)

    def to_stored(self, base: dict[str, Any] | None = None) -> dict[str, Any]:
        """Serialise for the JSONB column, preserving unrelated top-level keys of ``base``."""
        out = {k: v for k, v in (base or {}).items() if k not in _SECTIONS}
        for name in _SECTIONS:
            values = getattr(self, name).model_dump(mode="json", exclude_none=True)
            if values:
                out[name] = values
        return out


class OrgSettingsPatch(_Input):
    """Partial update: only fields present in the request change; ``null`` clears an override."""

    llm: OrgLlmSettings | None = None
    retention: OrgRetentionSettings | None = None
    exports: OrgExportSettings | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> OrgSettingsPatch:
        if self.llm is None and self.retention is None and self.exports is None:
            raise ValueError("nothing to update")
        return self

    def changed_fields(self) -> dict[str, dict[str, Any]]:
        """``{"llm": {"field": value, ...}, ...}`` for the fields explicitly sent."""
        out: dict[str, dict[str, Any]] = {}
        for name in _SECTIONS:
            section: BaseModel | None = getattr(self, name)
            if section is None:
                continue
            fields = {key: getattr(section, key) for key in section.model_fields_set}
            if fields:
                out[name] = fields
        return out


def validate_org_settings(changes: dict[str, dict[str, Any]], settings: Settings) -> list[str]:
    """Return human-readable problems if ``changes`` would relax a deployment guarantee."""
    problems: list[str] = []
    llm = changes.get("llm", {})
    ceiling = llm.get("external_max_classification")
    deployment_ceiling = settings.llm.external_max_classification
    if ceiling is not None and Classification(ceiling).rank > deployment_ceiling.rank:
        problems.append(
            "llm.external_max_classification may not exceed the deployment ceiling "
            f"({deployment_ceiling.value})"
        )
    budget = llm.get("monthly_token_budget")
    deployment_budget = settings.llm.monthly_token_budget_per_org
    if budget is not None and deployment_budget > 0 and budget > deployment_budget:
        problems.append(
            f"llm.monthly_token_budget may not exceed the deployment budget ({deployment_budget})"
        )
    retention = changes.get("retention", {})
    for field in ("audit_days", "llm_usage_days"):
        value = retention.get(field)
        minimum = int(getattr(settings.retention, field))
        if value is not None and value < minimum:
            problems.append(f"retention.{field} may not be shorter than the deployment ({minimum})")
    return problems


def effective_external_ceiling(settings: Settings, org: OrgSettings) -> Classification:
    """The stricter of the deployment and organisation external-classification ceilings."""
    deployment = settings.llm.external_max_classification
    own = org.llm.external_max_classification
    if own is None or own.rank >= deployment.rank:
        return deployment
    return own


def effective_token_budget(settings: Settings, org: OrgSettings) -> int | None:
    """Monthly token budget for the organisation; ``None`` means unlimited."""
    deployment = settings.llm.monthly_token_budget_per_org or None
    own = org.llm.monthly_token_budget
    if own is None:
        return deployment
    return own if deployment is None else min(own, deployment)


def effective_export_rows(org: OrgSettings) -> int:
    """Maximum rows per export for the organisation (never above the platform cap)."""
    own = org.exports.max_rows
    return EXPORT_ROW_CAP if own is None else min(own, EXPORT_ROW_CAP)


def effective_retention(settings: Settings, org: OrgSettings) -> dict[str, int]:
    """Retention periods (days) after applying tenant overrides and governance minimums."""
    base = settings.retention
    out: dict[str, int] = {}
    for field in (
        "conversation_days",
        "job_days",
        "deleted_document_purge_days",
        "audit_days",
        "llm_usage_days",
    ):
        deployment = int(getattr(base, field))
        own = getattr(org.retention, field)
        if own is None:
            out[field] = deployment
        elif field in {"audit_days", "llm_usage_days"}:
            out[field] = max(own, deployment)
        else:
            out[field] = int(own)
    return out


# --------------------------------------------------------------------------- #
# Organisations
# --------------------------------------------------------------------------- #
class OrganizationCreate(_Input):
    slug: str = Field(min_length=1, max_length=63)
    name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    admin_email: str = Field(min_length=3, max_length=320)
    admin_name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)

    @field_validator("slug")
    @classmethod
    def _check_slug(cls, value: str) -> str:
        return normalize_slug(value)

    @field_validator("admin_email")
    @classmethod
    def _check_email(cls, value: str) -> str:
        return normalize_email(value)

    @field_validator("name", "admin_name")
    @classmethod
    def _check_names(cls, value: str) -> str:
        return clean_name(value)


class OrganizationUpdate(_Input):
    name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_LENGTH)
    status: OrganizationStatus | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        return None if value is None else clean_name(value)

    @model_validator(mode="after")
    def _not_empty(self) -> OrganizationUpdate:
        if self.name is None and self.status is None:
            raise ValueError("nothing to update")
        return self


class OrganizationOut(_Output):
    id: uuid.UUID
    slug: str
    name: str
    status: OrganizationStatus
    created_at: datetime
    updated_at: datetime


class OrganizationPage(BaseModel):
    items: list[OrganizationOut]
    next_cursor: str | None = None


class OrganizationCreated(BaseModel):
    organization: OrganizationOut
    admin_user_id: uuid.UUID
    invitation_sent: bool


class EffectiveOrgPolicy(BaseModel):
    external_max_classification: Classification
    monthly_token_budget: int | None
    retention: dict[str, int]
    export_max_rows: int


class DeploymentLimits(BaseModel):
    external_max_classification: Classification
    monthly_token_budget: int | None
    retention: dict[str, int]
    export_max_rows: int


class OrganizationSettingsOut(BaseModel):
    organization: OrganizationOut
    settings: OrgSettings
    effective: EffectiveOrgPolicy
    deployment: DeploymentLimits


# --------------------------------------------------------------------------- #
# Departments
# --------------------------------------------------------------------------- #
def _department_name(value: str | None) -> str | None:
    return None if value is None else clean_name(value, MAX_DEPARTMENT_NAME)


def _optional_slug(value: str | None) -> str | None:
    return None if value is None else normalize_slug(value)


def _description(value: str | None) -> str | None:
    if value is None:
        return None
    return clean_line_text(value, MAX_DESCRIPTION) or None


class DepartmentCreate(_Input):
    name: str = Field(min_length=1, max_length=MAX_DEPARTMENT_NAME)
    slug: str | None = Field(default=None, min_length=1, max_length=63)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        return clean_name(value, MAX_DEPARTMENT_NAME)

    @field_validator("slug")
    @classmethod
    def _check_slug(cls, value: str | None) -> str | None:
        return _optional_slug(value)

    @field_validator("description")
    @classmethod
    def _check_description(cls, value: str | None) -> str | None:
        return _description(value)


class DepartmentUpdate(_Input):
    """Partial update; ``description: null`` clears the description."""

    name: str | None = Field(default=None, min_length=1, max_length=MAX_DEPARTMENT_NAME)
    slug: str | None = Field(default=None, min_length=1, max_length=63)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str | None) -> str | None:
        return _department_name(value)

    @field_validator("slug")
    @classmethod
    def _check_slug(cls, value: str | None) -> str | None:
        return _optional_slug(value)

    @field_validator("description")
    @classmethod
    def _check_description(cls, value: str | None) -> str | None:
        return _description(value)

    @model_validator(mode="after")
    def _not_empty(self) -> DepartmentUpdate:
        if not self.model_fields_set:
            raise ValueError("nothing to update")
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name cannot be null")
        if "slug" in self.model_fields_set and self.slug is None:
            raise ValueError("slug cannot be null")
        return self


class DepartmentOut(_Output):
    id: uuid.UUID
    name: str
    slug: str
    description: str | None
    created_at: datetime
    member_count: int = 0
    manager_count: int = 0


class DepartmentList(BaseModel):
    items: list[DepartmentOut]


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
class DepartmentMembershipIn(_Input):
    department_id: uuid.UUID
    is_manager: bool = False


def _unique_departments(value: list[DepartmentMembershipIn]) -> list[DepartmentMembershipIn]:
    ids = [m.department_id for m in value]
    if len(ids) != len(set(ids)):
        raise ValueError("each department may appear only once")
    return value


class UserCreate(_Input):
    email: str = Field(min_length=3, max_length=320)
    full_name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    role: Role
    clearance: Classification | None = None
    departments: list[DepartmentMembershipIn] = Field(
        default_factory=list, max_length=MAX_DEPARTMENTS_PER_USER
    )

    @field_validator("email")
    @classmethod
    def _check_email(cls, value: str) -> str:
        return normalize_email(value)

    @field_validator("departments")
    @classmethod
    def _check_departments(
        cls, value: list[DepartmentMembershipIn]
    ) -> list[DepartmentMembershipIn]:
        return _unique_departments(value)

    @field_validator("full_name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        return clean_name(value)


class UserUpdate(_Input):
    """Partial update. ``departments`` replaces the whole membership list when present."""

    full_name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_LENGTH)
    role: Role | None = None
    clearance: Classification | None = None
    status: UserStatus | None = None
    departments: list[DepartmentMembershipIn] | None = Field(
        default=None, max_length=MAX_DEPARTMENTS_PER_USER
    )

    @field_validator("full_name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        return None if value is None else clean_name(value)

    @field_validator("departments")
    @classmethod
    def _departments(
        cls, value: list[DepartmentMembershipIn] | None
    ) -> list[DepartmentMembershipIn] | None:
        return None if value is None else _unique_departments(value)

    @model_validator(mode="after")
    def _not_empty(self) -> UserUpdate:
        if all(
            getattr(self, name) is None
            for name in ("full_name", "role", "clearance", "status", "departments")
        ):
            raise ValueError("nothing to update")
        return self


class UserMembershipOut(BaseModel):
    department_id: uuid.UUID
    department_name: str
    is_manager: bool


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: Role
    clearance: Classification
    status: UserStatus
    mfa_enabled: bool
    locked: bool
    last_login_at: datetime | None
    created_at: datetime
    updated_at: datetime
    departments: list[UserMembershipOut]


class UserPage(BaseModel):
    items: list[UserOut]
    next_cursor: str | None = None


class SessionsRevoked(BaseModel):
    revoked_sessions: int


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
class AuditEventOut(BaseModel):
    id: int
    occurred_at: datetime
    actor_user_id: uuid.UUID | None
    actor_role: str | None
    actor_ip_prefix: str | None
    action: str
    resource_type: str | None
    resource_id: str | None
    outcome: AuditOutcome
    request_id: str | None
    details: dict[str, Any]
    sealed: bool
    seal_seq: int | None


class AuditEventPage(BaseModel):
    items: list[AuditEventOut]
    next_cursor: str | None = None


class AuditVerifyOut(BaseModel):
    chain: Literal["organization", "platform"]
    organization_id: uuid.UUID | None
    valid: bool
    checked: int
    head_seq: int
    unsealed: int
    first_bad_seq: int | None
    reason: str | None
    verified_at: datetime


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
class JobOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: JobStatus
    attempts: int
    max_attempts: int
    error_code: str | None
    resource_ids: dict[str, uuid.UUID]
    created_at: datetime
    updated_at: datetime
    run_after: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobPage(BaseModel):
    items: list[JobOut]
    next_cursor: str | None = None


# --------------------------------------------------------------------------- #
# Usage & health
# --------------------------------------------------------------------------- #
class UsageRow(BaseModel):
    model: str
    task: str
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


class UsageTotals(BaseModel):
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: Decimal


class UsageBudget(BaseModel):
    deployment_limit: int | None
    organization_limit: int | None
    effective_limit: int | None
    used_tokens: int
    remaining_tokens: int | None
    exhausted: bool


class UsageReport(BaseModel):
    month: str
    period_start: datetime
    period_end: datetime
    rows: list[UsageRow]
    totals: UsageTotals
    budget: UsageBudget


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "unavailable", "not_configured", "unknown"]
    detail: dict[str, Any] = Field(default_factory=dict)


class QueueHealth(BaseModel):
    scope: Literal["organization", "platform"]
    depth: dict[str, int]


class AdminHealth(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    version: str
    checked_at: datetime
    components: dict[str, ComponentHealth]
    queue: QueueHealth | None
