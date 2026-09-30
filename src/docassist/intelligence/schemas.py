"""Request and result models of the intelligence area.

Inputs forbid unknown fields and carry explicit bounds; outputs are frozen value objects
that FastAPI serialises directly. Every free-text field in an output has already passed
through the output sanitiser of the producing module (no control/invisible characters,
bounded length).
"""

from __future__ import annotations

import datetime as dt
import uuid
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from docassist.core.enums import DocumentType

SummaryStyle = Literal["brief", "detailed", "executive"]
SummaryMethod = Literal["llm", "extractive"]
ExtractionKind = Literal["contract", "invoice"]
ExportKind = Literal["extracted_fields", "deadlines", "search_results", "document_report"]
ExportFormat = Literal["csv", "json"]
FieldName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
TagName = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
class Usage(_Out):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class SourceRef(_Out):
    chunk_id: uuid.UUID
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


class FieldValue(_Out):
    """One stored ``extracted_fields`` row, display-ready."""

    field: str
    value: str | None
    value_text: str | None = None
    value_date: date | None = None
    value_number: str | None = None
    currency: str | None = None
    confidence: float
    method: Literal["rules", "llm"]
    page: int | None = None
    chunk_id: uuid.UUID | None = None
    evidence: str | None = None


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
class SummarizeRequest(_In):
    document_id: uuid.UUID
    style: SummaryStyle = "brief"


class KeyPoint(_Out):
    text: str
    citations: list[SourceRef]


class SummaryResult(_Out):
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    title: str
    style: SummaryStyle
    method: SummaryMethod
    summary: str
    key_points: list[KeyPoint]
    chunks_total: int
    chunks_used: int
    truncated: bool = False
    model: str | None = None
    provider: str | None = None
    prompt_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
class CompareRequest(_In):
    """Either two versions of ``document_id`` or ``document_id`` vs ``other_document_id``.

    ``base_version``/``target_version`` are version numbers. For a document-to-document
    comparison they select the version of ``document_id`` and of ``other_document_id``
    respectively (default: the current versions).
    """

    document_id: uuid.UUID
    other_document_id: uuid.UUID | None = None
    base_version: int | None = Field(default=None, ge=1, le=1_000_000)
    target_version: int | None = Field(default=None, ge=1, le=1_000_000)
    include_change_summary: bool = False
    max_hunks: int = Field(default=200, ge=1, le=500)

    @model_validator(mode="after")
    def _distinct(self) -> CompareRequest:
        if self.other_document_id == self.document_id:
            raise ValueError("other_document_id must differ from document_id")
        return self


class VersionRef(_Out):
    document_id: uuid.UUID
    document_title: str
    version_id: uuid.UUID
    version_number: int
    classification: str


class PageRef(_Out):
    chunk_id: uuid.UUID
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


class InlineChange(_Out):
    op: Literal["equal", "insert", "delete"]
    text: str


class DiffHunk(_Out):
    id: str
    kind: Literal["added", "removed", "changed"]
    before: str | None = None
    after: str | None = None
    before_ref: PageRef | None = None
    after_ref: PageRef | None = None
    similarity: float | None = None
    inline: list[InlineChange] = Field(default_factory=list)


class DiffStats(_Out):
    paragraphs_before: int
    paragraphs_after: int
    added: int
    removed: int
    changed: int
    unchanged: int
    similarity: float


class FieldChange(_Out):
    id: str
    field: str
    change: Literal["added", "removed", "changed"]
    before: list[FieldValue]
    after: list[FieldValue]
    delta_days: int | None = None
    delta_number: str | None = None


class ChangeSummaryItem(_Out):
    text: str
    refs: list[str]


class ChangeSummary(_Out):
    summary: str
    changes: list[ChangeSummaryItem]
    model: str | None = None
    provider: str | None = None
    prompt_version: str | None = None


class ComparisonResult(_Out):
    mode: Literal["versions", "documents"]
    base: VersionRef
    target: VersionRef
    stats: DiffStats
    hunks: list[DiffHunk]
    hunks_total: int
    truncated: bool
    field_changes: list[FieldChange]
    change_summary: ChangeSummary | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


# --------------------------------------------------------------------------- #
# Deadlines
# --------------------------------------------------------------------------- #
class DeadlineItem(_Out):
    document_id: uuid.UUID
    document_title: str
    doc_type: str
    classification: str
    version_id: uuid.UUID
    field: str
    date: dt.date
    days_left: int
    value_text: str | None = None
    evidence: str | None = None
    page: int | None = None
    chunk_id: uuid.UUID | None = None
    confidence: float
    method: Literal["rules", "llm"]


class DeadlineList(_Out):
    items: list[DeadlineItem]
    today: date
    window_start: date
    window_end: date
    timezone: str
    truncated: bool


# --------------------------------------------------------------------------- #
# Structured extraction
# --------------------------------------------------------------------------- #
class ExtractRequest(_In):
    document_id: uuid.UUID
    kind: ExtractionKind | None = None
    persist: bool = True


class ExtractedValue(_Out):
    field: str
    value: str
    value_text: str | None = None
    value_date: date | None = None
    value_number: str | None = None
    currency: str | None = None
    evidence: str | None = None
    chunk_id: uuid.UUID | None = None
    page: int | None = None
    confidence: float
    method: Literal["llm", "rules"]
    value_supported: bool = True


class LineItem(_Out):
    description: str
    quantity: str | None = None
    unit_price: str | None = None
    amount: str | None = None
    evidence: str
    chunk_id: uuid.UUID
    page: int | None = None
    confidence: float


class FieldConflict(_Out):
    field: str
    llm_value: str
    rules_value: str


class ExtractionResult(_Out):
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    kind: ExtractionKind
    method: Literal["llm", "rules_only"]
    fields: list[ExtractedValue]
    line_items: list[LineItem] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    llm_values: int = 0
    dropped_unverified: int = 0
    dropped_invalid: int = 0
    persisted: bool = False
    persisted_count: int = 0
    model: str | None = None
    provider: str | None = None
    prompt_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class DocumentFields(_Out):
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    fields: list[FieldValue]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
class RiskFlag(_Out):
    code: str
    severity: Literal["info", "warning", "high"]
    title: str
    detail: str
    evidence: str | None = None
    page: int | None = None
    chunk_id: uuid.UUID | None = None


class DocumentInfo(_Out):
    id: uuid.UUID
    title: str
    doc_type: str
    classification: str
    status: str
    version_id: uuid.UUID
    version_number: int
    version_count: int
    page_count: int | None = None
    chunk_count: int
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None


class DocumentReport(_Out):
    document: DocumentInfo
    summary: SummaryResult
    key_fields: list[FieldValue]
    deadlines: list[DeadlineItem]
    risk_flags: list[RiskFlag]
    generated_at: datetime


class ReportResponse(_Out):
    report: DocumentReport
    markdown: str


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
class ExtractedFieldsExportParams(_In):
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    fields: list[FieldName] = Field(default_factory=list, max_length=50)
    doc_type: DocumentType | None = None
    methods: list[Literal["rules", "llm"]] = Field(default_factory=list, max_length=2)


class DeadlinesExportParams(_In):
    within_days: int = Field(default=90, ge=0, le=3_650)
    overdue_days: int = Field(default=0, ge=0, le=3_650)
    doc_type: DocumentType | None = None
    fields: list[FieldName] = Field(default_factory=list, max_length=20)
    timezone: str = Field(default="UTC", max_length=64)


class SearchResultsExportParams(_In):
    query: str = Field(min_length=1, max_length=1_000)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    doc_types: list[DocumentType] = Field(default_factory=list, max_length=20)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    tags: list[TagName] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=50, ge=1, le=50)


class DocumentReportExportParams(_In):
    document_id: uuid.UUID


class ExportCreateRequest(_In):
    kind: ExportKind
    format: ExportFormat = "csv"
    params: dict[str, Any] = Field(default_factory=dict)


class ExportOut(_Out):
    id: uuid.UUID
    kind: str
    format: str
    status: str
    params: dict[str, Any]
    row_count: int | None = None
    size_bytes: int | None = None
    truncated: bool = False
    error_code: str | None = None
    created_at: datetime
    ready_at: datetime | None = None
    expires_at: datetime
    download_count: int = 0
    last_downloaded_at: datetime | None = None


class ExportList(_Out):
    items: list[ExportOut]


class ExportLink(_Out):
    url: str
    expires_at: datetime
