"""Ingestion pipeline: stored encrypted blob -> searchable, analysed, indexed version.

``process_version(org_id, version_id)`` (idempotent) runs on the *worker* database role with
``DbContext.system_for_org(org_id)``:

1. **claim** (short transaction, row lock): skip versions that are already indexed,
   quarantined, missing or whose document was deleted; mark the version ``processing``;
2. **decrypt** the blob (bounded by ``upload.max_upload_bytes``) and **parse it in the
   sandbox** (:mod:`docassist.ingestion.sandbox`); OCR scanned pages when enabled;
3. **analyse** off the event loop: Unicode hygiene, structure-aware chunking, a
   prompt-injection score per chunk, PII kinds per chunk, sensitivity suggestion, document
   type, rules-based fields, language;
4. **embed** the chunks - *unless* the provider is external and the document's effective
   classification (the higher of its classification, a pending suggestion and what the
   content itself suggests) is above ``llm.external_max_classification``: then the
   version is keyword-only (``semantic_indexed = false``) and no text leaves the network
   (re-checked inside the commit transaction: if the classification was raised meanwhile,
   the vectors are discarded);
5. **commit** everything in ONE transaction: delete partial rows of this version, insert
   chunks, embeddings and fields, update the version, switch ``current_version_id`` when
   this version is at least as new as the current one, mark the document ready, apply the
   automatic document type (never over a user's choice), record a sensitivity escalation
   suggestion (never lowering, never auto-applied), audit, and enqueue a vector sync when
   the Qdrant backend is configured.

Failures are recorded on the version (and on the document while it has no ready version)
with a sanitised ``error_code``; parse-type failures raise :class:`PermanentJobError`,
embedding-provider failures raise a retryable :class:`JobError`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor
from docassist.core.context import utcnow
from docassist.core.enums import (
    AuditOutcome,
    Classification,
    DocumentStatus,
    VersionStatus,
)
from docassist.core.ids import uuid7
from docassist.core.logging import get_logger
from docassist.core.redaction import PERSONAL_KINDS, redact
from docassist.core.text import clean_line_text
from docassist.db.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentVersion,
    ExtractedField,
    Organization,
)
from docassist.db.session import Database, DbContext
from docassist.documents.storage import StorageError, storage_context
from docassist.embeddings.base import EmbeddingError, validate_vectors
from docassist.ingestion.chunking import Chunk, ChunkedDocument, Chunker, ChunkingParams
from docassist.ingestion.classify import Classification as TypeGuess
from docassist.ingestion.classify import classify_document
from docassist.ingestion.extraction_rules import FieldCandidate, extract_fields
from docassist.ingestion.injection import InjectionReport, scan_for_injection
from docassist.ingestion.language import guess_language
from docassist.ingestion.model import FORMATS, Block, ParsedDocument, ParsedPage
from docassist.ingestion.normalize import NormalizedDocument, normalize_document
from docassist.ingestion.ocr import OcrConfig, OcrEngine
from docassist.ingestion.parsers.base import split_at_whitespace
from docassist.ingestion.sandbox import ParserSandbox, SandboxConfig, SandboxError
from docassist.ingestion.sensitivity import SensitivityAccumulator, SensitivityResult
from docassist.jobs.queue import JobError, PermanentJobError, enqueue
from docassist.observability import metrics
from docassist.security.crypto import DecryptionError

if TYPE_CHECKING:
    from docassist.api.container import Container
    from docassist.core.config import Settings
    from docassist.documents.storage import ObjectStorage
    from docassist.embeddings.base import EmbeddingProvider

log = get_logger(__name__)

MAX_METADATA_JSON_BYTES = 16_384
ERROR_DETAILS: dict[str, str] = {
    "parse_error": "The document could not be parsed.",
    "parse_timeout": "Parsing exceeded the time limit.",
    "unsupported": "This document format or variant is not supported.",
    "empty_document": "No extractable text was found in the document.",
    "too_large": "The document exceeds the processing limits.",
    "blob_missing": "The stored file could not be found.",
    "integrity_error": "The stored file failed its integrity check.",
    "embedding_unavailable": "The embedding service was unavailable; retries were exhausted.",
    "storage_unavailable": "Document storage was unavailable; retries were exhausted.",
    "job_timeout": "Processing exceeded the job time limit; retries were exhausted.",
    "lease_expired": "The worker processing the document stopped; retries were exhausted.",
    "internal_error": "Processing failed unexpectedly; retries were exhausted.",
}


class IngestionFailure(PermanentJobError):
    """A permanent, recorded ingestion failure with a sanitised ``code``."""

    def __init__(self, code: str) -> None:
        super().__init__(f"ingestion failed ({code})", code=code)


@dataclass(frozen=True, slots=True)
class _Claimed:
    org_id: uuid.UUID
    version_id: uuid.UUID
    document_id: uuid.UUID
    version_number: int
    extension: str
    storage_key: str
    title: str


@dataclass(frozen=True, slots=True)
class _ChunkRow:
    chunk_id: uuid.UUID
    index: int
    report: InjectionReport
    pii: list[str]


@dataclass(slots=True)
class _Analysis:
    parsed: ParsedDocument
    normalized: NormalizedDocument
    chunked: ChunkedDocument
    rows: list[_ChunkRow]
    sensitivity: SensitivityResult
    doc_type: TypeGuess
    fields: list[FieldCandidate]
    language: str | None
    ocr: dict[str, Any] = field(default_factory=dict)


def _skipped(reason: str, version_id: uuid.UUID) -> dict[str, Any]:
    metrics.INGESTION.labels(outcome="skipped").inc()
    return {"status": "skipped", "reason": reason, "version_id": str(version_id)}


def _clean_meta(value: str | None, limit: int = 300) -> str | None:
    if not value:
        return None
    return clean_line_text(value, limit) or None


class IngestionPipeline:
    def __init__(
        self,
        container: Container,
        *,
        settings: Settings | None = None,
        sandbox: ParserSandbox | None = None,
        ocr: OcrEngine | None = None,
        storage: ObjectStorage | None = None,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        self._container = container
        self.settings = settings or container.settings
        self.sandbox = sandbox or ParserSandbox(SandboxConfig.from_settings(self.settings))
        self.ocr = ocr or OcrEngine(OcrConfig.from_settings(self.settings.parser))
        self._storage = storage
        self._embeddings = embeddings
        self.params = ChunkingParams.from_settings(self.settings.chunking)

    # ------------------------------------------------------------------ collaborators
    @property
    def storage(self) -> ObjectStorage:
        return self._storage or self._container.storage

    @property
    def embeddings(self) -> EmbeddingProvider:
        return self._embeddings or self._container.embeddings

    @property
    def db(self) -> Database:
        if self._container.worker_db is None:
            raise RuntimeError("the ingestion pipeline runs on the worker database role")
        return self._container.worker_db

    # ------------------------------------------------------------------ entry point
    async def process_version(self, org_id: uuid.UUID, version_id: uuid.UUID) -> dict[str, Any]:
        """Parse, analyse, embed and index one document version (see module docstring)."""
        started = time.perf_counter()
        claimed = await self._claim(org_id, version_id)
        if isinstance(claimed, dict):
            return claimed
        needs_ocr = False
        try:
            if claimed.extension not in FORMATS:
                raise IngestionFailure("unsupported")
            data = await self._read_blob(claimed)
            parsed = await self.sandbox.parse(
                claimed.extension, data, ocr_images=self.ocr.available
            )
            del data
            needs_ocr = parsed.needs_ocr
            parsed, ocr_info = await self._apply_ocr(parsed)
            needs_ocr = parsed.needs_ocr
            analysis = await asyncio.to_thread(self._analyse, claimed, parsed, ocr_info)
        except PermanentJobError as exc:
            await self._record_failure(
                claimed.org_id, claimed.version_id, exc.code, needs_ocr=needs_ocr
            )
            if isinstance(exc, SandboxError | IngestionFailure):
                raise
            raise IngestionFailure(exc.code) from exc
        vectors, skip_reason = await self._embed(claimed, analysis)
        return await self._commit(claimed, analysis, vectors, skip_reason, started)

    async def mark_failed(
        self, org_id: uuid.UUID, version_id: uuid.UUID, code: str, *, only_if_pending: bool = False
    ) -> bool:
        """Record a final failure (e.g. retries exhausted) without re-processing.

        With ``only_if_pending`` only a version still ``uploaded``/``processing`` is touched
        (used by the maintenance sweeper). Returns whether the version was marked failed.
        """
        return await self._record_failure(
            org_id,
            version_id,
            code if code in ERROR_DETAILS else "internal_error",
            only_if_pending=only_if_pending,
        )

    # ------------------------------------------------------------------ step 1: claim
    async def _lock(
        self, session: AsyncSession, org_id: uuid.UUID, version_id: uuid.UUID
    ) -> tuple[DocumentVersion, Document] | None:
        row = (
            await session.execute(
                select(DocumentVersion, Document)
                .join(
                    Document,
                    and_(
                        Document.id == DocumentVersion.document_id,
                        Document.organization_id == DocumentVersion.organization_id,
                    ),
                )
                .where(DocumentVersion.id == version_id, DocumentVersion.organization_id == org_id)
                .with_for_update()
            )
        ).first()
        return None if row is None else (row[0], row[1])

    async def _claim(self, org_id: uuid.UUID, version_id: uuid.UUID) -> _Claimed | dict[str, Any]:
        async with self.db.transaction(DbContext.system_for_org(org_id)) as session:
            locked = await self._lock(session, org_id, version_id)
            if locked is None:
                return _skipped("not_found", version_id)
            version, document = locked
            if version.status == VersionStatus.INDEXED.value:
                return _skipped("already_indexed", version_id)
            if version.status == VersionStatus.QUARANTINED.value:
                return _skipped("quarantined", version_id)
            if document.status == DocumentStatus.DELETED.value:
                return _skipped("document_deleted", version_id)
            version.status = VersionStatus.PROCESSING.value
            version.error_code = None
            version.error_detail = None
            return _Claimed(
                org_id=org_id,
                version_id=version_id,
                document_id=document.id,
                version_number=version.version_number,
                extension=version.extension.lower(),
                storage_key=version.storage_key,
                title=document.title,
            )

    # ------------------------------------------------------------------ step 2: read + parse
    async def _read_blob(self, claimed: _Claimed) -> bytes:
        context = storage_context(claimed.org_id, "version", claimed.version_id)
        try:
            return await self.storage.read_bytes(
                claimed.storage_key, context, max_bytes=self.settings.upload.max_upload_bytes
            )
        except StorageError as exc:
            reason = str(exc)
            if "larger" in reason:
                code = "too_large"
            elif "not found" in reason:
                code = "blob_missing"
            else:
                code = "integrity_error"  # a malformed or escaping key on the version row
            raise IngestionFailure(code) from exc
        except DecryptionError as exc:
            raise IngestionFailure("integrity_error") from exc
        except OSError as exc:
            raise JobError("document storage unavailable", code="storage_unavailable") from exc

    async def _apply_ocr(self, parsed: ParsedDocument) -> tuple[ParsedDocument, dict[str, Any]]:
        pending = [page.number for page in parsed.pages if page.needs_ocr]
        info: dict[str, Any] = {"needed": bool(pending), "applied": False, "pages": len(pending)}
        if not pending:
            return parsed, info
        if not self.ocr.available:
            return _with_warning(parsed, "ocr_unavailable"), info
        by_page: dict[int, list[bytes]] = {}
        for image in parsed.ocr_images:
            by_page.setdefault(image.page, []).append(image.data)
        texts: dict[int, str] = {}
        failures = 0
        for page_number, images in by_page.items():
            recognised, failed = await self.ocr.recognize_all(images)
            failures += failed
            text = "\n\n".join(t.strip() for t in recognised if t.strip())
            if text:
                texts[page_number] = text
        info.update({"applied": bool(texts), "recognised_pages": len(texts), "failures": failures})
        merged = _merge_ocr(parsed, texts, self.sandbox.config.limits.max_block_chars)
        if failures:
            merged = _with_warning(merged, "ocr_failed")
        return merged, info

    # ------------------------------------------------------------------ step 3: analyse (thread)
    def _analyse(
        self, claimed: _Claimed, parsed: ParsedDocument, ocr_info: dict[str, Any]
    ) -> _Analysis:
        normalized = normalize_document(parsed)
        if not normalized.blocks:
            raise IngestionFailure("empty_document")
        chunked = Chunker(self.params).chunk(normalized.blocks)
        if not chunked.chunks:
            raise IngestionFailure("empty_document")
        accumulator = SensitivityAccumulator()
        rows = []
        for chunk in chunked.chunks:
            pii = accumulator.add(chunk.text)
            report = scan_for_injection(
                chunk.text, chunk.hidden_text, channel_flags=chunk.channel_flags
            )
            rows.append(_ChunkRow(uuid7(), chunk.index, report, pii))
        title = " ".join(filter(None, [claimed.title, parsed.metadata.title or ""]))
        return _Analysis(
            parsed=parsed,
            normalized=normalized,
            chunked=chunked,
            rows=rows,
            sensitivity=accumulator.result(),
            doc_type=classify_document(chunked.text, title=title),
            fields=extract_fields(chunked.text),
            language=guess_language(chunked.text),
            ocr=ocr_info,
        )

    # ------------------------------------------------------------------ step 4: embed
    async def _effective_classification(
        self, claimed: _Claimed, analysis: _Analysis
    ) -> Classification:
        async with self.db.session(DbContext.system_for_org(claimed.org_id)) as session:
            row = (
                await session.execute(
                    select(Document.classification, Document.suggested_classification).where(
                        Document.id == claimed.document_id,
                        Document.organization_id == claimed.org_id,
                    )
                )
            ).first()
        levels = [Classification.RESTRICTED]  # unknown document: assume the worst
        if row is not None:
            levels = [Classification(row[0])]
            if row[1]:
                levels.append(Classification(row[1]))
        if analysis.sensitivity.suggested is not None:
            levels.append(analysis.sensitivity.suggested)
        return Classification.highest(levels)

    async def _embed(
        self, claimed: _Claimed, analysis: _Analysis
    ) -> tuple[list[list[float]] | None, str | None]:
        provider = self.embeddings
        effective = await self._effective_classification(claimed, analysis)
        ceiling = await self._external_ceiling(claimed.org_id)
        if provider.is_external and effective.rank > ceiling.rank:
            log.info(
                "embedding_skipped_by_policy",
                version_id=str(claimed.version_id),
                classification=effective.value,
            )
            return None, "classification_above_external_ceiling"
        texts = [chunk.text for chunk in analysis.chunked.chunks]
        if provider.is_external:
            # Data minimisation: personal data is not needed to embed meaning, so it never
            # leaves the network (stored chunk text is unchanged).
            texts = [redact(text, PERSONAL_KINDS) for text in texts]
        vectors: list[list[float]] = []
        batch = self.settings.embedding.batch_size
        for start in range(0, len(texts), batch):
            part = texts[start : start + batch]
            try:
                out = await provider.embed(part, kind="document")
                vectors.extend(validate_vectors(out, len(part), provider.dimensions))
            except EmbeddingError as exc:
                raise JobError(
                    "embedding provider unavailable", code="embedding_unavailable"
                ) from exc
        return vectors, None

    # ------------------------------------------------------------------ step 5: commit
    async def _commit(
        self,
        claimed: _Claimed,
        analysis: _Analysis,
        vectors: list[list[float]] | None,
        skip_reason: str | None,
        started: float,
    ) -> dict[str, Any]:
        settings = self.settings
        chunked = analysis.chunked
        warn = settings.retrieval.injection_warn_threshold
        flagged_warn = sum(1 for row in analysis.rows if row.report.score >= warn)
        async with self.db.transaction(DbContext.system_for_org(claimed.org_id)) as session:
            locked = await self._lock(session, claimed.org_id, claimed.version_id)
            if locked is None:
                return _skipped("not_found", claimed.version_id)
            version, document = locked
            if version.status == VersionStatus.INDEXED.value:
                return _skipped("already_indexed", claimed.version_id)
            if document.status == DocumentStatus.DELETED.value:
                return _skipped("document_deleted", claimed.version_id)
            ceiling = await self._external_ceiling(claimed.org_id, session)
            if vectors is not None and self._above_external_ceiling(document, analysis, ceiling):
                # The classification was raised while we embedded: never store the vectors.
                vectors, skip_reason = None, "classification_above_external_ceiling"

            await self._replace_rows(session, claimed, analysis, vectors)
            becomes_current = await self._is_newest(session, document, version)
            now = utcnow()
            version.status = VersionStatus.INDEXED.value
            version.page_count = len(analysis.parsed.pages)
            version.char_count = analysis.normalized.char_count
            version.chunk_count = len(chunked.chunks)
            version.language = analysis.language
            version.needs_ocr = analysis.parsed.needs_ocr
            version.semantic_indexed = vectors is not None
            version.processed_at = now
            version.error_code = None
            version.error_detail = None
            version.doc_metadata = self._metadata(analysis, skip_reason)
            version.security_findings = self._findings(
                list(version.security_findings or []), analysis, flagged_warn
            )

            if becomes_current:
                document.current_version_id = version.id
                document.status = DocumentStatus.READY.value
                document.sensitivity_signals = list(analysis.sensitivity.signals)
                if document.doc_type_source != "user":
                    document.doc_type = analysis.doc_type.doc_type.value
                    document.doc_type_confidence = analysis.doc_type.confidence
                    document.doc_type_source = "auto"
            elif document.status in (DocumentStatus.PROCESSING.value, DocumentStatus.FAILED.value):
                document.status = DocumentStatus.READY.value
            escalation = self._escalation(document, analysis.sensitivity)
            actor = Actor.system(claimed.org_id)
            if escalation is not None:
                previous = document.suggested_classification
                document.suggested_classification = escalation.value
                self._container.audit.record(
                    session,
                    actor,
                    "document.sensitivity_escalation_suggested",
                    resource_type="document",
                    resource_id=document.id,
                    details={
                        "version_id": str(version.id),
                        "classification": document.classification,
                        "previous_suggestion": previous,
                        "suggested": escalation.value,
                        "signals": list(analysis.sensitivity.signals),
                    },
                )
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._container.audit.record(
                session,
                actor,
                "document.indexed",
                resource_type="document",
                resource_id=document.id,
                details={
                    "version_id": str(version.id),
                    "version_number": version.version_number,
                    "chunks": len(chunked.chunks),
                    "pages": version.page_count,
                    "semantic_indexed": version.semantic_indexed,
                    "needs_ocr": version.needs_ocr,
                    "became_current": becomes_current,
                    "doc_type": document.doc_type,
                    "injection_flagged_chunks": flagged_warn,
                    "fields": len(analysis.fields),
                    "duration_ms": duration_ms,
                },
            )
            if settings.retrieval.backend == "qdrant":
                await enqueue(
                    session,
                    kind="vector_sync_version",
                    organization_id=claimed.org_id,
                    payload={"version_id": str(version.id)},
                )
        metrics.INGESTION.labels(outcome="indexed").inc()
        metrics.INGESTION_LATENCY.labels(format=claimed.extension).observe(
            time.perf_counter() - started
        )
        if flagged_warn:
            metrics.SECURITY_EVENTS.labels(kind="injection_detected").inc(flagged_warn)
        log.info(
            "version_indexed",
            version_id=str(claimed.version_id),
            chunks=len(chunked.chunks),
            semantic=vectors is not None,
            became_current=becomes_current,
        )
        return {
            "status": "indexed",
            "version_id": str(claimed.version_id),
            "document_id": str(claimed.document_id),
            "chunks": len(chunked.chunks),
            "pages": len(analysis.parsed.pages),
            "fields": len(analysis.fields),
            "semantic_indexed": vectors is not None,
            "needs_ocr": analysis.parsed.needs_ocr,
            "became_current": becomes_current,
            "injection_flagged_chunks": flagged_warn,
            "doc_type": analysis.doc_type.doc_type.value,
        }

    async def _replace_rows(
        self,
        session: AsyncSession,
        claimed: _Claimed,
        analysis: _Analysis,
        vectors: list[list[float]] | None,
    ) -> None:
        org, version_id = claimed.org_id, claimed.version_id
        await session.execute(
            delete(ExtractedField).where(
                ExtractedField.organization_id == org, ExtractedField.version_id == version_id
            )
        )
        await session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.organization_id == org, DocumentChunk.version_id == version_id
            )
        )  # embeddings go with their chunks (ON DELETE CASCADE)
        chunks = analysis.chunked.chunks
        chunk_rows = [
            {
                "id": row.chunk_id,
                "organization_id": org,
                "document_id": claimed.document_id,
                "version_id": version_id,
                "chunk_index": chunk.index,
                "content": chunk.text,
                "content_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "heading_path": list(chunk.heading_path),
                "block_types": list(chunk.block_types),
                "token_count": chunk.token_count,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "injection_score": row.report.score,
                "injection_flags": list(row.report.flags),
                "pii_types": row.pii,
            }
            for chunk, row in zip(chunks, analysis.rows, strict=True)
        ]
        await session.execute(insert(DocumentChunk), chunk_rows)
        if vectors is not None:
            model = self.embeddings.model
            await session.execute(
                insert(ChunkEmbedding),
                [
                    {
                        "chunk_id": row.chunk_id,
                        "organization_id": org,
                        "document_id": claimed.document_id,
                        "version_id": version_id,
                        "model": model[:100],
                        "embedding": vector,
                    }
                    for row, vector in zip(analysis.rows, vectors, strict=True)
                ],
            )
        field_rows = []
        for candidate in analysis.fields:
            chunk = analysis.chunked.chunk_at(candidate.start) or _nearest_chunk(
                analysis.chunked, candidate.start
            )
            field_rows.append(
                {
                    "id": uuid7(),
                    "organization_id": org,
                    "document_id": claimed.document_id,
                    "version_id": version_id,
                    "field": candidate.field,
                    "value_text": candidate.value_text[:1000] if candidate.value_text else None,
                    "value_date": candidate.value_date,
                    "value_number": candidate.value_number,
                    "currency": candidate.currency,
                    "confidence": candidate.confidence,
                    "method": "rules",
                    "chunk_id": analysis.rows[chunk.index].chunk_id if chunk is not None else None,
                    "page": analysis.chunked.page_at(candidate.start),
                    "evidence": candidate.evidence,
                }
            )
        if field_rows:
            await session.execute(insert(ExtractedField), field_rows)

    @staticmethod
    async def _is_newest(
        session: AsyncSession, document: Document, version: DocumentVersion
    ) -> bool:
        if document.current_version_id is None or document.current_version_id == version.id:
            return True
        current_number = (
            await session.execute(
                select(DocumentVersion.version_number).where(
                    DocumentVersion.id == document.current_version_id,
                    DocumentVersion.organization_id == document.organization_id,
                )
            )
        ).scalar_one_or_none()
        return current_number is None or version.version_number >= current_number

    def _above_external_ceiling(
        self, document: Document, analysis: _Analysis, ceiling: Classification
    ) -> bool:
        if not self.embeddings.is_external:
            return False
        levels = [Classification(document.classification)]
        if document.suggested_classification:
            levels.append(Classification(document.suggested_classification))
        if analysis.sensitivity.suggested is not None:
            levels.append(analysis.sensitivity.suggested)
        return Classification.highest(levels).rank > ceiling.rank

    async def _external_ceiling(
        self, org_id: uuid.UUID, session: AsyncSession | None = None
    ) -> Classification:
        """The stricter of the deployment ceiling and the organisation's own AI policy.

        Organisations may only *lower* the ceiling (``organizations.settings.llm``), exactly as
        the LLM gateway applies it - embeddings are data egress too.
        """
        ceiling = self.settings.llm.external_max_classification
        stmt = select(Organization.settings).where(Organization.id == org_id)
        if session is None:
            async with self.db.session(DbContext.system_for_org(org_id)) as own:
                stored = (await own.execute(stmt)).scalar_one_or_none()
        else:
            stored = (await session.execute(stmt)).scalar_one_or_none()
        org_settings: dict[str, Any] = stored or {}
        llm_policy = org_settings.get("llm")
        value = (
            llm_policy.get("external_max_classification") if isinstance(llm_policy, dict) else None
        )
        try:
            org_ceiling = Classification(value) if isinstance(value, str) else None
        except ValueError:
            org_ceiling = None
        if org_ceiling is not None and org_ceiling.rank < ceiling.rank:
            return org_ceiling
        return ceiling

    @staticmethod
    def _escalation(document: Document, sensitivity: SensitivityResult) -> Classification | None:
        floor = Classification(document.classification)
        if document.suggested_classification:
            floor = Classification.highest(
                [floor, Classification(document.suggested_classification)]
            )
        return sensitivity.escalation_over(floor)

    def _metadata(self, analysis: _Analysis, skip_reason: str | None) -> dict[str, Any]:
        meta = analysis.parsed.metadata
        rows = analysis.rows
        data: dict[str, Any] = {
            "title": _clean_meta(meta.title),
            "author": _clean_meta(meta.author),
            "subject": _clean_meta(meta.subject),
            "created": _clean_meta(meta.created, 40),
            "modified": _clean_meta(meta.modified, 40),
            "source_page_count": meta.page_count,
            "page_basis": meta.page_basis,
            "sheet_names": [clean_line_text(name, 100) for name in meta.sheet_names[:50]],
            "warnings": list(analysis.parsed.warnings),
            "unicode": analysis.normalized.report.to_json(),
            "sandbox": {
                "network_isolated": analysis.parsed.network_isolated,
                "rlimits_applied": analysis.parsed.rlimits_applied,
            },
            "ocr": analysis.ocr,
            "doc_type_guess": {
                "doc_type": analysis.doc_type.doc_type.value,
                "confidence": analysis.doc_type.confidence,
            },
            "sensitivity": {
                "signals": list(analysis.sensitivity.signals),
                "counts": analysis.sensitivity.counts,
                "suggested": analysis.sensitivity.suggested.value
                if analysis.sensitivity.suggested
                else None,
            },
            "injection": {
                "flagged_chunks": sum(1 for r in rows if r.report.score > 0),
                "max_score": max((r.report.score for r in rows), default=0.0),
            },
            "embedding": {
                "model": self.embeddings.model if skip_reason is None else None,
                "skipped_reason": skip_reason,
            },
            "fields": len(analysis.fields),
        }
        if len(json.dumps(data, default=str)) > MAX_METADATA_JSON_BYTES:
            data["sheet_names"] = data["sheet_names"][:5]
            data["warnings"] = data["warnings"][:10]
        return data

    def _findings(
        self, existing: list[dict[str, Any]], analysis: _Analysis, flagged_warn: int
    ) -> list[dict[str, Any]]:
        findings = [
            f for f in existing if not (isinstance(f, dict) and f.get("source") == "ingestion")
        ]
        rows = analysis.rows
        if flagged_warn:
            top = max(r.report.score for r in rows)
            severity = (
                "high" if top >= self.settings.retrieval.injection_exclude_threshold else "warning"
            )
            flags = sorted({flag for r in rows for flag in r.report.flags})
            findings.append(
                {
                    "code": "prompt_injection_suspected",
                    "severity": severity,
                    "detail": (
                        f"{flagged_warn} chunk(s) at or above the warning threshold; "
                        f"max score {top:.2f}; flags: {','.join(flags)}"
                    )[:500],
                    "source": "ingestion",
                }
            )
        report = analysis.normalized.report
        if report.tags or report.bidi:
            findings.append(
                {
                    "code": "hidden_unicode_characters",
                    "severity": "warning",
                    "detail": f"tag characters: {report.tags}; bidi controls: {report.bidi}",
                    "source": "ingestion",
                }
            )
        if report.hidden_blocks:
            findings.append(
                {
                    "code": "hidden_document_text",
                    "severity": "info",
                    "detail": f"{report.hidden_blocks} block(s) contain text hidden from readers",
                    "source": "ingestion",
                }
            )
        if analysis.sensitivity.suggested is not None:
            findings.append(
                {
                    "code": "sensitive_content_detected",
                    "severity": "info",
                    "detail": "signals: " + ",".join(analysis.sensitivity.signals),
                    "source": "ingestion",
                }
            )
        return findings

    # ------------------------------------------------------------------ failures
    async def _record_failure(
        self,
        org_id: uuid.UUID,
        version_id: uuid.UUID,
        code: str,
        *,
        needs_ocr: bool = False,
        only_if_pending: bool = False,
    ) -> bool:
        async with self.db.transaction(DbContext.system_for_org(org_id)) as session:
            locked = await self._lock(session, org_id, version_id)
            if locked is None:
                return False
            version, document = locked
            if version.status == VersionStatus.INDEXED.value:
                return False
            pending = (VersionStatus.UPLOADED.value, VersionStatus.PROCESSING.value)
            if only_if_pending and version.status not in pending:
                return False
            version.status = VersionStatus.FAILED.value
            version.error_code = code[:64]
            version.error_detail = ERROR_DETAILS.get(code, ERROR_DETAILS["internal_error"])
            version.processed_at = utcnow()
            version.needs_ocr = version.needs_ocr or needs_ocr
            if (
                document.current_version_id is None
                and document.status == DocumentStatus.PROCESSING.value
            ):
                document.status = DocumentStatus.FAILED.value
            self._container.audit.record(
                session,
                Actor.system(org_id),
                "document.ingestion_failed",
                outcome=AuditOutcome.FAILURE,
                resource_type="document",
                resource_id=document.id,
                details={"version_id": str(version_id), "error_code": code[:64]},
            )
        metrics.INGESTION.labels(outcome="failed").inc()
        log.warning("version_failed", version_id=str(version_id), error_code=code)
        return True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _nearest_chunk(chunked: ChunkedDocument, offset: int) -> Chunk | None:
    """The last chunk starting at or before ``offset`` (fields found in a separator gap)."""
    candidates = [chunk for chunk in chunked.chunks if chunk.char_start <= offset]
    return candidates[-1] if candidates else (chunked.chunks[0] if chunked.chunks else None)


def _with_warning(parsed: ParsedDocument, code: str) -> ParsedDocument:
    if code in parsed.warnings:
        return parsed
    return ParsedDocument(
        format=parsed.format,
        pages=parsed.pages,
        metadata=parsed.metadata,
        warnings=(*parsed.warnings, code),
        needs_ocr=parsed.needs_ocr,
        network_isolated=parsed.network_isolated,
        rlimits_applied=parsed.rlimits_applied,
        ocr_images=(),
    )


def _merge_ocr(
    parsed: ParsedDocument, texts: dict[int, str], max_block_chars: int
) -> ParsedDocument:
    """Append recognised text as paragraphs to their pages; such pages no longer need OCR."""
    pages: list[ParsedPage] = []
    for page in parsed.pages:
        text = texts.get(page.number)
        if not text:
            pages.append(page)
            continue
        extra = tuple(
            Block("paragraph", piece)
            for paragraph in text.split("\n\n")
            for piece in split_at_whitespace(" ".join(paragraph.split()), max_block_chars)
        )
        pages.append(ParsedPage(number=page.number, blocks=(*page.blocks, *extra), needs_ocr=False))
    return ParsedDocument(
        format=parsed.format,
        pages=tuple(pages),
        metadata=parsed.metadata,
        warnings=(*parsed.warnings, "ocr_applied") if texts else parsed.warnings,
        needs_ocr=any(page.needs_ocr for page in pages),
        network_isolated=parsed.network_isolated,
        rlimits_applied=parsed.rlimits_applied,
        ocr_images=(),
    )
