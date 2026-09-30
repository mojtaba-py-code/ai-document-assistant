"""Value types shared by the search layer, the RAG layer and the search API.

Everything here is immutable. :class:`SearchFilters` validates itself on construction so a
filter object that exists is always safe to compile into SQL or a Qdrant filter: enum values
are checked against the domain vocabularies, list sizes are bounded, tags cannot carry control
characters (PostgreSQL rejects NUL bytes with a server error) and time bounds must be
timezone-aware.
"""

from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from docassist.core.enums import Classification, DocumentType
from docassist.core.errors import ValidationFailed

SearchMode = Literal["hybrid", "semantic", "keyword"]
SEARCH_MODES: tuple[SearchMode, ...] = ("hybrid", "semantic", "keyword")

MAX_FILTER_VALUES = 50
"""Upper bound for every list-valued filter (ids, tags, types...)."""

MAX_TAG_CHARS = 64

_DOC_TYPES = frozenset(t.value for t in DocumentType)
_CLASSIFICATIONS = frozenset(c.value for c in Classification)


def _bounded(name: str, values: tuple[object, ...]) -> None:
    if len(values) > MAX_FILTER_VALUES:
        raise ValidationFailed(f"At most {MAX_FILTER_VALUES} values are allowed for '{name}'.")


def _clean_tag(tag: str) -> str:
    if not isinstance(tag, str):
        raise ValidationFailed("Tags must be strings.")
    stripped = tag.strip()
    if not stripped or len(stripped) > MAX_TAG_CHARS:
        raise ValidationFailed(f"Tags must be 1 to {MAX_TAG_CHARS} characters long.")
    if any(unicodedata.category(ch) in {"Cc", "Cf", "Cs"} for ch in stripped):
        raise ValidationFailed("Tags must not contain control or invisible characters.")
    return stripped


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Optional narrowing of a search. Every facet is an *any-of* match; facets combine with AND.

    Filters only ever narrow what the caller may read - they are appended to the
    authorisation predicate, never substituted for it. ``created_after`` is inclusive and
    ``created_before`` exclusive (a half-open interval on ``documents.created_at``).
    """

    doc_types: tuple[str, ...] = ()
    department_ids: tuple[uuid.UUID, ...] = ()
    classifications: tuple[str, ...] = ()
    document_ids: tuple[uuid.UUID, ...] = ()
    tags: tuple[str, ...] = ()
    created_after: datetime | None = None
    created_before: datetime | None = None
    include_old_versions: bool = False

    def __post_init__(self) -> None:
        for name in ("doc_types", "department_ids", "classifications", "document_ids", "tags"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
            _bounded(name, getattr(self, name))
        if any(t not in _DOC_TYPES for t in self.doc_types):
            raise ValidationFailed("Unknown document type in filters.")
        if any(c not in _CLASSIFICATIONS for c in self.classifications):
            raise ValidationFailed("Unknown classification in filters.")
        for name in ("department_ids", "document_ids"):
            if any(not isinstance(v, uuid.UUID) for v in getattr(self, name)):
                raise ValidationFailed(f"'{name}' must contain UUIDs.")
        object.__setattr__(self, "tags", tuple(dict.fromkeys(_clean_tag(t) for t in self.tags)))
        for name in ("created_after", "created_before"):
            bound = getattr(self, name)
            if bound is not None and (not isinstance(bound, datetime) or bound.tzinfo is None):
                raise ValidationFailed(f"'{name}' must be a timezone-aware timestamp.")
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after >= self.created_before
        ):
            raise ValidationFailed("'created_after' must be earlier than 'created_before'.")

    @property
    def active_facets(self) -> int:
        """How many facets narrow the search (used for audit details, never the values)."""
        facets = (
            self.doc_types,
            self.department_ids,
            self.classifications,
            self.document_ids,
            self.tags,
        )
        count = sum(1 for f in facets if f)
        count += sum(1 for b in (self.created_after, self.created_before) if b is not None)
        return count


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One authorised chunk with its provenance and scores.

    ``score`` is the final ranking score in ``0..1``; ``keyword_score`` (normalised
    ``ts_rank_cd``) and ``semantic_score`` (cosine similarity clamped to ``0..1``) are the raw
    signals, ``None`` when the chunk was not found by that retriever.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    is_current: bool
    document_title: str
    classification: str
    doc_type: str
    department_id: uuid.UUID | None
    page_start: int | None
    page_end: int | None
    section: str | None
    heading_path: tuple[str, ...]
    content: str
    score: float
    keyword_score: float | None
    semantic_score: float | None
    injection_score: float
    injection_flags: tuple[str, ...]
    pii_types: tuple[str, ...]
    content_sha256: str = ""
    chunk_index: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A search hit as shown to a user (``snippet`` instead of the whole chunk)."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    document_title: str
    version_number: int
    is_current: bool
    classification: str
    doc_type: str
    page_start: int | None
    page_end: int | None
    section: str | None
    snippet: str
    score: float
    keyword_score: float | None
    semantic_score: float | None
    flagged: bool


@dataclass(frozen=True, slots=True)
class SearchResponse:
    results: list[SearchResult]
    mode_used: SearchMode
    degraded: bool
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def took_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)


class RetrievalResult(list[RetrievedChunk]):
    """Chunks for the RAG layer (best first) plus retrieval statistics.

    It *is* a ``list[RetrievedChunk]`` so callers can iterate, index and test for emptiness
    directly; the attributes explain what happened on the way:

    * ``excluded_injection`` - candidates dropped because their prompt-injection score reached
      ``retrieval.injection_exclude_threshold``;
    * ``below_relevance`` - candidates dropped by ``retrieval.min_relevance``;
    * ``candidates`` - authorised candidates considered after fusion;
    * ``degraded`` / ``mode_used`` - ``True``/``"keyword"`` when semantic retrieval was
      unavailable (embedding provider or vector store failure).
    """

    excluded_injection: int
    below_relevance: int
    candidates: int
    degraded: bool
    mode_used: SearchMode
    timings_ms: dict[str, float]

    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        *,
        excluded_injection: int = 0,
        below_relevance: int = 0,
        candidates: int = 0,
        degraded: bool = False,
        mode_used: SearchMode = "hybrid",
        timings_ms: dict[str, float] | None = None,
    ) -> None:
        super().__init__(chunks or [])
        self.excluded_injection = excluded_injection
        self.below_relevance = below_relevance
        self.candidates = candidates
        self.degraded = degraded
        self.mode_used = mode_used
        self.timings_ms = dict(timings_ms or {})
