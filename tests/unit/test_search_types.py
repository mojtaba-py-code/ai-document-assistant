"""SearchFilters validation and the RetrievalResult container."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from docassist.core.errors import ValidationFailed
from docassist.search.types import MAX_FILTER_VALUES, RetrievalResult, SearchFilters

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def test_defaults_are_empty() -> None:
    filters = SearchFilters()
    assert filters.active_facets == 0
    assert filters.include_old_versions is False


def test_lists_become_tuples_and_tags_are_deduplicated() -> None:
    dept = uuid.uuid4()
    filters = SearchFilters(
        doc_types=["contract"],  # type: ignore[arg-type]
        department_ids=[dept],  # type: ignore[arg-type]
        tags=[" legal ", "legal", "hr"],  # type: ignore[arg-type]
        created_after=NOW - timedelta(days=1),
        created_before=NOW,
    )
    assert filters.doc_types == ("contract",)
    assert filters.department_ids == (dept,)
    assert filters.tags == ("legal", "hr")
    assert filters.active_facets == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"doc_types": ("contract'; DROP TABLE documents;--",)},
        {"classifications": ("TOP_SECRET",)},
        {"classifications": ("public",)},
        {"department_ids": ("not-a-uuid",)},
        {"document_ids": (1,)},
        {"tags": ("",)},
        {"tags": ("x" * 65,)},
        {"tags": ("bad\x00tag",)},
        {"tags": ("zero" + chr(0x200B) + "width",)},
        {"tags": ("bidi" + chr(0x202E),)},
        {"tags": (5,)},
        {"created_after": datetime(2026, 1, 1)},  # naive
        {"created_before": "2026-01-01"},
        {"created_after": NOW, "created_before": NOW},
        {"created_after": NOW, "created_before": NOW - timedelta(seconds=1)},
        {"document_ids": tuple(uuid.uuid4() for _ in range(MAX_FILTER_VALUES + 1))},
    ],
)
def test_invalid_filters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationFailed):
        SearchFilters(**kwargs)


def test_filters_are_immutable() -> None:
    filters = SearchFilters()
    with pytest.raises(AttributeError):
        filters.include_old_versions = True  # type: ignore[misc]


def test_retrieval_result_is_a_list_with_statistics() -> None:
    empty = RetrievalResult()
    assert empty == [] and not empty
    assert empty.excluded_injection == 0 and empty.degraded is False
    assert empty.mode_used == "hybrid"
    result = RetrievalResult(
        [],
        excluded_injection=2,
        below_relevance=3,
        candidates=9,
        degraded=True,
        mode_used="keyword",
        timings_ms={"total": 1.5},
    )
    assert (result.excluded_injection, result.below_relevance, result.candidates) == (2, 3, 9)
    assert result.degraded and result.mode_used == "keyword"
    assert result.timings_ms == {"total": 1.5}
