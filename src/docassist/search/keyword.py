"""Keyword retrieval with PostgreSQL full-text search.

Primary strategy: ``websearch_to_tsquery('english', q) || websearch_to_tsquery('simple', q)``
matched against ``document_chunks.tsv`` (section text weighted A, English stems B, raw words
C) and ranked with ``ts_rank_cd(..., 32)``, which is already normalised to ``0..1``
(``rank / (rank + 1)``). ``websearch_to_tsquery`` accepts arbitrary user input without
syntax errors, and the query text is always a bound parameter.

Fallbacks, used only when the primary query finds nothing:

* **prefix** - every query term as a prefix (``'contr':*`` finds "contract"), for partial
  words typed as you go;
* **substring** - when the query has no word characters at all (``"$$"``, ``"§ 4.2"``), a
  case-insensitive substring match with LIKE wildcards escaped.

Every strategy embeds the authorisation scope of :mod:`docassist.search.sql`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import ColumnClause, ColumnElement, bindparam, func, literal, literal_column, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import Float, String

from docassist.authz.principal import Principal
from docassist.db.models import DocumentChunk
from docassist.search.hybrid import clamp01
from docassist.search.sql import scoped
from docassist.search.text import like_pattern, prefix_tsquery
from docassist.search.types import SearchFilters

KeywordStrategy = Literal["fts", "prefix", "substring", "none"]

SUBSTRING_SCORE = 0.1
_RANK_NORMALIZATION = 32  # rank / (rank + 1)
_ENGLISH: ColumnClause[Any] = literal_column("'english'::regconfig")
_SIMPLE: ColumnClause[Any] = literal_column("'simple'::regconfig")


@dataclass(frozen=True, slots=True)
class KeywordResult:
    hits: list[tuple[uuid.UUID, float]] = field(default_factory=list)
    strategy: KeywordStrategy = "none"


def _websearch_query(query: str) -> ColumnElement[Any]:
    text_param = bindparam("kw_query", query, type_=String())
    english = func.websearch_to_tsquery(_ENGLISH, text_param, type_=TSQUERY)
    simple = func.websearch_to_tsquery(_SIMPLE, text_param, type_=TSQUERY)
    return english.op("||", return_type=TSQUERY)(simple)


async def _ranked(
    session: AsyncSession,
    principal: Principal,
    tsquery: ColumnElement[Any],
    *,
    now: datetime,
    filters: SearchFilters,
    limit: int,
) -> list[tuple[uuid.UUID, float]]:
    rank = func.ts_rank_cd(DocumentChunk.tsv, tsquery, _RANK_NORMALIZATION, type_=Float())
    stmt = (
        scoped(
            select(DocumentChunk.id, rank.label("rank")).select_from(DocumentChunk),
            principal,
            now,
            filters,
        )
        .where(DocumentChunk.tsv.op("@@")(tsquery))
        .order_by(rank.desc(), DocumentChunk.id)
        .limit(limit)
    )
    return [(row.id, clamp01(float(row.rank))) for row in (await session.execute(stmt)).all()]


async def keyword_search(
    session: AsyncSession,
    principal: Principal,
    query: str,
    *,
    filters: SearchFilters,
    limit: int,
    now: datetime,
) -> KeywordResult:
    """Up to ``limit`` ``(chunk_id, score 0..1)`` pairs, best first, for readable chunks only."""
    if principal.org_id is None or limit <= 0 or not query.strip():
        return KeywordResult()
    hits = await _ranked(
        session, principal, _websearch_query(query), now=now, filters=filters, limit=limit
    )
    if hits:
        return KeywordResult(hits, "fts")

    prefix = prefix_tsquery(query)
    if prefix is not None:
        tsquery = func.to_tsquery(
            _SIMPLE, bindparam("kw_prefix", prefix, type_=String()), type_=TSQUERY
        )
        hits = await _ranked(session, principal, tsquery, now=now, filters=filters, limit=limit)
        return KeywordResult(hits, "prefix")

    pattern = like_pattern(query)
    if pattern is None:
        return KeywordResult()
    stmt = (
        scoped(select(DocumentChunk.id).select_from(DocumentChunk), principal, now, filters)
        .where(DocumentChunk.content.ilike(literal(pattern, String()), escape="\\"))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return KeywordResult([(chunk_id, SUBSTRING_SCORE) for chunk_id in rows], "substring")
