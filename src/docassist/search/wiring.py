"""Service wiring for the search area.

Sets ``container.embeddings`` (honours ``overrides["embeddings"]``), ``container.vector_store``
(honours ``overrides["vector_store"]``; otherwise pgvector, or Qdrant when
``retrieval.backend == "qdrant"`` - the optional ``qdrant-client`` dependency is imported only
then) and ``container.search``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docassist.embeddings.base import EmbeddingProvider
from docassist.embeddings.factory import create_embedding_provider
from docassist.search.pgvector_store import PgVectorStore
from docassist.search.service import SearchService
from docassist.search.vector_store import VectorStore

if TYPE_CHECKING:
    from docassist.api.container import Container


def build_vector_store(container: Container, embeddings: EmbeddingProvider) -> VectorStore:
    settings = container.settings
    if settings.retrieval.backend == "qdrant":
        try:
            from docassist.search.qdrant_store import QdrantVectorStore
        except ImportError as exc:  # pragma: no cover - depends on the installed extras
            raise RuntimeError(
                "retrieval.backend 'qdrant' requires the optional dependency: "
                "pip install 'docassist[qdrant]'"
            ) from exc
        return QdrantVectorStore.from_settings(
            settings,
            db=container.worker_db or container.db,
            dimensions=embeddings.dimensions,
            model=embeddings.model,
        )
    return PgVectorStore(
        db=container.db,
        model=embeddings.model,
        dimensions=embeddings.dimensions,
        ef_search=settings.retrieval.hnsw_ef_search,
    )


def wire(container: Container) -> None:
    """Attach this area's services to the container."""
    embeddings: EmbeddingProvider = container.overrides.get(
        "embeddings"
    ) or create_embedding_provider(container.settings, container.egress)
    container.embeddings = embeddings
    store: VectorStore = container.overrides.get("vector_store") or build_vector_store(
        container, embeddings
    )
    container.vector_store = store
    container.search = SearchService(container)
