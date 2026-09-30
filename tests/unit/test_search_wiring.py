"""Search-area wiring: overrides, backend selection and the optional Qdrant dependency."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from docassist.embeddings.hashing import HashingEmbedder
from docassist.search.pgvector_store import PgVectorStore
from docassist.search.qdrant_store import QdrantVectorStore
from docassist.search.service import SearchService
from docassist.search.wiring import build_vector_store, wire
from tests.conftest import make_settings


def _container(settings: Any, **overrides: Any) -> SimpleNamespace:
    return SimpleNamespace(
        settings=settings, overrides=overrides, egress=None, db=object(), worker_db=None
    )


def test_default_wiring_uses_hashing_embedder_and_pgvector() -> None:
    container = _container(make_settings())
    wire(container)  # type: ignore[arg-type]
    assert isinstance(container.embeddings, HashingEmbedder)
    assert container.embeddings.dimensions == 1024
    assert isinstance(container.vector_store, PgVectorStore)
    assert isinstance(container.search, SearchService)
    assert container.search.vector_store is container.vector_store
    assert container.search.embeddings is container.embeddings


def test_overrides_are_honoured() -> None:
    embedder, store = HashingEmbedder(dimensions=32, model="custom"), object()
    container = _container(make_settings(), embeddings=embedder, vector_store=store)
    wire(container)  # type: ignore[arg-type]
    assert container.embeddings is embedder
    assert container.vector_store is store


async def test_qdrant_backend_is_built_from_settings() -> None:
    settings = make_settings(
        retrieval={
            "backend": "qdrant",
            "qdrant_url": "http://qdrant:6333",
            "qdrant_collection": "c1",
        }
    )
    worker_db = object()
    container = _container(settings)
    container.worker_db = worker_db
    store = build_vector_store(container, HashingEmbedder(dimensions=64))  # type: ignore[arg-type]
    assert isinstance(store, QdrantVectorStore)
    assert store._collection == "c1" and store._dimensions == 64
    assert store._db is worker_db  # synchronisation runs in the worker
    await store.aclose()


def test_qdrant_backend_requires_a_url() -> None:
    container = _container(make_settings(retrieval={"backend": "qdrant"}))
    with pytest.raises(ValueError, match="qdrant_url"):
        build_vector_store(container, HashingEmbedder())  # type: ignore[arg-type]
