"""Contract check: the answer service over the real search service (no retrieval fakes).

Chunks are seeded with embeddings from the configured (offline hashing) embedder, so hybrid
retrieval runs PostgreSQL full-text search and pgvector similarity, both with the
authorisation predicate inside the query; the offline extractive model answers.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers_rag import ChunkSpec, seed_document

pytestmark = [pytest.mark.db]

PAYMENT = (
    "The Customer shall pay each undisputed invoice within thirty (30) days of receipt (net 30)."
)


async def test_answer_over_real_retrieval(container: Any, factory: Any) -> None:
    if not hasattr(container, "search"):
        pytest.skip("search area not wired")
    org = await factory.org()
    other_org = await factory.org()
    user = await factory.user(org)
    outsider = await factory.user(other_org)
    contract = await seed_document(
        container,
        org_id=org,
        owner_id=user.id,
        title="Acme Supply Agreement",
        doc_type="contract",
        chunks=[ChunkSpec(PAYMENT, page=4, section="Payment Terms")],
        embed=True,
    )
    await seed_document(
        container,
        org_id=other_org,
        owner_id=outsider.id,
        title="Globex Supply Agreement",
        doc_type="contract",
        chunks=[
            ChunkSpec(
                "Globex shall pay each undisputed invoice within ninety (90) days.",
                section="Payment Terms",
            )
        ],
        embed=True,
    )
    principal = await factory.principal(user)
    answer = await container.answers.ask(
        principal, question="How many days does the customer have to pay each undisputed invoice?"
    )
    assert answer.status == "answered", answer
    assert {c.document_id for c in answer.citations} == {contract.id}
    assert all(e.document_title != "Globex Supply Agreement" for e in answer.evidence)
    assert answer.citations[0].quote == PAYMENT
