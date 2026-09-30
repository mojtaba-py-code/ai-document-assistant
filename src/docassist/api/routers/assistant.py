"""Assistant endpoints: grounded answers, private conversations, the read-only agent and the
data-governance routing table shown by the UI.

* ``POST /ask``                      - ``assistant:use``
* ``GET /conversations``             - ``assistant:use`` (own conversations only)
* ``GET /conversations/{id}``        - ``assistant:use`` (another user's id -> 404)
* ``DELETE /conversations/{id}``     - ``assistant:use`` (another user's id -> 404)
* ``POST /agent``                    - ``assistant:agent``
* ``GET /policy``                    - ``assistant:use``

Responses carry ``Cache-Control: no-store`` (they contain document-derived text).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field

from docassist.api.container import Container
from docassist.api.deps import get_container, require
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import Classification, DocumentType
from docassist.search.types import SearchFilters

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])

_NO_STORE = {"Cache-Control": "no-store"}
_can_use = require(Permission.ASSISTANT_USE)
_can_run_agent = require(Permission.ASSISTANT_AGENT)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FiltersIn(_Strict):
    doc_types: list[DocumentType] = Field(default_factory=list, max_length=20)
    department_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)
    classifications: list[Classification] = Field(default_factory=list, max_length=4)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=20)
    created_after: datetime | None = None
    created_before: datetime | None = None

    def to_search_filters(self, include_old_versions: bool) -> SearchFilters:
        return SearchFilters(
            doc_types=tuple(t.value for t in self.doc_types),
            department_ids=tuple(self.department_ids),
            classifications=tuple(c.value for c in self.classifications),
            document_ids=tuple(self.document_ids),
            tags=tuple(self.tags),
            created_after=self.created_after,
            created_before=self.created_before,
            include_old_versions=include_old_versions,
        )


class AskRequest(_Strict):
    question: str = Field(min_length=1, max_length=20_000)
    conversation_id: uuid.UUID | None = None
    filters: FiltersIn | None = None
    include_old_versions: bool = False


class AgentRequest(_Strict):
    task: str = Field(min_length=1, max_length=20_000)


def _summary(item: Any) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "title": item.title,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
        "message_count": item.message_count,
    }


@router.post("/ask")
async def ask(
    body: AskRequest,
    response: Response,
    principal: Principal = Depends(_can_use),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    filters = body.filters.to_search_filters(body.include_old_versions) if body.filters else None
    answer = await container.answers.ask(
        principal,
        question=body.question,
        conversation_id=body.conversation_id,
        filters=filters,
        include_old_versions=body.include_old_versions,
    )
    response.headers.update(_NO_STORE)
    return answer.to_dict()


@router.get("/conversations")
async def list_conversations(
    response: Response,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=200),
    principal: Principal = Depends(_can_use),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    items, next_cursor = await container.answers.conversations.list(
        principal, limit=limit, cursor=cursor
    )
    response.headers.update(_NO_STORE)
    return {"items": [_summary(i) for i in items], "next_cursor": next_cursor}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID,
    response: Response,
    principal: Principal = Depends(_can_use),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    detail = await container.answers.conversations.get(principal, conversation_id)
    response.headers.update(_NO_STORE)
    return {
        "id": str(detail.id),
        "title": detail.title,
        "created_at": detail.created_at.isoformat(),
        "updated_at": detail.updated_at.isoformat(),
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "status": m.status,
                "confidence": m.confidence,
                "citations": m.citations,
                "warnings": m.warnings,
                "model": m.model,
                "created_at": m.created_at.isoformat(),
            }
            for m in detail.messages
        ],
    }


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    principal: Principal = Depends(_can_use),
    container: Container = Depends(get_container),
) -> Response:
    await container.answers.conversations.delete(principal, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/agent")
async def run_agent(
    body: AgentRequest,
    response: Response,
    principal: Principal = Depends(_can_run_agent),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    result = await container.agent.run(principal, task=body.task)
    response.headers.update(_NO_STORE)
    return result.to_dict()


@router.get("/policy")
async def routing_policy(
    principal: Principal = Depends(_can_use),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    org_id = principal.require_org()
    rows = await container.llm.policy(org_id)
    ceiling = await container.llm.org_ceiling(org_id)
    effective = container.llm.deployment_ceiling
    if ceiling is not None and ceiling.rank < effective.rank:
        effective = ceiling
    agent_ceiling = await container.agent.ceiling_for(principal)
    return {
        "external_max_classification": effective.value,
        "pseudonymize_pii_for_external": container.settings.llm.pseudonymize_pii_for_external,
        "agent_available": agent_ceiling is not None,
        "routes": [
            {
                "classification": row.classification.value,
                "allowed": row.allowed,
                "provider": row.provider,
                "model": row.model,
                "external": row.external,
            }
            for row in rows
        ],
    }
