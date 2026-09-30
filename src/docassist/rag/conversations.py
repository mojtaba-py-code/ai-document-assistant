"""Private conversation history.

Conversations and messages are protected twice: the ``owner_only`` row-level-security
policy (organisation *and* user must match the transaction context) and an explicit
``user_id`` predicate in every query here. Another user's conversation - even in the same
organisation - is indistinguishable from a missing one (:class:`NotFound`).
"""

from __future__ import annotations

import base64
import binascii
import uuid
from datetime import datetime

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.audit.service import Actor, AuditLogger
from docassist.authz.principal import Principal
from docassist.core.enums import MessageRole
from docassist.core.errors import NotFound, ValidationFailed
from docassist.db.models import Conversation, Message
from docassist.db.session import Database
from docassist.rag.types import ConversationDetail, ConversationMessage, ConversationSummary

MAX_PAGE = 100
MAX_MESSAGES_RETURNED = 500


async def load_owned_conversation(
    session: AsyncSession, principal: Principal, conversation_id: uuid.UUID
) -> Conversation:
    row = (
        await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.organization_id == principal.require_org(),
                Conversation.user_id == principal.user_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise NotFound(internal_detail="conversation not visible to caller")
    return row


async def previous_questions(
    session: AsyncSession, principal: Principal, conversation_id: uuid.UUID, *, limit: int = 3
) -> list[str]:
    """The caller's most recent questions in the conversation (newest first) - never answers."""
    rows = (
        await session.execute(
            select(Message.content)
            .where(
                Message.conversation_id == conversation_id,
                Message.user_id == principal.user_id,
                Message.role == MessageRole.USER.value,
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
    ).scalars()
    return [str(content) for content in rows]


def encode_cursor(updated_at: datetime, conversation_id: uuid.UUID) -> str:
    raw = f"{updated_at.isoformat()}|{conversation_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        stamp, _, ident = base64.urlsafe_b64decode(padded.encode()).decode().partition("|")
        moment = datetime.fromisoformat(stamp)
        conversation_id = uuid.UUID(ident)
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ValidationFailed("The cursor is invalid.") from exc
    if moment.tzinfo is None:
        raise ValidationFailed("The cursor is invalid.")
    return moment, conversation_id


class ConversationStore:
    def __init__(self, db: Database, audit: AuditLogger) -> None:
        self._db = db
        self._audit = audit

    async def list(
        self, principal: Principal, *, limit: int = 20, cursor: str | None = None
    ) -> tuple[list[ConversationSummary], str | None]:
        org_id = principal.require_org()
        limit = max(1, min(limit, MAX_PAGE))
        counts = (
            select(Message.conversation_id, func.count(Message.id).label("n"))
            .where(Message.user_id == principal.user_id)
            .group_by(Message.conversation_id)
            .subquery()
        )
        stmt = (
            select(Conversation, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.conversation_id == Conversation.id)
            .where(
                Conversation.organization_id == org_id, Conversation.user_id == principal.user_id
            )
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit + 1)
        )
        if cursor:
            stamp, last_id = decode_cursor(cursor)
            stmt = stmt.where(
                or_(
                    Conversation.updated_at < stamp,
                    and_(Conversation.updated_at == stamp, Conversation.id < last_id),
                )
            )
        async with self._db.session(principal.db_context) as session:
            rows = (await session.execute(stmt)).all()
        items = [
            ConversationSummary(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=int(count),
            )
            for conv, count in rows[:limit]
        ]
        next_cursor = (
            encode_cursor(items[-1].updated_at, items[-1].id) if len(rows) > limit else None
        )
        return items, next_cursor

    async def get(self, principal: Principal, conversation_id: uuid.UUID) -> ConversationDetail:
        async with self._db.session(principal.db_context) as session:
            conv = await load_owned_conversation(session, principal, conversation_id)
            messages = (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conv.id, Message.user_id == principal.user_id)
                    .order_by(Message.created_at, Message.id)
                    .limit(MAX_MESSAGES_RETURNED)
                )
            ).scalars()
            items = [
                ConversationMessage(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    status=m.status,
                    confidence=m.confidence,
                    citations=list(m.citations or []),
                    warnings=list(m.warnings or []),
                    model=m.model,
                    created_at=m.created_at,
                )
                for m in messages
            ]
        return ConversationDetail(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=items,
        )

    async def delete(self, principal: Principal, conversation_id: uuid.UUID) -> None:
        async with self._db.transaction(principal.db_context) as session:
            conv = await load_owned_conversation(session, principal, conversation_id)
            result = await session.execute(
                delete(Message).where(
                    Message.conversation_id == conv.id, Message.user_id == principal.user_id
                )
            )
            removed = int(getattr(result, "rowcount", 0) or 0)
            await session.delete(conv)
            self._audit.record(
                session,
                Actor.of(principal),
                "assistant.conversation_deleted",
                resource_type="conversation",
                resource_id=conv.id,
                details={"messages": removed},
            )
