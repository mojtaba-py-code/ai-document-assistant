"""Engine / session management with PostgreSQL row-level-security context.

Every transaction opened through :class:`Database` runs

    SELECT set_config('app.org_id', ..., true),
           set_config('app.user_id', ..., true),
           set_config('app.platform', ..., true)

right after ``BEGIN`` (SQLAlchemy ``after_begin`` hook). ``is_local = true`` scopes the values
to the transaction, so a pooled connection can never carry one tenant's context into another
request. If no context is set, the RLS helper functions return NULL and every tenant policy
evaluates to false - the failure mode is "sees nothing", never "sees everything".
"""

from __future__ import annotations

import ssl as ssl_lib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import event, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, SessionTransaction

from docassist.core.config import DatabaseSettings
from docassist.core.errors import ServiceUnavailable

_CONTEXT_KEY = "docassist_db_context"
_SET_CONTEXT_SQL = text(
    "SELECT set_config('app.org_id', :org_id, true),"
    " set_config('app.user_id', :user_id, true),"
    " set_config('app.platform', :platform, true)"
)


@dataclass(frozen=True, slots=True)
class DbContext:
    """The tenant/user identity the database enforces RLS against."""

    org_id: uuid.UUID | None
    user_id: uuid.UUID | None = None
    platform: bool = False

    @classmethod
    def system_for_org(cls, org_id: uuid.UUID) -> DbContext:
        """Worker / pipeline context: one organisation, no user."""
        return cls(org_id=org_id, user_id=None, platform=False)

    @classmethod
    def anonymous(cls) -> DbContext:
        return cls(org_id=None, user_id=None, platform=False)


def _apply_context(session: Session, _tx: SessionTransaction, connection: Connection) -> None:
    ctx: DbContext = session.info.get(_CONTEXT_KEY) or DbContext.anonymous()
    connection.execute(
        _SET_CONTEXT_SQL,
        {
            "org_id": str(ctx.org_id) if ctx.org_id else "",
            "user_id": str(ctx.user_id) if ctx.user_id else "",
            "platform": "on" if ctx.platform else "off",
        },
    )


def _ssl_arg(mode: str) -> Any:
    if mode == "disable":
        return False
    if mode == "prefer":
        return "prefer"
    if mode == "require":
        return "require"
    context = ssl_lib.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl_lib.CERT_REQUIRED
    return context


# No asyncpg binary codec for ``vector`` is registered on purpose: the ORM ``Vector`` type
# sends and parses the text form ("[0.1,0.2,...]"), which asyncpg passes through for types
# without a codec. Registering pgvector's codec would make those binds fail.
def create_engine(
    dsn: str,
    settings: DatabaseSettings,
    *,
    application_name: str,
    pool_size: int | None = None,
) -> AsyncEngine:
    return create_async_engine(
        dsn,
        pool_size=pool_size or settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_pre_ping=True,
        pool_recycle=1_800,
        echo=settings.echo,
        connect_args={
            "ssl": _ssl_arg(settings.ssl),
            "timeout": 10,
            "command_timeout": settings.statement_timeout_ms / 1000 + 5,
            "server_settings": {
                "application_name": application_name,
                "statement_timeout": str(settings.statement_timeout_ms),
                "lock_timeout": str(settings.lock_timeout_ms),
                "idle_in_transaction_session_timeout": str(settings.idle_in_transaction_timeout_ms),
            },
        },
    )


class Database:
    """Owns one engine and hands out RLS-scoped sessions."""

    def __init__(self, engine: AsyncEngine, role: Literal["api", "worker", "owner"]) -> None:
        self.engine = engine
        self.role = role
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    @asynccontextmanager
    async def session(self, ctx: DbContext) -> AsyncIterator[AsyncSession]:
        """A session whose *every* transaction carries ``ctx`` as RLS context.

        The caller commits explicitly (unit of work); anything uncommitted is rolled back.
        """
        session = self._sessionmaker()
        session.info[_CONTEXT_KEY] = ctx
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def transaction(self, ctx: DbContext) -> AsyncIterator[AsyncSession]:
        """Session + automatic commit on success."""
        async with self.session(ctx) as session:
            yield session
            await session.commit()

    async def ping(self) -> None:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            raise ServiceUnavailable(
                internal_detail=f"database ping failed: {type(exc).__name__}"
            ) from exc

    async def dispose(self) -> None:
        await self.engine.dispose()


# One listener for every Session class: the context lives in ``session.info``.
event.listen(Session, "after_begin", _apply_context)


async def verify_least_privilege(db: Database, expected_role: str) -> list[str]:
    """Return a list of problems if the connected role could bypass row-level security."""
    problems: list[str] = []
    async with db.engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT current_user, r.rolsuper, r.rolbypassrls,"
                    " r.rolcreaterole, r.rolcreatedb,"
                    " EXISTS (SELECT 1 FROM pg_tables t WHERE t.schemaname = 'public'"
                    "         AND t.tableowner = current_user) AS owns_tables"
                    " FROM pg_roles r WHERE r.rolname = current_user"
                )
            )
        ).one()
    user, is_super, bypass, createrole, createdb, owns = row
    if user != expected_role:
        problems.append(f"connected as {user!r}, expected {expected_role!r}")
    if is_super:
        problems.append("role is SUPERUSER (bypasses RLS)")
    if bypass:
        problems.append("role has BYPASSRLS")
    if createrole or createdb:
        problems.append("role can create roles/databases")
    if owns:
        problems.append("role owns tables (table owners bypass RLS)")
    return problems
