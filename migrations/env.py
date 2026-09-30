"""Alembic environment.

Migrations run as the *schema owner* role (``DOCASSIST_DATABASE__MIGRATION_URL``), never as
the API or worker role - those roles have no DDL privileges at all. Only the handful of
variables migrations need are read here, so running ``alembic upgrade`` does not require
the application's JWT/encryption secrets to be present.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from docassist.db.models import Base

sys.path.insert(0, str(Path(__file__).resolve().parent))  # docassist_migration_security

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    url = (
        x_args.get("dburl")
        or os.environ.get("DOCASSIST_DATABASE__MIGRATION_URL")
        or os.environ.get("DOCASSIST_DATABASE__URL")
    )
    if not url:
        file_var = os.environ.get("DOCASSIST_DATABASE__MIGRATION_URL_FILE")
        if file_var:
            with open(file_var, encoding="utf-8") as handle:
                url = handle.read().strip()
    if not url:
        raise RuntimeError("Set DOCASSIST_DATABASE__MIGRATION_URL (schema owner DSN) to migrate.")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url(), connect_args={"timeout": 15})
    async with engine.connect() as connection:
        await connection.run_sync(_run_sync)
        await connection.commit()
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
