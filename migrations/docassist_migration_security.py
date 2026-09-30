"""Database-enforced security for the initial schema.

Three database roles:

* **schema owner** (runs migrations) - owns every table; never used by a running service.
* **app role** (API)     - DML only, row-level security on every tenant table, no DDL, cannot
  UPDATE/DELETE audit rows, cannot hard-delete documents.
* **worker role**        - like the app role plus cross-tenant job claiming, audit sealing and
  retention purges.

Tenant context comes from transaction-local settings (``app.org_id``, ``app.user_id``,
``app.platform``) that the application sets right after ``BEGIN``. When they are missing the
helper functions return NULL and every policy evaluates to false: **fail closed**.

Login-time lookups that must work *before* the tenant is known (email -> user, refresh token
-> session) go through narrow ``SECURITY DEFINER`` functions that return only identifiers,
with a pinned ``search_path``.
"""

from __future__ import annotations

import os
import re

from alembic import op

_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def _role(env: str, default: str) -> str:
    value = os.environ.get(env, default)
    if not _IDENT.fullmatch(value):
        raise RuntimeError(f"{env} must be a lowercase SQL identifier")
    return value


APP = _role("DOCASSIST_DATABASE__APP_ROLE", "docassist_app")
WORKER = _role("DOCASSIST_DATABASE__WORKER_ROLE", "docassist_worker")
EMBEDDING_DIMENSIONS = int(os.environ.get("DOCASSIST_EMBEDDING__DIMENSIONS", "1024"))
if not 16 <= EMBEDDING_DIMENSIONS <= 2000:
    raise RuntimeError("DOCASSIST_EMBEDDING__DIMENSIONS must be between 16 and 2000")

TENANT_TABLES = (
    "departments",
    "user_departments",
    "documents",
    "document_versions",
    "document_grants",
    "document_chunks",
    "chunk_embeddings",
    "extracted_fields",
    "llm_usage",
    "exports",
)
NULLABLE_TENANT_TABLES = (
    "users",
    "auth_sessions",
    "refresh_tokens",
    "password_reset_tokens",
    "mfa_challenges",
)
PRIVATE_TABLES = ("conversations", "messages")

# table -> privileges for the API role (anything not listed is not granted)
APP_GRANTS: dict[str, str] = {
    "organizations": "SELECT, INSERT, UPDATE",
    "departments": "SELECT, INSERT, UPDATE, DELETE",
    "users": "SELECT, INSERT, UPDATE",
    "user_departments": "SELECT, INSERT, UPDATE, DELETE",
    "auth_sessions": "SELECT, INSERT, UPDATE, DELETE",
    "refresh_tokens": "SELECT, INSERT, UPDATE, DELETE",
    "password_reset_tokens": "SELECT, INSERT, UPDATE, DELETE",
    "mfa_challenges": "SELECT, INSERT, UPDATE, DELETE",
    "documents": "SELECT, INSERT, UPDATE",
    "document_versions": "SELECT, INSERT, UPDATE",
    "document_grants": "SELECT, INSERT, UPDATE, DELETE",
    "document_chunks": "SELECT, INSERT, UPDATE, DELETE",
    "chunk_embeddings": "SELECT, INSERT, UPDATE, DELETE",
    "extracted_fields": "SELECT, INSERT, UPDATE, DELETE",
    "conversations": "SELECT, INSERT, UPDATE, DELETE",
    "messages": "SELECT, INSERT, UPDATE, DELETE",
    "jobs": "SELECT, INSERT, UPDATE",
    "audit_events": "SELECT, INSERT",
    "llm_usage": "SELECT, INSERT",
    "exports": "SELECT, INSERT, UPDATE",
    "audit_chain_heads": "SELECT",
}
WORKER_GRANTS: dict[str, str] = {
    **dict.fromkeys(APP_GRANTS, "SELECT, INSERT, UPDATE, DELETE"),
    "organizations": "SELECT",
    "audit_events": "SELECT, INSERT, UPDATE",
    "audit_chain_heads": "SELECT, INSERT, UPDATE",
}


STATEMENT_BREAK = "-- @@"


def _x(sql: str) -> None:
    """Execute one or more statements separated by ``-- @@`` lines (asyncpg runs one at a time)."""
    for statement in sql.split(STATEMENT_BREAK):
        if statement.strip():
            op.execute(statement)


def security_prelude() -> None:
    _x("CREATE EXTENSION IF NOT EXISTS vector")
    _x("CREATE SCHEMA IF NOT EXISTS app")
    for role in (APP, WORKER):
        _x(
            f"""
            DO $$
            BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                CREATE ROLE {role} NOLOGIN;
              END IF;
            END
            $$;
            """
        )


def _context_functions() -> None:
    _x(
        """
        CREATE OR REPLACE FUNCTION app.current_org_id() RETURNS uuid
        LANGUAGE sql STABLE PARALLEL SAFE
        AS $$ SELECT NULLIF(current_setting('app.org_id', true), '')::uuid $$
        -- @@

        CREATE OR REPLACE FUNCTION app.current_user_id() RETURNS uuid
        LANGUAGE sql STABLE PARALLEL SAFE
        AS $$ SELECT NULLIF(current_setting('app.user_id', true), '')::uuid $$
        -- @@

        CREATE OR REPLACE FUNCTION app.is_platform() RETURNS boolean
        LANGUAGE sql STABLE PARALLEL SAFE
        AS $$ SELECT coalesce(current_setting('app.platform', true), '') = 'on' $$
        """
    )


def _auth_lookup_functions() -> None:
    definer = "LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public"
    _x(
        f"""
        CREATE OR REPLACE FUNCTION app.auth_find_user(p_email text)
        RETURNS TABLE (user_id uuid, organization_id uuid, is_platform boolean) {definer}
        AS $$ SELECT u.id, u.organization_id, u.organization_id IS NULL
              FROM public.users u WHERE u.email = lower(p_email) $$
        -- @@

        CREATE OR REPLACE FUNCTION app.auth_find_refresh_token(p_hash bytea)
        RETURNS TABLE (session_id uuid, organization_id uuid, is_platform boolean) {definer}
        AS $$ SELECT t.session_id, t.organization_id, t.organization_id IS NULL
              FROM public.refresh_tokens t WHERE t.token_hash = p_hash $$
        -- @@

        CREATE OR REPLACE FUNCTION app.auth_find_reset_token(p_hash bytea)
        RETURNS TABLE (user_id uuid, organization_id uuid, is_platform boolean) {definer}
        AS $$ SELECT t.user_id, t.organization_id, t.organization_id IS NULL
              FROM public.password_reset_tokens t WHERE t.token_hash = p_hash $$
        -- @@

        CREATE OR REPLACE FUNCTION app.auth_find_mfa_challenge(p_hash bytea)
        RETURNS TABLE (user_id uuid, organization_id uuid, is_platform boolean) {definer}
        AS $$ SELECT c.user_id, c.organization_id, c.organization_id IS NULL
              FROM public.mfa_challenges c WHERE c.token_hash = p_hash $$
        """
    )
    for fn in (
        "auth_find_user(text)",
        "auth_find_refresh_token(bytea)",
        "auth_find_reset_token(bytea)",
        "auth_find_mfa_challenge(bytea)",
    ):
        _x(f"REVOKE ALL ON FUNCTION app.{fn} FROM PUBLIC")
        _x(f"GRANT EXECUTE ON FUNCTION app.{fn} TO {APP}")


def _rls_policies() -> None:
    same_org = "organization_id = app.current_org_id()"
    for table in TENANT_TABLES:
        _x(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        _x(
            f"CREATE POLICY tenant_isolation ON {table} TO {APP}, {WORKER} "
            f"USING ({same_org}) WITH CHECK ({same_org})"
        )
    platform_or_org = f"({same_org} OR (organization_id IS NULL AND app.is_platform()))"
    for table in NULLABLE_TENANT_TABLES:
        _x(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        _x(
            f"CREATE POLICY tenant_isolation ON {table} TO {APP}, {WORKER} "
            f"USING {platform_or_org} WITH CHECK {platform_or_org}"
        )
    owner_only = f"({same_org} AND user_id = app.current_user_id())"
    for table in PRIVATE_TABLES:
        _x(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        _x(
            f"CREATE POLICY owner_only ON {table} TO {APP} USING {owner_only} WITH CHECK {owner_only}"
        )
        _x(
            f"CREATE POLICY tenant_maintenance ON {table} TO {WORKER} "
            f"USING ({same_org}) WITH CHECK ({same_org})"
        )

    _x("ALTER TABLE organizations ENABLE ROW LEVEL SECURITY")
    _x(
        f"CREATE POLICY member_read ON organizations FOR SELECT TO {APP} "
        "USING (id = app.current_org_id() OR app.is_platform())"
    )
    _x(
        f"CREATE POLICY platform_insert ON organizations FOR INSERT TO {APP} WITH CHECK (app.is_platform())"
    )
    _x(
        f"CREATE POLICY member_update ON organizations FOR UPDATE TO {APP} "
        "USING (id = app.current_org_id() OR app.is_platform()) "
        "WITH CHECK (id = app.current_org_id() OR app.is_platform())"
    )
    _x(f"CREATE POLICY worker_read ON organizations FOR SELECT TO {WORKER} USING (true)")

    _x("ALTER TABLE jobs ENABLE ROW LEVEL SECURITY")
    _x(
        f"CREATE POLICY tenant_jobs ON jobs TO {APP} USING {platform_or_org} WITH CHECK {platform_or_org}"
    )
    _x(f"CREATE POLICY worker_jobs ON jobs TO {WORKER} USING (true) WITH CHECK (true)")

    _x("ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY")
    _x(f"CREATE POLICY audit_read ON audit_events FOR SELECT TO {APP} USING {platform_or_org}")
    _x(
        f"CREATE POLICY audit_append ON audit_events FOR INSERT TO {APP} "
        f"WITH CHECK ({same_org} OR organization_id IS NULL)"
    )
    _x(f"CREATE POLICY audit_worker ON audit_events TO {WORKER} USING (true) WITH CHECK (true)")

    _x("ALTER TABLE audit_chain_heads ENABLE ROW LEVEL SECURITY")
    _x(
        f"CREATE POLICY chain_read ON audit_chain_heads FOR SELECT TO {APP} "
        "USING (chain_key = app.current_org_id() OR "
        "(chain_key = '00000000-0000-0000-0000-000000000000'::uuid AND app.is_platform()))"
    )
    _x(
        f"CREATE POLICY chain_worker ON audit_chain_heads TO {WORKER} USING (true) WITH CHECK (true)"
    )


def _grants() -> None:
    _x("REVOKE ALL ON SCHEMA public FROM PUBLIC")
    _x(f"GRANT USAGE ON SCHEMA public, app TO {APP}, {WORKER}")
    for table, privileges in APP_GRANTS.items():
        _x(f"GRANT {privileges} ON {table} TO {APP}")
    for table, privileges in WORKER_GRANTS.items():
        _x(f"GRANT {privileges} ON {table} TO {WORKER}")
    _x(f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO {APP}, {WORKER}")
    _x(
        f"GRANT EXECUTE ON FUNCTION app.current_org_id(), app.current_user_id(), app.is_platform() TO {APP}, {WORKER}"
    )


def _integrity_triggers() -> None:
    # Audit log: append-only. Sealing may fill hash columns exactly once; nothing else changes.
    _x(
        """
        CREATE OR REPLACE FUNCTION app.audit_guard() RETURNS trigger
        LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
        BEGIN
          IF TG_OP = 'DELETE' THEN
            IF current_setting('app.audit_purge', true) = 'on' AND OLD.hash IS NOT NULL THEN
              RETURN OLD;
            END IF;
            RAISE EXCEPTION 'audit_events is append-only' USING ERRCODE = '42501';
          END IF;
          IF OLD.hash IS NOT NULL THEN
            RAISE EXCEPTION 'sealed audit events are immutable' USING ERRCODE = '42501';
          END IF;
          IF NEW.id <> OLD.id
             OR NEW.organization_id IS DISTINCT FROM OLD.organization_id
             OR NEW.occurred_at <> OLD.occurred_at
             OR NEW.actor_user_id IS DISTINCT FROM OLD.actor_user_id
             OR NEW.actor_role IS DISTINCT FROM OLD.actor_role
             OR NEW.actor_ip_prefix IS DISTINCT FROM OLD.actor_ip_prefix
             OR NEW.action <> OLD.action
             OR NEW.resource_type IS DISTINCT FROM OLD.resource_type
             OR NEW.resource_id IS DISTINCT FROM OLD.resource_id
             OR NEW.outcome <> OLD.outcome
             OR NEW.request_id IS DISTINCT FROM OLD.request_id
             OR NEW.details IS DISTINCT FROM OLD.details THEN
            RAISE EXCEPTION 'only seal columns of an audit event may be written' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END
        $$
        -- @@
        CREATE TRIGGER audit_events_guard BEFORE UPDATE OR DELETE ON audit_events
          FOR EACH ROW EXECUTE FUNCTION app.audit_guard()
        -- @@
        CREATE OR REPLACE FUNCTION app.audit_no_truncate() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          RAISE EXCEPTION 'audit_events cannot be truncated' USING ERRCODE = '42501';
        END
        $$
        -- @@
        CREATE TRIGGER audit_events_no_truncate BEFORE TRUNCATE ON audit_events
          FOR EACH STATEMENT EXECUTE FUNCTION app.audit_no_truncate()
        """
    )
    # Retention purge of sealed audit rows: owner-privileged, anchors the chain first.
    _x(
        f"""
        CREATE OR REPLACE FUNCTION app.audit_purge(p_chain uuid, p_before timestamptz)
        RETURNS bigint
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
        DECLARE
          v_last record;
          v_deleted bigint;
        BEGIN
          SELECT e.seal_seq, e.hash INTO v_last
            FROM audit_events e
           WHERE coalesce(e.organization_id, '00000000-0000-0000-0000-000000000000'::uuid) = p_chain
             AND e.hash IS NOT NULL AND e.occurred_at < p_before
           ORDER BY e.seal_seq DESC LIMIT 1;
          IF v_last IS NULL THEN
            RETURN 0;
          END IF;
          UPDATE audit_chain_heads SET anchor_seq = v_last.seal_seq, anchor_hash = v_last.hash
           WHERE chain_key = p_chain AND anchor_seq < v_last.seal_seq;
          PERFORM set_config('app.audit_purge', 'on', true);
          DELETE FROM audit_events e
           WHERE coalesce(e.organization_id, '00000000-0000-0000-0000-000000000000'::uuid) = p_chain
             AND e.hash IS NOT NULL AND e.seal_seq <= v_last.seal_seq;
          GET DIAGNOSTICS v_deleted = ROW_COUNT;
          PERFORM set_config('app.audit_purge', 'off', true);
          RETURN v_deleted;
        END
        $$
        -- @@
        REVOKE ALL ON FUNCTION app.audit_purge(uuid, timestamptz) FROM PUBLIC
        -- @@
        GRANT EXECUTE ON FUNCTION app.audit_purge(uuid, timestamptz) TO {WORKER}
        """
    )
    # Documents: only soft-deleted documents can be purged, and never under legal hold.
    _x(
        """
        CREATE OR REPLACE FUNCTION app.documents_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.legal_hold THEN
            RAISE EXCEPTION 'document is under legal hold' USING ERRCODE = '42501';
          END IF;
          IF OLD.status <> 'deleted' THEN
            RAISE EXCEPTION 'documents must be soft-deleted before purge' USING ERRCODE = '42501';
          END IF;
          RETURN OLD;
        END
        $$
        -- @@
        CREATE TRIGGER documents_guard BEFORE DELETE ON documents
          FOR EACH ROW EXECUTE FUNCTION app.documents_guard()
        """
    )


def _indexes_and_late_constraints() -> None:
    _x(
        "CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    _x(
        "ALTER TABLE documents ADD CONSTRAINT fk_documents_current_version "
        "FOREIGN KEY (organization_id, current_version_id) "
        "REFERENCES document_versions (organization_id, id) ON DELETE SET NULL (current_version_id)"
    )
    _x("CREATE INDEX ix_documents_tags ON documents USING gin (tags)")


def security_upgrade() -> None:
    _context_functions()
    _auth_lookup_functions()
    _rls_policies()
    _grants()
    _integrity_triggers()
    _indexes_and_late_constraints()


def security_downgrade() -> None:
    _x("DROP FUNCTION IF EXISTS app.audit_purge(uuid, timestamptz)")
    _x("DROP FUNCTION IF EXISTS app.audit_guard() CASCADE")
    _x("DROP FUNCTION IF EXISTS app.audit_no_truncate() CASCADE")
    _x("DROP FUNCTION IF EXISTS app.documents_guard() CASCADE")
    for fn in (
        "auth_find_user(text)",
        "auth_find_refresh_token(bytea)",
        "auth_find_reset_token(bytea)",
        "auth_find_mfa_challenge(bytea)",
        "current_org_id()",
        "current_user_id()",
        "is_platform()",
    ):
        _x(f"DROP FUNCTION IF EXISTS app.{fn}")
    _x("DROP SCHEMA IF EXISTS app")
