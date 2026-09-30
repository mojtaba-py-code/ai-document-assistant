# Chapter 2 — Data, tenancy and identity (Levels 5–8)

## Level 5 — PostgreSQL database architecture

**Objective.** A schema that is correct, migratable and hard to misuse.

**Files.** `db/models.py` (22 tables), `db/base.py` (naming convention), `db/session.py`
(engines, RLS-scoped sessions), `migrations/versions/0001_initial_schema.py`,
`migrations/docassist_migration_security.py`.

**Theory.** Constraints are security controls: a `CHECK`, a unique key or a foreign key is
enforced for every client, every bug and every future code path.

**Key code.**
* CHECK constraints for every enum-like column (roles, statuses, classifications), slug
  formats, the platform-admin/no-organisation rule, lowercase emails, tag limits.
* **Composite foreign keys** `(organization_id, parent_id) → parent(organization_id, id)`
  make it impossible to link, say, a document of organisation A to a department of B.
* Deletes default to `RESTRICT`; only pure child data (chunks, embeddings, grants) cascades.
* Per-connection `statement_timeout`, `lock_timeout`, `idle_in_transaction_session_timeout`.
* `hnsw` index on embeddings and a GIN index on a *generated* weighted `tsvector`.
* Migrations run as the schema owner only; `alembic check` shows zero drift.

**Common vulnerabilities avoided.** SQL injection (bound parameters everywhere, bandit in
CI), accidental cascade deletes, orphaned cross-tenant references, runaway queries.

**Run / verify.**

```bash
DOCASSIST_DATABASE__MIGRATION_URL=postgresql+asyncpg://owner@host/db alembic upgrade head
alembic check
```

**Security review**

| | |
|---|---|
| Attack surface | SQL built by the application; migration credentials |
| Threats | SQL injection, data corruption, tenant cross-linking |
| Controls | ORM/bound parameters, constraints, composite FKs, timeouts, owner-only DDL |
| Weaknesses | a superuser can still read and edit everything |
| Improvements | pgaudit for DBA sessions; column encryption for the most sensitive fields |
| Tests | `test_composite_fk_blocks_cross_tenant_reference`, migration round-trip in CI |

## Level 6 — Multi-tenant architecture

**Objective.** A user of organisation A must never read documents, embeddings, metadata,
conversations, search results or audit records of organisation B — enforced server-side.

**Strategy.** Shared schema + **PostgreSQL row-level security** (ADR-02).

**Key code.**
* `DbContext(org_id, user_id, platform)` is attached to each session; an `after_begin`
  listener runs `set_config('app.org_id', …, true)` (transaction-local) at the start of every
  transaction, so pooled connections never carry a previous tenant's context.
* Policies: `organization_id = app.current_org_id()` on every tenant table;
  conversations/messages additionally require `user_id = app.current_user_id()`;
  NULL-organisation rows (platform operators) need `app.is_platform()`.
* `verify_least_privilege` refuses to start production if the API role is superuser, has
  `BYPASSRLS`, can create roles/databases or owns tables.
* Content tables have **no** platform clause: platform operators manage organisations but
  cannot read tenant documents through the application.

**Why it matters.** Most multi-tenant breaches are one forgotten `WHERE tenant_id = ?`.
With RLS that bug returns nothing instead of another customer's data.

**Security review**

| | |
|---|---|
| Attack surface | every SQL statement issued by the API/worker |
| Threats | missing tenant filter, context leaking across pooled connections, tenant hopping via UPDATE |
| Controls | RLS USING + WITH CHECK, transaction-local context, fail-closed helpers, composite FKs |
| Weaknesses | SECURITY DEFINER login lookups are a deliberate, narrow bypass |
| Improvements | database-per-tenant option for regulated customers (same code, different DSN) |
| Tests | `tests/integration/test_rls_isolation.py` (no context sees nothing, cross-tenant write/move blocked, pooled-connection leak test) |

## Level 7 — Authentication

**Files.** `identity/auth.py`, `security/passwords.py`, `security/tokens.py`,
`security/totp.py`, `api/routers/auth.py`.

**Theory.** Passwords are verified with a memory-hard KDF; the proof of login is a short-lived
signed token *plus* server-side session state, so revocation is immediate.

**Key code.**
* Argon2id; unknown accounts burn the same hashing work (`PasswordService.verify(None, …)`).
* Identical error for unknown email / wrong password / disabled / locked.
* Exponential lockout + per-IP and per-account GCRA rate limits.
* Access JWT (HS256 pinned, `kid` rotation, 10 min) carrying the session id; every request
  re-checks the session, `token_version`, role and organisation status.
* Refresh tokens: opaque, stored as peppered HMAC, single use; **reuse revokes the session**.
* Cookie transport: `__Host-` prefix, `HttpOnly`, `Secure`, `SameSite=Strict`, plus the
  `X-CSRF-Protection: 1` header requirement.
* TOTP MFA with an encrypted secret, replay protection and an attempt-limited challenge.
  Wrong codes count toward the same lockout as wrong passwords, failure counters are only
  cleared after the *whole* login succeeds, and a new login invalidates older challenges.
  Enrolment requires the current password; admins can reset a user's MFA (audited).
* Every in-session password check (change password, enrol/disable MFA) is rate-limited and
  counts toward lockout, and validates the new password first — so a stolen access token is
  not a password oracle.
* Password reset: 256-bit single-use token, 30-minute TTL, all sessions revoked afterwards,
  every other outstanding reset link consumed; the "forgot password" work runs in the
  background so response time does not reveal whether an account exists.

**Run / verify.**

```bash
python -m pytest tests/integration/test_auth_api.py tests/unit/test_security_tokens.py -q
```

**Security review**

| | |
|---|---|
| Attack surface | login, MFA, refresh, reset endpoints; tokens in transit and at rest |
| Threats | stuffing, enumeration, token theft/replay, JWT forgery, CSRF on refresh |
| Controls | Argon2id, lockout, limits, pinned JWT verification, rotation + reuse detection, cookie hardening |
| Weaknesses | users without MFA remain phishable |
| Improvements | WebAuthn/passkeys; org policy "MFA required" |
| Tests | lockout, reuse detection, CSRF header, reset single use, MFA replay, forged tokens (`alg=none`, wrong key, tampered) |

## Level 8 — Authorization and RBAC

**Files.** `authz/permissions.py`, `authz/principal.py`, `authz/policy.py`, `api/deps.py`.

**Theory.** RBAC answers "may this *kind* of user try this *kind* of action"; ABAC answers
"may this user touch *this* document". Both must pass (complete mediation), default deny.

**Key code.**
* `ROLE_PERMISSIONS` — five roles; auditors have no content permissions (separation of duties).
* `require(Permission.X)` dependency on every route; denials are audited (`authz.denied`).
* Document policy: classification is a hard ceiling (even for owners), `allowed_roles` a hard
  filter, then PUBLIC/INTERNAL | owner | CONFIDENTIAL in the user's department | active grant.
  RESTRICTED requires owner or grant. Org admins *manage* everything but do not *read*
  RESTRICTED content without a grant.
* One definition, **two compilations**: `can_read/can_manage/can_list` (Python) and
  `readable_clause/manageable_clause/listable_clause` (SQL embedded in queries).

**Security review**

| | |
|---|---|
| Attack surface | every route and every document query |
| Threats | IDOR, privilege escalation, Python/SQL policy drift, existence leaks |
| Controls | RBAC gate, ACL-in-SQL, 404 for invisible objects, assignable-role matrix |
| Weaknesses | admins can grant themselves access (audited, not preventable) |
| Improvements | four-eyes approval for RESTRICTED grants |
| Tests | `tests/integration/test_policy_equivalence.py` (randomised differential test), `tests/security/test_privilege_escalation.py` |
