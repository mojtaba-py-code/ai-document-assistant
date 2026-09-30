# Chapter 7 — Operating it (Levels 25–28)

## Level 25 — Redis and caching

**Files.** `cache/redis.py`, `rag/service.py` (answer cache), `search/service.py`
(query-embedding cache), `audit/usage.py` (budget cache).

**Rules.**
* Keys are `prefix:purpose:scope:sha256(parts)` — user text never becomes a key verbatim,
  and every content-bearing key is scoped to the organisation.
* Values derived from documents (answers, query embeddings) are **encrypted** with the key
  ring before they reach Redis; a Redis snapshot reveals nothing readable.
* The answer cache key includes the organisation, the principal's access fingerprint, the
  question, the **exact set of authorised chunk ids and hashes**, the model and the prompt
  version — so a hit can only serve an answer built from context the caller is allowed to
  see. Answers that used RESTRICTED sources are never cached. Result sets are never cached
  (permissions change).
* Every operation has a short socket timeout; failures degrade to cache misses.
* Deployment: ACL user limited to `~docassist:*` and no dangerous commands,
  `maxmemory` with `volatile-lru` (all keys have TTLs), no persistence required.

**Security review**

| | |
|---|---|
| Attack surface | cache keys and values, the Redis network endpoint |
| Threats | cross-tenant cache poisoning/leaks, stale authorization, snapshot disclosure |
| Controls | hashed scoped keys, context-bound encryption, exact-context answer keys, ACL user, internal network |
| Weaknesses | cache timing reveals whether the same authorised context was asked recently |
| Improvements | per-tenant key prefixes with separate ACL users for the largest tenants |
| Tests | answer-cache tests in `tests/integration/test_rag_service.py` |

## Level 26 — Rate limiting

**Files.** `cache/ratelimit.py`, `core/config.py` (`RateLimitSettings`), callers in auth,
documents, search, RAG, agent tools and exports.

**Algorithm.** GCRA (one timestamp per key, exact bursts) in a single atomic Lua script on
Redis time, shared by all replicas. If Redis is down an in-process GCRA takes over — limits
become per replica, never disappear.

**Buckets.** login per IP and per account, refresh per session, password reset per IP and
per account, API per user, uploads, search, LLM per user **and** per organisation, exports,
agent tool calls, signed export downloads per IP. The client IP is taken from
`X-Forwarded-For` only when the peer is a trusted proxy.

**Security review**

| | |
|---|---|
| Attack surface | every expensive or sensitive endpoint |
| Threats | brute force, credential stuffing, scraping, cost exhaustion, limit bypass via spoofed IPs or email case variants |
| Controls | GCRA, normalised identities, trusted-proxy logic, fallback limiter |
| Weaknesses | distributed attacks from many IPs against many accounts |
| Improvements | adaptive CAPTCHA/risk scoring at the edge |
| Tests | `tests/unit/test_core_config_and_auth_primitives.py` (burst, fallback, disabled) |

## Level 27 — Audit logging

**Files.** `audit/service.py`, migration triggers, `api/routers/audit.py`, `jobs/maintenance.py`.

**Design.** Audit rows are written in the same transaction as the action they describe
(denials and failures in their own transaction). Details are size-capped and PII/secret
redacted; queries are stored as length + hash (+ redacted text by setting). The API role can
only INSERT; triggers block UPDATE/DELETE/TRUNCATE except sealing unsealed rows. The worker
seals rows into per-organisation **HMAC-SHA256 chains** whose key is not in the database;
`POST /api/v1/audit/verify` recomputes the chain. Retention purges go through a SECURITY
DEFINER function that first anchors the chain.

**Events.** logins/failures/lockouts, MFA, token reuse, password changes/resets, uploads,
quarantines, downloads, views, permission and grant changes, deletions, searches, questions,
tool calls, extractions, summaries, exports and their downloads, admin actions, retention
purges, dead-lettered jobs, authorization denials.

**Security review**

| | |
|---|---|
| Attack surface | audit table, sealing worker, verification API |
| Threats | tampering, deletion, secrets in audit details, repudiation |
| Controls | privileges + triggers + HMAC chain + verification; redaction |
| Weaknesses | a superuser could delete *unsealed* rows in the seconds before sealing |
| Improvements | ship chain heads to external WORM storage |
| Tests | `tests/integration/test_audit_chain.py` (tamper and deletion detection by a superuser) |

## Level 28 — Observability

**Files.** `core/logging.py`, `api/middleware.py`, `observability/metrics.py`,
`api/routers/health.py`, `api/routers/admin.py` (`/admin/health`).

* Structured JSON logs (structlog) with `request_id` on every line; a redaction processor
  masks secret-like values and sensitive keys; prompts, documents and tokens are never logged.
* Prometheus metrics with low-cardinality labels: HTTP rate/latency, auth events, rate-limit
  rejections, degraded modes, LLM requests/tokens/cost/latency, embedding and retrieval
  latency, ingestion outcomes and duration, jobs and queue depth, security detections,
  citation outcomes. `/metrics` requires a bearer token and is off in production without one.
* `/health/live` and `/health/ready` reveal only `ok/degraded/unavailable`; detailed health
  (database, Redis, vector store, circuit states, queue depth) is an admin endpoint.
* Alert rules are listed in [operations.md](../operations.md#7-monitoring-and-alerting).

**Security review**

| | |
|---|---|
| Attack surface | logs, metrics endpoint, health endpoints |
| Threats | secret leakage in logs, reconnaissance via health/metrics |
| Controls | redaction, token-protected metrics, minimal public health output |
| Weaknesses | logs of third-party libraries depend on their own hygiene (their levels are raised) |
| Improvements | OpenTelemetry traces with the same redaction rules |
| Tests | logging redaction and header tests |
