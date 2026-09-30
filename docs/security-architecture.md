# Security Architecture

Security is not a layer of this system; it is a property of every layer. This page lists
the controls per layer, the principle each one implements, and where it lives in the code.

## 1. Defence in depth — one request, many independent checks

A user asking "What is our annual leave policy?" passes through these controls, each of
which would stop a different failure of the others:

```
browser ── CSP + Trusted Types (no HTML sinks) ── access token in memory only
  │
edge ───── TLS, trusted-host check, body-size limit (header AND streamed bytes)
  │
API ────── request ID, security headers, JWT (pinned alg) + live session check
  │         RBAC gate: assistant:use
  │         rate limits: per user, per organisation; monthly token budget
  │
service ── question validated, sanitised, injection-scanned (flag, not trust)
  │         retrieval SQL = tenant + clearance + allowed_roles + department/grants + current version
  │
database ─ row-level security (fail-closed) under a role without BYPASSRLS
  │         composite FKs make cross-tenant references impossible
  │
context ── sources escaped and delimited with a per-request nonce; risky chunks excluded
  │
gateway ── classification ceiling decides which provider may see the data at all
  │         PII pseudonymised for external providers; token caps; circuit breaker
  │
model ──── no tools, no network, JSON schema output
  │
output ─── schema validation → citation quotes verified against sources → canary check
            → URLs/images/HTML stripped → secrets redacted → UI renders as text
```

## 2. Controls by layer

### 2.1 Identity
| Control | Code |
|---|---|
| Argon2id (memory-hard) password hashing, NFKC-normalised, rehash on parameter change | `security/passwords.py` |
| NIST-style policy: length, blocklist, not derived from email/name | `security/passwords.py` |
| Timing-equalised login for unknown accounts | `PasswordService.verify(None, ...)` |
| Exponential account lockout + per-IP and per-account GCRA limits | `identity/auth.py`, `cache/ratelimit.py` |
| Access JWT: HS256 pinned, rotation by `kid`, required claims, 10 min TTL | `security/tokens.py` |
| Session row checked per request (revocation is immediate) | `AuthService.authenticate` |
| Refresh tokens: opaque, peppered HMAC at rest, single use, reuse ⇒ session kill | `AuthService.refresh` |
| `__Host-` HttpOnly SameSite=Strict cookie + custom CSRF header | `api/routers/auth.py` |
| TOTP MFA: encrypted secret, replay-proof step tracking, attempt-limited challenge | `security/totp.py`, `AuthService.complete_mfa` |
| Password reset: 256-bit token, 30 min, single use, all sessions revoked | `AuthService.reset_password` |

### 2.2 Authorization
| Control | Code |
|---|---|
| RBAC matrix, default deny | `authz/permissions.py` |
| Document ACL (classification ceiling, allowed roles, department, owner, grants with expiry) | `authz/policy.py` |
| Same policy compiled to SQL and embedded in every query | `readable_clause` / `listable_clause` / `manageable_clause` |
| Invisible resources → 404, never 403 | services |
| Separation of duties: auditors see activity, not content; admins manage, don't read RESTRICTED | `authz/permissions.py`, `authz/policy.py` |
| Anti-escalation rules for role/clearance assignment | `identity/admin.py` |

### 2.3 Data layer
| Control | Code |
|---|---|
| Row-level security on every tenant table, transaction-local context, fail closed | `migrations/docassist_migration_security.py`, `db/session.py` |
| Three roles: owner (DDL), app (DML, no audit edits, no hard deletes), worker | same |
| Start-up refusal if the API role is superuser / BYPASSRLS / table owner | `db/session.verify_least_privilege` |
| Composite `(organization_id, id)` foreign keys | `db/models.py` |
| `SECURITY DEFINER` login lookups return identifiers only, pinned `search_path` | migration |
| Append-only audit table (privileges + triggers, TRUNCATE blocked) | migration |
| Legal hold and soft-delete-before-purge enforced by trigger | migration |
| Statement, lock and idle-in-transaction timeouts per connection | `db/session.create_engine` |

### 2.4 Files and storage
| Control | Code |
|---|---|
| Streaming size limit, SHA-256, private temp files (0600) | `documents/service.py` |
| Magic-byte sniffing; extension must match; macro-enabled OOXML rejected | `documents/validation.py` |
| Zip inspection without extraction (entries, sizes, ratios, names, nesting, encryption) | `documents/validation.py` |
| Active-content scanner (PDF JS/launch/embedded files, OOXML macros/OLE/external rels/DDE) + ClamAV | `documents/scanning.py` |
| Quarantine instead of processing | `documents/service.py` |
| Envelope encryption, context-bound AEAD, atomic writes, server-generated keys | `security/crypto.py`, `documents/storage.py` |
| Downloads: `attachment`, `nosniff`, `CSP: sandbox`, `no-store` | documents router |

### 2.5 Parsing
| Control | Code |
|---|---|
| Separate interpreter (`python -I`), stdin/stdout only, no secrets in env | `ingestion/sandbox.py` |
| POSIX rlimits (memory, CPU, file size 0, fds), timeout + kill, output cap | `ingestion/sandbox_child.py` |
| No network namespace on Linux (best effort), no egress in containers | same, compose/k8s network policy |
| defusedxml, read-only openpyxl, page/row/cell caps | parsers |
| Output validated before use | `ingestion/model.py` |

### 2.6 AI layer
| Control | Code |
|---|---|
| Classification ceiling for external providers (LLM and embeddings) | `llm/gateway.py`, `ingestion/pipeline.py` |
| PII pseudonymisation with per-request salted placeholders | `core/redaction.Pseudonymizer` |
| Spotlighting: nonce-delimited, escaped, labelled untrusted sources | `rag/context.py` |
| Injection scoring at ingestion and for questions | `ingestion/injection.py` |
| No tools in the answer path; read-only principal-bound tools in agent mode | `rag/service.py`, `rag/tools.py` |
| Citation verification (verbatim / fuzzy quote match) | `rag/citations.py` |
| Output guard: canary, URLs, images, HTML, secrets, length | `rag/guard.py` |
| Deterministic answers for date queries | `rag/service.py`, `intelligence/deadlines.py` |
| Budgets, rate limits, circuit breaker, token caps, usage/cost records | `llm/gateway.py`, `audit/usage.py` |

### 2.7 Network and HTTP
| Control | Code |
|---|---|
| Security headers (CSP with Trusted Types, HSTS in prod, frame/COOP/CORP, no-store for API) | `api/middleware.py` |
| Trusted hosts; CORS only for configured origins | `api/app.py` |
| RFC 9457 errors without internals; validation errors never echo input | `api/problems.py` |
| SSRF guard for every outbound request (allowlist, IP classes, DNS pinning, redirects, caps) | `security/ssrf.py` |
| `X-Forwarded-For` honoured only from trusted proxy CIDRs | `api/deps.client_ip` |

### 2.8 Operations
| Control | Code |
|---|---|
| Production configuration validator (refuses unsafe settings, weak/placeholder secrets) | `core/config.py` |
| Secrets from files (`*_FILE`), never baked into images | `core/config.load_settings`, compose |
| Structured logs with redaction; no prompts/documents/tokens in logs | `core/logging.py` |
| Tamper-evident audit chain + verification API | `audit/service.py` |
| Hardened containers (non-root, read-only, caps dropped, no-new-privileges, segmented networks) | `Dockerfile`, `compose.yaml`, `deployment/` |
| CI: lint, types, import contracts, tests on PostgreSQL, bandit, pip-audit, gitleaks, Trivy, CodeQL | `.github/workflows/` |

## 3. Security principles and where they are applied

| Principle | Applied where |
|---|---|
| **Zero Trust** | every request re-authenticated against a live session; every query re-authorised; model output, parser output and provider responses are untrusted |
| **Least Privilege** | three DB roles; API role cannot edit audit or hard-delete; RAG model has no tools; parser has no secrets; containers drop all capabilities |
| **Defence in Depth** | RBAC + ACL-in-SQL + RLS + composite FKs; spotlighting + injection scoring + no tools + output guard + citation checks |
| **Secure by Design** | tenant isolation and document ACL are schema/query properties, not UI filters |
| **Fail Securely** | missing RLS context ⇒ no rows; Redis down ⇒ local rate limiting; scanner down ⇒ upload rejected; invalid model output ⇒ error, not partial answer |
| **Default Deny** | permissions granted explicitly per role; egress allowlist; extensions allowlist; CSP `default-src 'self'` / `'none'` for API |
| **Data Minimisation** | IP addresses stored as /24 or /48 prefixes; audit stores lengths/hashes of queries; LLM receives only the selected chunks, pseudonymised |
| **Separation of Duties** | auditors vs admins vs platform operators; admins cannot silently read RESTRICTED content |
| **Explicit Authorization** | `require(Permission.X)` on every route; `can_manage` / `can_read` on every object |
| **Complete Mediation** | no cached authorization decisions; the ACL runs in each query; sessions checked per request |
| **Secure Defaults** | offline AI providers by default (nothing leaves the host), docs off in production, cookies Secure, rate limits on |
