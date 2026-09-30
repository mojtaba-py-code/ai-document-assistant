# Operations Runbook

Backup and recovery, data retention, key rotation, incident response, scaling and
monitoring for the AI Document Assistant.

## 1. Components and state

| Component | State | Source of truth? | Backed up |
|---|---|---|---|
| PostgreSQL | tenants, users, documents metadata, chunks, embeddings, extracted fields, conversations, jobs, audit | **yes** | continuous (WAL) + daily base backup |
| Object storage (`DOCASSIST_STORAGE__ROOT`) | encrypted document versions and exports | **yes** | daily snapshot / versioned bucket |
| Encryption keys (KEKs), JWT key, token pepper, audit HMAC key | secrets | **yes** | in the secret manager, *separately* from data backups |
| Redis | rate-limit counters, encrypted caches | no (disposable) | not backed up |
| Qdrant (optional) | derived vector index | no — rebuilt from PostgreSQL | not required |

Keeping keys outside the data backups is deliberate: a stolen backup without the KEKs does
not reveal document bodies, MFA secrets or refresh tokens.

## 2. Backups

### PostgreSQL
* **Continuous WAL archiving** (`archive_mode=on`, `archive_command` to object storage, or a
  managed service's PITR) — RPO ≈ minutes.
* **Daily base backup** with `pg_basebackup` (or managed snapshots); retain 35 days.
* **Weekly logical dump** (`pg_dump --format=custom`) for long-term / cross-version restores;
  retain 12 weeks. See `scripts/backup.sh`.
* Backups are encrypted at rest by the storage layer and access is limited to the backup role.

### Object storage
* Versioned bucket (or daily filesystem snapshot) with the same 35-day retention.
* Blobs are already client-side encrypted; the bucket should additionally use provider
  encryption (SSE-KMS or equivalent).

### Consistency
Blobs are written **before** the database row that references them is committed, and blobs
are deleted **after** the rows are purged. Therefore a database restored to time *T* may
reference only blobs that already existed at *T* — restore the object store to a point
**at or after** the database restore point. Orphan blobs (present in storage, unknown to
the database) are harmless and are removed by the retention worker.

### Frequency and targets
| Metric | Target |
|---|---|
| RPO (data loss) | ≤ 15 minutes (WAL) |
| RTO (service restored) | ≤ 4 hours for a full region loss; ≤ 1 hour for a database restore |

## 3. Restore procedure (tested quarterly)

1. Provision an empty PostgreSQL 16 with pgvector and the three roles
   (`deployment/postgres/initdb/`).
2. Restore the base backup and replay WAL to the target time (or `pg_restore` a dump).
3. Restore the object store to the same or a later point in time.
4. Restore secrets (KEKs, JWT keys, pepper, audit key) from the secret manager.
5. Run `docassist migrate` (no-op if already at head).
6. Start one worker; it seals any unsealed audit rows and reclaims expired job leases.
7. Verify: `docassist audit-verify` for every organisation, `GET /health/ready`, open three
   random documents per tenant, run one search and one question.
8. If Qdrant is used, enqueue a full re-sync (`vector_sync_document` for all documents) or
   rebuild the collection from `chunk_embeddings`.

`scripts/restore.sh` automates steps 2–3 for the Compose deployment and is used for restore drills.

## 4. Data retention

Configured with `DOCASSIST_RETENTION__*`; organisations may shorten (never extend beyond
policy) some values in their settings. The worker enforces retention periodically:

| Data | Default | Mechanism |
|---|---|---|
| Conversations and messages | 90 days | deleted by the retention job |
| Soft-deleted documents | 30 days grace, then purged (blobs first, then rows) | retention job; **never** while `legal_hold` |
| Chunks / embeddings / extracted fields of a deleted document | removed **immediately** at deletion | same transaction as the delete |
| Exports | 24 hours | blob deleted, row marked expired |
| Finished jobs | 30 days | retention job |
| LLM usage records | 400 days | retention job |
| Audit events | 7 years | `app.audit_purge` anchors the chain, then deletes sealed rows |
| Sessions / refresh / reset / MFA challenge rows | after expiry | retention job |
| Backups | 35 days (PITR), 12 weeks (logical) | backup tooling |

Right to erasure: delete the user's documents and conversations through the API; the user
record is disabled (not deleted) so audit history remains attributable; personal fields can
be pseudonymised by an administrator process if required by policy.

## 5. Key and secret rotation

| Secret | Rotation | Procedure |
|---|---|---|
| JWT signing key | 90 days | add the new key as `security.jwt_signing_key`, move the old one to `security.jwt_previous_signing_keys`, deploy; remove the old key after the refresh window (12 h) |
| Encryption KEK | yearly or on suspicion | add `kid2:<key>` to `security.encryption_keys`, set `active_encryption_key_id=kid2`, deploy. New objects use kid2; old objects stay readable. Remove kid1 only after re-encrypting old objects (future `rewrap` job) |
| Token pepper | on suspicion | changing it invalidates all refresh/reset/MFA-challenge tokens (users sign in again) |
| Audit HMAC key | never casually | rotating breaks verification of old chains; anchor the chain first (`audit_purge` anchors) and archive the old key with the chain export |
| Database role passwords | 90 days | `ALTER ROLE ... PASSWORD`, update secret files, rolling restart |
| LLM / embedding API keys | per provider policy | update secret, rolling restart |

`docassist init-env` generates correctly formatted values for development; in production use
the secret manager and mount values as files (`DOCASSIST_*_FILE`).

## 6. Incident response

| Scenario | Immediate actions |
|---|---|
| Suspected account takeover | disable the user (`PATCH /api/v1/users/{id}` status=disabled) — sessions die immediately; review `auth.*` audit events for the user; force password reset |
| Refresh-token reuse alerts | the session is already revoked; check the anonymised IP prefixes and user agent in `auth_sessions`; contact the user |
| Leaked JWT signing key | rotate the key **without** keeping the old one as previous — all access tokens become invalid at once |
| Leaked KEK | add a new active key, re-encrypt objects, then remove the leaked key; treat stored blobs as potentially exposed |
| Malicious document detected | it is quarantined automatically; review `document.quarantined` events; delete; search audit for downloads of the same hash |
| Prompt-injection campaign | `security_events_total{kind="injection_detected"}` spikes; review chunks with high `injection_score`; lower `retrieval.injection_exclude_threshold` |
| LLM provider outage | circuit breaker opens; answers fail fast with a clear message; search keeps working; optionally switch `llm.provider` to the local provider |
| Audit verification failure | preserve the database (snapshot), export the chain, compare with the last external anchor, escalate as a security incident |
| Tenant data exposure suspected | disable affected tenants (`status=suspended`), export audit events, verify the chain, review RLS policies with `\d+` and the policy-equivalence tests |

Every incident ends with: timeline from audit events, root cause, regression test, and a
threat-model update.

## 7. Monitoring and alerting

Prometheus metrics (`/metrics`, token-protected):

| Alert | Expression (sketch) |
|---|---|
| API error rate | `sum(rate(docassist_http_requests_total{status=~"5.."}[5m])) / sum(rate(docassist_http_requests_total[5m])) > 0.02` |
| Login attack | `rate(docassist_auth_events_total{event="login",outcome!="success"}[5m]) > 5` |
| Refresh-token reuse | `increase(docassist_security_events_total{kind="refresh_token_reuse"}[15m]) > 0` |
| Injection spike | `increase(docassist_security_events_total{kind="injection_detected"}[1h]) > 50` |
| Ingestion backlog | `docassist_job_queue_depth{status="queued"} > 500` for 15 m |
| Dead letters | `docassist_job_queue_depth{status="dead"} > 0` |
| LLM failures | `rate(docassist_llm_requests_total{status!="ok"}[5m]) > 0.1` |
| Spend | `increase(docassist_llm_cost_usd_total[1d]) > <budget>` |
| Degraded mode | `increase(docassist_degraded_mode_total[10m]) > 0` |

Logs are JSON with `request_id` on every line; ship them to the SIEM. Audit events are the
authoritative security record; the logs are for operations.

## 8. Scaling playbook

* **API** — scale replicas on CPU/latency; keep `database.pool_size × replicas` below the
  PostgreSQL connection budget (or front PostgreSQL with PgBouncer in transaction mode —
  compatible because RLS context is transaction-local).
* **Worker** — scale on `job_queue_depth{status="queued"}`; parser concurrency per worker is
  `parser.max_concurrency` (CPU bound).
* **Vector search** — tune `retrieval.hnsw_ef_search`; move to Qdrant (`retrieval.backend=qdrant`)
  beyond tens of millions of chunks; PostgreSQL remains the system of record.
* **LLM** — raise per-org budgets deliberately; prefer the fast tier for classification,
  reranking and extraction.
