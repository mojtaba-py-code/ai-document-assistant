# Final Enterprise Security Audit (Level 36)

**Date:** 2026-09-30 · **Version audited:** 1.0.0 · **Result:** no open Critical or High
issues; all 6 Medium and 7 Low findings fixed and locked with regression tests.

## 1. Scope and method

**Scope:** the whole platform — authentication, authorization, tenant isolation, API, database,
files and parsers, storage and cryptography, search and vector stores, RAG and the LLM
gateway, the agent and its tools, exports, audit, workers, retention, configuration,
containers and CI.

**Method:**
1. Threat modelling ([threat-model.md](threat-model.md)): STRIDE per component, OWASP API
   Top 10, OWASP Top 10, OWASP LLM Top 10, 28 registered threats.
2. Two independent adversarial code reviews, each limited to **verified** findings: every
   finding had to be demonstrated by a running reproduction (a failing test or script) or an
   exact code path. Reviewers had no write access to the code.
   * Review A — file handling end to end (upload, scanning, sandbox, parsers, storage,
     crypto, SSRF, exports, retention, worker).
   * Review B — identity and administration, the AI paths (RAG, gateway, agent, tools) and
     search authorization.
3. Differential testing of the access policy (Python vs SQL) over randomised organisations.
4. Automated gates: 1,700+ tests on real PostgreSQL as the least-privileged role, ruff,
   mypy strict, import-linter architecture contracts, bandit, pip-audit, gitleaks, Trivy,
   CodeQL.
5. An end-to-end run of the real stack (migrate → seed → serve → HTTP scenarios and browser
   login/assistant) including the brief's adversarial prompts.

## 2. Summary

| Severity | Found | Fixed | Open |
|---|---|---|---|
| Critical | 0 | — | 0 |
| High | 0 | — | 0 |
| Medium | 6 | 6 | 0 |
| Low | 7 | 7 (1 mitigated, see L3) | 0 |

Areas both reviewers checked and found sound include: JWT verification and session binding,
refresh rotation and reuse detection, admin anti-escalation rules, platform/tenant
separation, RLS on every tenant table, append-only audit with HMAC chain, authorization
inside every keyword/vector/hydration query, Qdrant re-verification, spotlighted prompts,
RESTRICTED routing, answer-cache keys, agent tool scoping, IDOR/404 behaviour for documents,
duplicate-detection scoping, quarantine handling, upload validation (sniffing, zip
inspection, XXE), filename/`Content-Disposition` hygiene, CSV injection, envelope encryption
(unique nonces, AAD binding), the SSRF guard, the parser sandbox, export links, retention and
legal hold, worker fencing.

## 3. Findings

Each finding lists Issue · Impact · Attack scenario · Fix · Verification.

### M1 — MFA codes could be brute-forced across fresh challenges (Medium, fixed)
* **Issue:** a correct password reset the failure counters before the MFA step; MFA failures
  counted only against the single challenge.
* **Impact:** an attacker who knows the password (credential stuffing) could guess TOTP codes
  indefinitely (~10 %/day success at the default limits).
* **Attack:** log in → 5 wrong codes → log in again for a new challenge → repeat.
* **Fix:** failure counters are cleared only after a *complete* authentication; wrong codes
  count toward the exponential lockout; a per-account limit on `/mfa/verify`; issuing a
  challenge invalidates earlier ones; password change/reset invalidates open challenges.
* **Verification:** `tests/security/test_review_findings.py::test_m1_*`.

### M2 — An authenticated session was an unthrottled password oracle (Medium, fixed)
* **Issue:** `/auth/password/change` returned 403 for a wrong current password but 422 for a
  correct one with a weak new password; no lockout; `/mfa/disable` had distinct messages and
  no audit.
* **Impact:** a 10-minute token theft could be turned into the real password.
* **Fix:** the new password is validated first; re-authentication is rate-limited per account
  and counts toward lockout; `/mfa/disable` returns one generic error and audits failures.
* **Verification:** `test_m2_stolen_session_is_not_a_password_oracle`,
  `test_m2_mfa_disable_errors_are_generic_and_audited`.

### M3 — Embeddings ignored the organisation's AI ceiling and carried raw PII (Medium, fixed)
* **Issue:** ingestion checked only the deployment ceiling; external embedders received raw
  chunk text and raw queries.
* **Impact:** an organisation that lowered its external ceiling to INTERNAL still had
  CONFIDENTIAL text (with personal data) sent to an external embedding API.
* **Fix:** ingestion uses the stricter of deployment and organisation ceilings (re-checked in
  the commit transaction); personal data is redacted from text and queries sent to external
  embedders; lowering the ceiling deletes stored embeddings above it in the same transaction.
* **Verification:** `test_l6_m3_lowering_the_org_ceiling_applies_immediately`,
  `test_m3_queries_to_an_external_embedder_are_redacted`.

### M4 — Citation check accepted reworded quotes (Medium, fixed)
* **Issue:** the fuzzy matcher compared bags of words (85 % overlap, order ignored) and the
  UI showed the model's wording as the verified quote.
* **Impact:** "Beta pays Acme" was accepted against "Acme pays Beta"; "is **not** liable"
  against "is liable" — a fabricated meaning shown as verified evidence.
* **Fix:** a shared order-aware matcher (`core/quotes.py`): exact contiguous match, or an
  in-order alignment whose only differences are spelling variants, dropped source words or
  function words; negations and numbers must be identical; the citation carries the
  **source's own text**. Used by RAG citations and intelligence evidence.
* **Verification:** `tests/unit/test_core_quotes.py`, citation tests.

### M5 — User enumeration through forgot-password timing (Medium, fixed)
* **Issue:** existing accounts triggered a transaction and an inline e-mail send (≈ 11 ms vs
  6 ms); send errors surfaced as 500 only for existing accounts.
* **Fix:** after the rate limits the endpoint returns immediately; lookup, token creation
  and delivery run in a background task; delivery errors are logged, never returned.
* **Verification:** `test_m5_forgot_password_does_the_work_off_the_request_path`.

### M6 — PDF active-content scanner failed open on two layouts (Medium, fixed)
* **Issue:** an object stream whose `/Filter` was an indirect reference was treated as
  uncompressed; a `/Length` that ended before `endstream` hid real objects in the skipped gap.
* **Impact:** a PDF with `/OpenAction`/`/JavaScript` could be accepted instead of quarantined
  (exploitation still needs the victim's reader; downloads are `attachment` + `CSP: sandbox`).
* **Fix:** an unresolved `/Filter` or indirect `/DecodeParms` makes the stream unscannable
  (HIGH → quarantine); the gap after a mismatched `/Length` is scanned as objects and the
  mismatch is reported.
* **Verification:** `tests/unit/test_documents_scanning.py::test_object_stream_with_indirect_*`,
  `test_objects_hidden_behind_a_short_length_are_scanned`.

### L1 — Older reset links stayed valid after a reset (Low, fixed)
* **Fix:** any successful reset or password change consumes every outstanding reset token.
* **Verification:** `test_l1_every_reset_link_dies_when_one_is_used`.

### L2 — MFA could be enrolled from a stolen access token (Low, fixed)
* **Issue:** enrolment needed no password; there was no administrative recovery.
* **Fix:** enrolment requires the current password (throttled, lockout-counted, audited);
  users are e-mailed on MFA and password changes; new audited admin action
  `POST /api/v1/users/{id}/reset-mfa` (revokes sessions, not usable on oneself).
* **Verification:** `test_l2_mfa_enrolment_needs_the_password`, `test_l2_admin_can_reset_a_hijacked_mfa`.

### L3 — Org admins can learn that an e-mail exists in another tenant (Low, mitigated)
* **Issue:** e-mail addresses are platform-wide login identifiers, so creating a user with a
  taken address returns 409 even if the address belongs to another tenant.
* **Mitigation:** every such conflict is audited (`admin.user_email_conflict`) and user
  creation is rate-limited; hiding the conflict would break administration.
* **Residual risk:** accepted and documented; the complete fix is organisation-qualified
  login (per-tenant e-mail uniqueness), a larger product change.
* **Verification:** `test_l3_cross_tenant_email_probe_is_audited`.

### L4 — Agent excerpt tool returned passages the retriever excludes (Low, fixed)
* **Fix:** passages at/above the injection exclusion threshold are withheld from tool output
  and counted; passages above the warning threshold are labelled.
* **Verification:** `test_l4_agent_excerpt_tool_withholds_injected_passages`.

### L5 — Output guard missed some markdown/URL forms and a fullwidth canary (Low, fixed)
* **Fix:** NFKC + invisible-character removal before all checks; reference-style images and
  link definitions removed; images across newlines; scheme-relative and `mailto:`/`file:` URLs
  treated as URLs.
* **Verification:** `tests/unit/test_rag_context_citations_guard.py::test_guard_*`.

### L6 — A tightened AI policy took up to 60 s to apply (Low, fixed)
* **Fix:** the organisation-policy cache entry is deleted right after the settings commit.
* **Verification:** `test_l6_m3_lowering_the_org_ceiling_applies_immediately`.

### L7 — Disclosed documents were not fully audited (Low, fixed)
* **Fix:** `assistant.tool_call` records the ids of documents whose content a tool returned;
  `assistant.ask` records every document whose text was returned as evidence.
* **Verification:** agent and RAG integration tests.

## 4. Assessment by area

| Area | Assessment |
|---|---|
| Authentication | strong after M1/M2/M5/L1/L2: Argon2id, lockout on every credential check, MFA, revocable sessions, rotating refresh tokens |
| Authorization | strong: RBAC + ACL-in-SQL, differential-tested, 404 for invisible objects |
| Tenant isolation | strong: RLS fail-closed, least-privileged roles verified at start-up, composite FKs |
| API security | strong: strict schemas, body limits, security headers, problem responses without internals |
| Database security | strong: parameterised SQL, constraints, append-only audit, legal-hold trigger |
| File and parser security | strong after M6: sniffing, zip inspection, scanners fail closed, sandboxed parsing |
| RAG / LLM | strong after M3/M4/L4/L5: authorised retrieval, no-tool answers, verified citations, data governance |
| Vector database | strong: authorization inside vector queries; Qdrant hits re-verified |
| Prompt injection | contained architecturally; residual wording bias documented |
| Sensitive data | classification routing, pseudonymisation/redaction for external providers, redacted logs and audit |
| Secrets management | file-based secrets, production validator, no secrets in images/CI |
| SSRF / SQL injection / IDOR / escalation | no findings |
| Rate limiting | GCRA with fallback; per-account limits on every credential check |
| Logging and auditability | structured redacted logs; tamper-evident audit with verification; disclosure audited (L7) |
| Docker / CI/CD / dependencies | hardened containers, SHA-pinned actions, lock file, scanners |
| Backup / retention / incident response | documented in [operations.md](operations.md); restore drill script |

## 5. Residual risks

See [threat-model.md §9](threat-model.md#9-assumptions-and-residual-risks), plus:
* L3 — cross-tenant e-mail existence (audited, rate-limited).
* PDF heuristics are format-specific; novel malware relies on ClamAV signatures.
* A database superuser can read row content (mitigated operationally and by storage encryption).

## 6. Recommended next steps

1. Organisation-qualified login to remove L3.
2. WebAuthn/passkeys and an organisation policy "MFA required".
3. Content disarm & reconstruction for PDFs from untrusted sources.
4. Signed images and SLSA provenance on release.
5. Schemathesis fuzzing of the OpenAPI surface in CI.
