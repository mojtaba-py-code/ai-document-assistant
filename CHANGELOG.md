# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-30

First release.

### Added
- Multi-tenant platform with PostgreSQL row-level security, three least-privilege database
  roles, composite same-tenant foreign keys and an append-only, HMAC-chained audit trail.
- Authentication with Argon2id, lockout, TOTP MFA, revocable sessions, rotating refresh
  tokens with reuse detection and secure password reset.
- RBAC with five roles and a document ACL (classification ceiling, allowed roles,
  departments, owners, expiring grants) compiled to both Python and SQL.
- Secure uploads (PDF, DOCX, XLSX, CSV, TXT, Markdown): content sniffing, zip inspection,
  active-content and ClamAV scanning, quarantine, envelope encryption, versions, grants,
  legal hold, retention, SSRF-safe URL import.
- Sandboxed ingestion: parsing, optional OCR, structure-aware chunking, injection scoring,
  document-type classification, sensitivity detection, rules-based field extraction.
- Keyword, semantic (pgvector or Qdrant) and hybrid search with authorization inside every
  query.
- Grounded question answering with verified citations, deterministic deadline and
  document-finding answers, private conversations and a bounded read-only agent.
- LLM gateway with classification-based routing, PII pseudonymisation, budgets, circuit
  breaker and cost tracking; providers for Claude (official SDK), OpenAI-compatible local
  servers and an offline extractive model.
- Summaries, version comparison, contract/invoice extraction, reports and secure exports.
- Administration, audit and job APIs; `docassist` CLI; demo data; evaluation harness.
- Single-page web UI under a strict CSP with Trusted Types.
- Docker image, Compose stack, Kubernetes manifests, CI with CodeQL, Trivy, gitleaks,
  pip-audit and SBOM generation.
