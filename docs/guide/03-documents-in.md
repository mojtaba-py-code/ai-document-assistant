# Chapter 3 — Documents in (Levels 9–12)

## Level 9 — Secure document upload

**Objective.** Accept business files from users without ever trusting their name, extension,
MIME type, metadata or content.

**Files.** `documents/validation.py`, `documents/scanning.py`, `documents/storage.py`,
`documents/service.py`, `documents/url_import.py`, `api/routers/documents.py`.

**Flow.** authenticate → RBAC `document:upload` → rate limit → stream into a bounded spool
while hashing (abort at the first chunk over the limit) → `sniff` magic bytes (extension
must agree) → `inspect_zip` for OOXML (entry count, total and per-entry ratios, traversal,
encryption, symlinks, nesting, macro-enabled package types — metadata only, nothing
extracted) → `validate_text` for text formats → `ActiveContentScanner` + malware scanner →
`decide` (reject, quarantine or accept) → encrypt with a per-object key bound to
`org/version/id` → one transaction: document + version + ingestion job + audit.

**Security details worth studying.**
* `sanitize_filename` keeps only a bounded basename; storage keys are server-generated.
* PDF scanning decodes `#xx` name escapes and inflates object streams so `/J#61vaScript`
  or a compressed `/OpenAction` cannot hide.
* `ClamAvScanner` fails **closed**: when a scanner is configured and unreachable, uploads
  are refused rather than accepted unscanned.
* Duplicate detection only reveals a duplicate the uploader can already list.
* Quarantined files are stored encrypted, never parsed, and downloadable only by managers
  after an explicit, audited risk acknowledgement.
* Downloads use `Content-Disposition: attachment`, `nosniff`, `Content-Security-Policy:
  sandbox` and `no-store`.

**Common vulnerabilities avoided.** Unrestricted file upload (OWASP), zip bombs, XXE,
path traversal, content-type confusion, stored XSS via served files.

**Run / verify.**

```bash
python -m pytest tests/unit/test_documents_validation.py tests/unit/test_documents_scanning.py tests/security/test_upload_security.py -q
```

**Security review**

| | |
|---|---|
| Attack surface | multipart body, filename, declared MIME, file bytes, URL import |
| Threats | malware, active content, bombs, traversal, SSRF via import, existence leaks |
| Controls | streaming limits, sniffing, zip inspection, active-content + ClamAV, quarantine, encryption, egress allowlist |
| Weaknesses | signature scanning cannot catch novel malware; heuristics are format-specific |
| Improvements | CDR (content disarm & reconstruction) for PDFs; sandboxed detonation for high-risk tenants |
| Tests | malicious-upload suite (EICAR, macro docm renamed .docx, zip bomb, `/JavaScript` incl. hex-escaped names, encrypted PDF, external relationships, traversal filenames) |

## Level 10 — Document processing pipeline

**Objective.** Turn an encrypted upload into searchable, analysed chunks without blocking the
API and without trusting the parser.

**Files.** `ingestion/pipeline.py`, `ingestion/jobs.py`, `jobs/queue.py`, `jobs/worker.py`.

**Key design.** The upload transaction enqueues `ingest_version` (transactional outbox). The
worker claims it with `FOR UPDATE SKIP LOCKED`, marks the version `processing`, decrypts,
parses in the sandbox, analyses, embeds (policy-aware), then commits chunks, embeddings,
fields and the status switch in **one** transaction. The document keeps serving its previous
version until the new one is fully indexed. Failures are recorded with a sanitised error
code; parse errors are permanent (dead-lettered), provider outages are retried with backoff.

**Security review**

| | |
|---|---|
| Attack surface | job payloads, decrypted plaintext in memory, parser output |
| Threats | poison jobs, plaintext on disk, cross-tenant processing, half-indexed versions |
| Controls | IDs-only payloads, stdin-only plaintext, system context per organisation, single-transaction commit |
| Weaknesses | very large documents are held in memory during parsing (bounded by the upload limit) |
| Improvements | streaming parsers for multi-hundred-megabyte archives |
| Tests | `tests/integration/test_ingestion_pipeline.py`, `tests/integration/test_worker_runtime.py` |

## Level 11 — Text extraction and metadata

**Files.** `ingestion/sandbox.py`, `ingestion/sandbox_child.py`, `ingestion/parsers/*`,
`ingestion/model.py`, `ingestion/normalize.py`, `ingestion/ocr.py`, `ingestion/classify.py`,
`ingestion/sensitivity.py`, `ingestion/extraction_rules.py`, `ingestion/language.py`.

**Parser choices.** pypdf (pure Python, text layer only), python-docx (entity resolution
disabled), openpyxl (`read_only`, `data_only`, no external links) with defusedxml, stdlib csv
with bounded fields/rows, text/Markdown with heading detection. No native PDF renderer.

**Sandbox.** `python -I -B -m docassist.ingestion.sandbox_child`: minimal environment (no
secrets, no `DOCASSIST_*`), plaintext only on stdin, output capped, wall-clock timeout that
kills the process tree, POSIX `RLIMIT_AS/CPU/FSIZE=0/NOFILE/CORE/NPROC`, best-effort Linux
network namespace, strict validation of the returned JSON. A `probe` mode reports the
*names* of inherited variables so operators can verify isolation.

**Metadata is untrusted.** Titles/authors from files are sanitised and length-capped and
never used for authorization. Sensitivity detection counts PII kinds and may *suggest* a
higher classification (never lowers, never auto-applies; audited).

**Security review**

| | |
|---|---|
| Attack surface | parser libraries processing attacker bytes |
| Threats | parser RCE, DoS, secret theft from the environment, hidden-text payloads |
| Controls | sandbox process, rlimits, timeout, no secrets, output validation, Unicode hygiene |
| Weaknesses | on Windows development machines only timeout + clean environment apply |
| Improvements | seccomp profile / gVisor for the worker container |
| Tests | `tests/unit/test_ingestion_sandbox.py`, `test_ingestion_parsers.py`, `test_ingestion_normalize.py` |

## Level 12 — Chunking and document indexing

**Files.** `ingestion/chunking.py`, `ingestion/injection.py`, `db/models.py`
(`DocumentChunk`, `ChunkEmbedding`).

**Theory.** Chunks must be small enough to embed and cite precisely, large enough to carry
meaning, and must keep structure (headings, table headers, pages) for citations.

**Guarantees (property-tested with Hypothesis).** No text lost; every chunk ≤ `max_tokens`;
strictly increasing offsets; overlap bounded; never split inside a word; deterministic.
Tables split only between rows, repeating the header line. Every chunk records
`page_start/page_end`, `section`, `heading_path`, character offsets, an injection score with
flags, and the PII kinds it contains.

**Indexing.** A generated weighted `tsvector` (section heading weight A, English stems B,
exact tokens C) behind a GIN index; embeddings in `chunk_embeddings` behind HNSW.

**Security review**

| | |
|---|---|
| Attack surface | chunk text later shown to users and models |
| Threats | instructions hidden in documents, invisible Unicode, oversized chunks exhausting context |
| Controls | injection scoring through plain/deobfuscated/folded/hidden views, Unicode hygiene, size bounds |
| Weaknesses | heuristics cannot recognise every paraphrased instruction |
| Improvements | a small classifier model as an additional signal |
| Tests | `tests/unit/test_ingestion_chunking.py`, `tests/unit/test_ingestion_injection.py`, `tests/adversarial/test_document_injection_corpus.py` |
