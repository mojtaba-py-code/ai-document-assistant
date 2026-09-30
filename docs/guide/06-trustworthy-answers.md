# Chapter 6 — Answers people can trust (Levels 21–24)

## Level 21 — Document summarisation

**Files.** `intelligence/summarize.py`, `intelligence/prompting.py`, `api/routers/intelligence.py`.

**Approach.** Map-reduce over the current version's chunks: injection-flagged chunks are
dropped, the rest get local ids `C1…Cn` (the model never sees database ids), batches fit the
context budget, partial summaries cite `C` ids, the reduce step merges them (hierarchically
when needed). Key points without a valid citation are dropped. When no model may process
the document (policy, quota, outage, invalid output) a deterministic **extractive** summary
of lead sentences per section is returned and flagged `method: "extractive"`.

**Security review**

| | |
|---|---|
| Attack surface | full-document text flowing to a model |
| Threats | data egress above the ceiling, injected instructions, fabricated points |
| Controls | classification routing per call, flagged chunks excluded, citation-id checks, extractive fallback |
| Weaknesses | long documents cost several model calls |
| Improvements | cache summaries per version (they change only with a new version) |
| Tests | `tests/unit/test_intelligence_summarize.py`, `tests/integration/test_intelligence_api.py` |

## Level 22 — Document comparison

**Files.** `intelligence/compare.py`.

**Approach.** A deterministic diff is always returned: chunk overlaps removed via recorded
offsets, paragraphs aligned with `difflib.SequenceMatcher`, hunks `added/removed/changed`
with page references and word-level inline diffs, plus a field diff (e.g. "expiration moved
by 90 days", "payment terms changed from net 30 to net 60"). An optional LLM change summary
sees **only the hunks and field changes**, never the whole documents, and must reference
hunk ids that were sent.

**Security review**

| | |
|---|---|
| Attack surface | two documents (possibly of different classification) |
| Threats | comparing a readable document with an unreadable one; egress |
| Controls | both documents must be readable; call classification = the higher of the two |
| Weaknesses | very large diffs are truncated for the model (the deterministic diff is complete) |
| Improvements | semantic alignment of moved clauses |
| Tests | `tests/unit/test_intelligence_compare.py` |

## Level 23 — Citation and grounding system

**Files.** `rag/citations.py`, `rag/service.py`, `rag/guard.py`, UI `views/answer.js`.

**Rules.** A citation survives only if its quote, after normalisation, is a whole-word
substring of its source, or (for longer quotes) a same-length window shares ≥ 85 % of tokens.
Short "quotes" never match. Unknown source ids are invalid. An "answered" result without any
verified citation becomes "insufficient context". Confidence combines retrieval strength,
citation validity and the model's own label. The UI shows the AI-generated answer and the
**source evidence** (verbatim quotes with document, version, page, section) in separate
panels, and each citation opens the document preview at that page with the quote
highlighted.

**Security review**

| | |
|---|---|
| Attack surface | citations produced by the model |
| Threats | invented citations, citations to unauthorised documents, misleading confidence |
| Controls | quote verification, provided-id check, authorised retrieval, confidence from evidence |
| Weaknesses | a verified quote can still be cited for a conclusion it does not support |
| Improvements | entailment checking of answer sentences against quotes |
| Tests | `tests/unit/test_rag_context_citations_guard.py`, evaluation harness citation precision |

## Level 24 — Background workers

**Files.** `jobs/queue.py`, `jobs/registry.py`, `jobs/worker.py`, `jobs/maintenance.py`,
`ingestion/jobs.py`, `search/jobs.py`, `intelligence/jobs.py`.

**Design.** PostgreSQL queue with the transactional-outbox property, `FOR UPDATE SKIP
LOCKED` claims in short transactions, leases with heartbeats, **fencing** on completion,
exponential backoff with jitter, permanent errors dead-lettered immediately, unknown kinds
dead-lettered, per-job timeouts, graceful shutdown, a least-privilege start-up check.
Maintenance (guarded by an advisory lock, so one replica runs it): audit sealing, lease
reclaim, stuck-version repair, queue gauges and retention purges.

**Security review**

| | |
|---|---|
| Attack surface | job payloads, the worker's cross-tenant claim privilege |
| Threats | poison jobs, zombie workers overwriting results, cross-tenant processing |
| Controls | IDs-only payloads, fencing, per-organisation system context, worker role checks |
| Weaknesses | one queue table for all tenants (fairness under a noisy tenant) |
| Improvements | per-organisation concurrency caps |
| Tests | `tests/integration/test_job_queue.py`, `test_worker_runtime.py`, `test_worker_maintenance.py` |
