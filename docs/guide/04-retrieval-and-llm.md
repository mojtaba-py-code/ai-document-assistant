# Chapter 4 — Retrieval and the LLM (Levels 13–16)

## Level 13 — Embeddings and the vector database

**Files.** `embeddings/base.py`, `embeddings/hashing.py`, `embeddings/openai_compat.py`,
`embeddings/factory.py`, `search/vector_store.py`, `search/pgvector_store.py`,
`search/qdrant_store.py`, `search/jobs.py`.

**Theory.** An embedding maps text to a vector so that similar meaning ⇒ nearby vectors.
Approximate nearest-neighbour indexes (HNSW) make similarity search sub-linear.

**Design.**
* Providers behind one protocol: an offline feature-hashing embedder (development, tests,
  air-gapped demos; never sends text anywhere) and any OpenAI-compatible `/embeddings`
  server through the SSRF-guarded client. Returned vectors are validated (count, dimension,
  finiteness).
* **Embeddings are data egress.** The pipeline refuses to send text above the external
  classification ceiling to an external embedder; such versions are keyword-only.
* **pgvector (default)** — the ACL runs in the same SQL statement as `ORDER BY embedding <=>
  :q`; if the approximate scan returns too few rows for a small tenant, an exact query runs.
* **Qdrant (optional)** — payload filters mirror the policy (organisation is the tenant key)
  and every hit is **re-verified in PostgreSQL**, so a stale payload can only hide results.
  PostgreSQL stays the source of truth; Qdrant can be rebuilt from it.

**Security review**

| | |
|---|---|
| Attack surface | vector queries, embedding provider traffic, external index payloads |
| Threats | similarity search returning unauthorised chunks, stale permissions, data egress |
| Controls | ACL in the vector query, re-verification, classification ceiling for embedders |
| Weaknesses | embeddings can leak approximate content if exfiltrated (treat as sensitive) |
| Improvements | per-tenant Qdrant collections for the largest customers |
| Tests | `tests/integration/test_search_authz.py`, `test_search_qdrant.py` (parity + stale payloads), `tests/unit/test_search_qdrant_filter.py` |

## Level 14 — Semantic and hybrid search

**Files.** `search/keyword.py`, `search/hybrid.py`, `search/rerank.py`, `search/service.py`,
`search/sql.py`, `search/text.py`, `api/routers/search.py`.

**When each method helps.** Keyword (FTS) wins on exact identifiers ("INV-2026-0142", clause
numbers, names); semantic wins on paraphrases ("time off" → "annual leave"); hybrid fuses
both with **Reciprocal Rank Fusion** (`1/(k + rank)`), which needs no score calibration.
A lexical reranker then rewards coverage of query terms, phrases and matching headings, and
**MMR** removes near-duplicate chunks.

**Security.** Every candidate query embeds the authorization scope (`search/sql.py`); final
rows are loaded again with the readable clause (defence in depth); snippets are plain text;
queries are length-limited, rate-limited and audited as a hash + redacted text; embedding
failures degrade to keyword search (`degraded: true`).

**Run / verify.**

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/search -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"query": "payment terms", "mode": "hybrid"}'
```

**Security review**

| | |
|---|---|
| Attack surface | query text, filters, pagination |
| Threats | SQL injection, metadata leaks in titles/snippets, DoS by expensive queries |
| Controls | bound parameters, `websearch_to_tsquery`, typed filters, limits, statement timeout |
| Weaknesses | result ranking can reveal *relative* relevance of the caller's own documents only |
| Improvements | an optional cross-encoder reranker (hook exists: `LLMReranker`) |
| Tests | `tests/security/test_search_injection.py`, `tests/unit/test_search_hybrid.py` |

## Level 15 — RAG pipeline

**Files.** `rag/service.py`, `rag/query.py`, `rag/deadlines.py`, `rag/context.py`,
`rag/prompts.py`, `rag/citations.py`, `rag/guard.py`, `rag/conversations.py`.

**Pipeline.** validate → analyse (deterministic intent, time window, doc-type hints, abuse
flags) → **refuse** high-confidence abuse without retrieval → **deadline path** (SQL over
extracted fields, no model) → **document-finding path** ("find documents about X": ranked
authorised documents with their best passage, no model) → authorised hybrid retrieval →
nothing relevant ⇒ "insufficient context" (no model call) → data governance → spotlighted
context → gateway (JSON schema) → citation verification → output guard → confidence →
persist in a private conversation → audit.

**Why it matters.** Enterprise users forgive "I couldn't find that"; they do not forgive a
confident invented answer with a fake citation.

**Security review**

| | |
|---|---|
| Attack surface | the question, conversation history, retrieved text, model output |
| Threats | unauthorised retrieval, injection, hallucination, conversation leakage |
| Controls | retrieval-time authorization, no tools, verified citations, guard, owner-only conversations (RLS), previous *questions* (not answers) reused for follow-ups |
| Weaknesses | semantic manipulation of wording by an injected document the user may read |
| Improvements | human feedback loop on low-confidence answers |
| Tests | `tests/integration/test_rag_service.py`, `test_rag_with_search_service.py`, `tests/unit/test_rag_query.py` |

## Level 16 — LLM gateway

**Files.** `llm/base.py`, `llm/gateway.py`, `llm/anthropic_provider.py`,
`llm/openai_compat.py`, `llm/local_extractive.py`, `llm/circuit.py`, `llm/pricing.py`,
`llm/schema.py`, `llm/wiring.py`, `audit/usage.py`.

**Responsibilities (in order).** request shape checks → **routing by data classification**
(external only up to the ceiling; organisation policy may lower it) → model tier (fast:
classify/rerank/extract; main: answer/summarise/compare/agent) → per-organisation rate limit
and monthly token budget → circuit breaker → **PII pseudonymisation** for external providers
(restored afterwards) → context-size estimate → provider call → strict JSON-schema
validation with one repair attempt → usage, latency and cost recording → metrics.

**Claude specifics** (`anthropic_provider.py`): official SDK; `claude-opus-5-5` main /
`claude-haiku-4-5` fast; structured output via `output_config.format`; effort instead of
thinking budgets; no sampling parameters, no prefill, no forced tool choice (strict tools +
`auto`); server-side refusal fallbacks; `stop_reason` checked before content.

**Provider abstraction.** The rest of the platform depends only on `LLMGateway.complete`;
switching to a local model server or an offline provider is configuration.

**Security review**

| | |
|---|---|
| Attack surface | prompts leaving the network; provider responses entering it |
| Threats | sensitive-data disclosure, cost abuse, provider outages, malformed output |
| Controls | classification routing, pseudonymisation, budgets, circuit breaker, schema validation, no logging of prompts/completions |
| Weaknesses | pseudonymisation is pattern-based (names are not masked) |
| Improvements | NER-based name masking for CONFIDENTIAL traffic |
| Tests | `tests/unit/test_llm_gateway.py`, `test_llm_anthropic_provider.py` (request shapes via a mocked transport), `test_llm_openai_compat.py` |
