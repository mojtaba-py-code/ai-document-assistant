"""Prometheus metrics. Labels are low-cardinality by construction: no user IDs, no queries,
no document names - only route templates, statuses, providers, models and outcomes."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

HTTP_REQUESTS = Counter(
    "docassist_http_requests_total",
    "HTTP requests",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "docassist_http_request_duration_seconds",
    "HTTP latency",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
AUTH_EVENTS = Counter(
    "docassist_auth_events_total", "Authentication events", ["event", "outcome"], registry=REGISTRY
)
RATE_LIMITED = Counter(
    "docassist_rate_limited_total",
    "Requests rejected by rate limiting",
    ["bucket"],
    registry=REGISTRY,
)
DEGRADED_MODE = Counter(
    "docassist_degraded_mode_total",
    "Fallbacks taken because a dependency was unavailable",
    ["component"],
    registry=REGISTRY,
)
LLM_REQUESTS = Counter(
    "docassist_llm_requests_total",
    "LLM requests",
    ["provider", "model", "task", "status"],
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "docassist_llm_tokens_total",
    "LLM tokens",
    ["provider", "model", "direction"],
    registry=REGISTRY,
)
LLM_COST = Counter(
    "docassist_llm_cost_usd_total",
    "Estimated LLM spend (USD)",
    ["provider", "model"],
    registry=REGISTRY,
)
LLM_LATENCY = Histogram(
    "docassist_llm_latency_seconds",
    "LLM call latency",
    ["provider", "task"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
EMBEDDING_LATENCY = Histogram(
    "docassist_embedding_latency_seconds",
    "Embedding batch latency",
    ["provider"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
RETRIEVAL_LATENCY = Histogram(
    "docassist_retrieval_latency_seconds",
    "Retrieval latency",
    ["mode"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
DB_QUERY_LATENCY = Histogram(
    "docassist_db_query_duration_seconds",
    "Database statement latency",
    ["role"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
INGESTION = Counter(
    "docassist_ingestion_total", "Document ingestion outcomes", ["outcome"], registry=REGISTRY
)
INGESTION_LATENCY = Histogram(
    "docassist_ingestion_duration_seconds",
    "End-to-end ingestion time",
    ["format"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)
JOBS = Counter(
    "docassist_jobs_total", "Background job outcomes", ["kind", "outcome"], registry=REGISTRY
)
JOB_QUEUE_DEPTH = Gauge(
    "docassist_job_queue_depth", "Jobs by status", ["status"], registry=REGISTRY
)
SECURITY_EVENTS = Counter(
    "docassist_security_events_total",
    "Security detections (injection, malware, leakage...)",
    ["kind"],
    registry=REGISTRY,
)
CITATIONS = Counter(
    "docassist_citations_total", "Citation validation outcomes", ["outcome"], registry=REGISTRY
)
