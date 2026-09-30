# Build Guide — 36 Levels

The project was specified as 36 progressive levels. This guide walks through each one as it
was actually built: the objective, the theory in brief, why it matters in real companies,
the files involved, the key code, the security implications and common vulnerabilities,
the tests, how to run and verify it, production considerations and a mini security review.

| Chapter | Levels |
|---|---|
| [1. Foundations](01-foundations.md) | 1 Requirements & architecture · 2 Threat model · 3 Structure & configuration · 4 FastAPI foundation |
| [2. Data, tenancy & identity](02-data-tenancy-identity.md) | 5 PostgreSQL · 6 Multi-tenancy · 7 Authentication · 8 Authorization & RBAC |
| [3. Documents in](03-documents-in.md) | 9 Secure upload · 10 Processing pipeline · 11 Text extraction & metadata · 12 Chunking & indexing |
| [4. Retrieval & the LLM](04-retrieval-and-llm.md) | 13 Embeddings & vector DB · 14 Hybrid search · 15 RAG pipeline · 16 LLM gateway |
| [5. Making the LLM safe](05-llm-safety.md) | 17 Prompt-injection defence · 18 Secure context · 19 Structured outputs · 20 Document intelligence |
| [6. Answers people can trust](06-trustworthy-answers.md) | 21 Summarisation · 22 Comparison · 23 Citations & grounding · 24 Background workers |
| [7. Operating it](07-operating-it.md) | 25 Redis & caching · 26 Rate limiting · 27 Audit logging · 28 Observability |
| [8. Proving it](08-proving-it.md) | 29 Security testing · 30 Adversarial LLM testing · 31 Performance · 32 Cost |
| [9. Shipping it](09-shipping-it.md) | 33 Docker · 34 CI/CD · 35 Production deployment · 36 Final security audit |

Each level ends with the review format the brief asked for:

```text
Security Review
Attack Surface / Threats / Current Security Controls / Potential Weaknesses /
Recommended Improvements / Security Tests
```

Commands assume the project root, an activated virtual environment (or `.venv/Scripts/`
prefixes on Windows), and for database tests
`DOCASSIST_TEST_DATABASE_URL=postgresql://<superuser>@<host>:<port>/postgres`.
