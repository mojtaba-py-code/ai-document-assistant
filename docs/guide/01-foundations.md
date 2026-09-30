# Chapter 1 — Foundations (Levels 1–4)

## Level 1 — Requirements and enterprise architecture

**Objective.** Decide what the platform must do and the shape that makes the security goals
achievable. The full write-up is [architecture.md](../architecture.md).

**Theory.** Security properties are much cheaper to *design in* than to bolt on. The two
decisions that shape everything else are *where authorization is enforced* (answer: inside
every data query, backed by the database) and *what the model is allowed to do* (answer:
nothing but produce text that we validate).

**Why it matters.** In enterprises the failure that ends a product is not a slow query — it
is "the HR salary sheet showed up in a sales employee's chat answer". Architecture decides
whether such a failure is impossible or merely unlikely.

**Structure.** Modular monolith (`src/docassist/<area>`), two process types (API, worker),
PostgreSQL as system of record (relational data, full-text index, vectors, job queue, audit),
Redis as a disposable accelerator, encrypted object storage for files.

**Key decisions.** ADR-01…ADR-12 in [architecture.md](../architecture.md#6-architecture-decisions-adrs).

**Security review**

| | |
|---|---|
| Attack surface | HTTP API, file uploads, LLM/embedding provider traffic, operator access |
| Threats | cross-tenant leakage, over-privileged components, model-driven actions |
| Controls | RLS, ACL-in-SQL, no-tool RAG, gateway data policy, sandboxed parsing |
| Weaknesses | a monolith shares one process memory space between features |
| Improvements | split parsing into its own no-egress service for very high-risk deployments |
| Tests | import-linter contracts (`lint-imports`) keep the module boundaries honest |

## Level 2 — Threat model and security architecture

**Objective.** Enumerate assets, actors, trust boundaries and threats, and map every threat
to a control, a detection signal and a test. See [threat-model.md](../threat-model.md) and
[security-architecture.md](../security-architecture.md).

**Theory.** STRIDE (Spoofing, Tampering, Repudiation, Information disclosure, Denial of
service, Elevation of privilege) applied per component; OWASP API Top 10, OWASP Top 10 and
the OWASP Top 10 for LLM Applications as cross-checks.

**Why it matters.** Security reviewers and buyers ask "what did you think could go wrong?".
A threat register with residual risks is the honest answer.

**Security review**

| | |
|---|---|
| Attack surface | all trust boundaries (browser, edge, API, worker, sandbox, data, third parties) |
| Threats | 28 registered threats (T01–T28) |
| Controls | mapped per threat in the register |
| Weaknesses | residual risks listed explicitly (semantic injection, superuser access, insider export) |
| Improvements | revisit after each feature and each incident |
| Tests | the "Test" column of the register points at the suites that prove each control |

## Level 3 — Project structure and configuration

**Objective.** A layout that separates concerns and a configuration system that cannot be
misconfigured silently.

**Files.**

| File | Role |
|---|---|
| `pyproject.toml` | dependencies (lower+upper bounds), ruff, mypy strict, pytest, bandit, import-linter contracts |
| `src/docassist/core/config.py` | typed settings (`DOCASSIST_` env, `__` nesting, `*_FILE` secrets) |
| `src/docassist/core/enums.py` | shared vocabularies (classification, roles, statuses) |
| `.env.example` | every setting with placeholders only |
| `src/docassist/cli.py` | `init-env` writes a private `.env` with fresh random secrets |

**Key code.** `Settings._check_secrets` rejects secrets shorter than 32 bytes, low-entropy or
placeholder values, and identical keys reused for different purposes.
`Settings._check_production` refuses to start production with wildcard hosts, exposed API
docs, a non-TLS database, no Redis, no malware scanner, offline AI components or DEBUG logs.
`hide_input_in_errors=True` keeps secret values out of validation errors.
`_WithoutFileRefs` hides `DOCASSIST_*_FILE` pointers from the environment source so secret
files can be mounted by Docker/Kubernetes.

**Common vulnerabilities avoided.** Hard-coded secrets; defaults that "work" in production;
secrets printed in tracebacks; `.env` committed (gitignored; gitleaks in CI).

**Run / verify.**

```bash
docassist init-env --path .env
python -m pytest tests/unit/test_core_config_and_auth_primitives.py tests/unit/test_deployment_config.py -q
```

**Production.** Secrets from a secret manager mounted as files; `.env` only for development.

**Security review**

| | |
|---|---|
| Attack surface | environment variables, mounted secret files, `.env` |
| Threats | weak/placeholder secrets, unsafe production flags, secret disclosure in errors |
| Controls | validators, `SecretStr`, `hide_input_in_errors`, production gate |
| Weaknesses | a developer can still set `allow_offline_ai_in_production=true` deliberately |
| Improvements | policy-as-code check of rendered Kubernetes manifests |
| Tests | `test_unsafe_production_settings_refused`, `test_secret_values_never_in_error_messages`, `test_compose_secret_files_load_through_load_settings` |

## Level 4 — FastAPI foundation

**Objective.** An HTTP layer that is secure before any feature is added.

**Files.** `api/app.py` (factory, router registry, middleware order), `api/middleware.py`
(request context, security headers, body limits), `api/problems.py` (RFC 9457 errors),
`api/deps.py` (authentication, permission gates, client IP), `api/container.py`
(composition root).

**Key code.**
* `BodySizeLimitMiddleware` rejects oversize bodies on the declared `Content-Length` **and**
  by counting streamed bytes, so chunked uploads cannot bypass it.
* `SecurityHeadersMiddleware` sets CSP (`default-src 'none'` for the API; for the UI
  `script-src 'self'` + `require-trusted-types-for 'script'` + `trusted-types 'none'`),
  `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer`, COOP/CORP,
  `Cache-Control: no-store` for API responses and HSTS in production.
* `install_error_handlers` maps domain errors to problem documents and strips the rejected
  input from validation errors (a password must never be echoed back).
* `client_ip` trusts `X-Forwarded-For` only when the peer is a configured proxy, so an
  attacker cannot rotate IPs to dodge rate limits.
* Middleware is pure ASGI: no response buffering, streaming downloads stay streaming.

**Run / verify.**

```bash
docassist serve --port 8000
curl -sI http://127.0.0.1:8000/health/live
```

**Security review**

| | |
|---|---|
| Attack surface | every HTTP request: headers, bodies, paths |
| Threats | oversize bodies, header injection, clickjacking, XSS, information leakage in errors, spoofed client IPs |
| Controls | body limits, trusted hosts, CSP + Trusted Types, problem responses, trusted-proxy IP logic |
| Weaknesses | TLS termination is delegated to the ingress |
| Improvements | CSP `report-uri` to a collector (setting `security.csp_report_uri`) |
| Tests | `test_security_headers_present`, `test_weak_password_rejected_without_echo`, UI static checks in `tests/unit/test_web_assets.py` |
