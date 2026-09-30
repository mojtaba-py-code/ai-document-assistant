# API Reference

Base path: `/api/v1`. All endpoints except login, refresh, password reset and signed export
downloads require `Authorization: Bearer <access token>`. The interactive OpenAPI UI is at
`/docs` outside production (`DOCASSIST_SECURITY__EXPOSE_API_DOCS`).

Every response carries `X-Request-ID` (send your own, 8–64 URL-safe characters, to correlate
logs). API responses are `Cache-Control: no-store`.

## Endpoints

| Area | Method | Path | Permission |
|---|---|---|---|
| Auth | POST | `/auth/login` | public (rate-limited) |
| | POST | `/auth/mfa/verify` | MFA challenge |
| | POST | `/auth/refresh` | refresh token (cookie + `X-CSRF-Protection: 1`, or body) |
| | POST | `/auth/logout` | authenticated |
| | GET | `/auth/me` | authenticated |
| | POST | `/auth/password/change` · `/password/forgot` · `/password/reset` | authenticated · public · reset token |
| | POST | `/auth/mfa/enroll` (current password) · `/mfa/confirm` · `/mfa/disable` (password + code) | authenticated |
| Documents | POST, GET | `/documents` | `document:upload` · `document:read` |
| | POST | `/documents/import-url` | `document:upload` (feature flag) |
| | GET, PATCH, DELETE | `/documents/{id}` | listable · manage · manage |
| | GET | `/documents/{id}/content` | read |
| | GET | `/documents/{id}/download` · `/versions/{n}/download` | read |
| | POST | `/documents/{id}/versions` | manage |
| | GET, POST | `/documents/{id}/grants` | manage |
| | DELETE | `/documents/{id}/grants/{grant_id}` | manage |
| Search | POST | `/search` | `search:use` |
| Assistant | POST | `/assistant/ask` | `assistant:use` |
| | POST | `/assistant/agent` | `assistant:agent` |
| | GET | `/assistant/conversations` · `/conversations/{id}` | owner only |
| | DELETE | `/assistant/conversations/{id}` | owner only |
| | GET | `/assistant/policy` | `assistant:use` |
| Intelligence | POST | `/intelligence/summarize` · `/compare` | `intelligence:use` |
| | GET | `/intelligence/deadlines` | `intelligence:use` |
| | POST | `/intelligence/extract` | `document:read` (+ `intelligence:use` to persist) |
| | GET | `/intelligence/documents/{id}/fields` · `/report` | `document:read` · `intelligence:use` |
| Exports | POST, GET | `/exports` | `export:create` |
| | GET | `/exports/{id}` · `/exports/{id}/download` | creator only |
| | POST | `/exports/{id}/link` | creator only |
| | GET | `/exports/download?token=…` | signed single-use link |
| Organisation | GET, PATCH | `/organization` · `/organization/settings` | `org:read` · `org:update` |
| | GET, POST | `/departments` | `department:read` · `department:manage` |
| | PATCH, DELETE | `/departments/{id}` | `department:manage` |
| | GET, POST | `/users` | `user:read` · `user:manage` |
| | GET, PATCH | `/users/{id}` | `user:read` · `user:manage` |
| | POST | `/users/{id}/revoke-sessions` · `/send-reset` · `/reset-mfa` | `user:manage` |
| | GET | `/usage` | `usage:read` |
| Audit | GET | `/audit/events` | `audit:read` |
| | POST | `/audit/verify` | `audit:verify` |
| Jobs | GET | `/jobs` | `jobs:read` |
| | POST | `/jobs/{id}/retry` · `/cancel` | `jobs:manage` |
| Platform | POST, GET | `/platform/organizations` | `org:create` · `org:read_any` |
| | GET, PATCH | `/platform/organizations/{id}` | `org:read_any` · `org:update_any` |
| Operations | GET | `/admin/health` | admins |
| | GET | `/health/live`, `/health/ready`, `/metrics` | public / public / metrics token |

A document the caller may not see always yields **404**, never 403, so the API does not
reveal whether it exists.

## Errors

Errors are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) problem documents
(`application/problem+json`):

```json
{
  "type": "https://github.com/mojtaba-py-code/ai-document-assistant/blob/main/docs/api.md#not_found",
  "title": "Not found",
  "status": 404,
  "detail": "The requested resource was not found.",
  "code": "not_found",
  "request_id": "9f3c0b6a1d2e4f5a6b7c8d9e"
}
```

`detail` is always safe to show to a user; internal causes are only in server logs.
Validation errors list field locations and messages but never echo the submitted value.

### validation_failed
422 — the request body or parameters are invalid.

### authentication_failed
401 — missing, expired, revoked or forged credentials. Login failures use one identical
message for every cause.

### mfa_required
401 — a one-time code is required to finish sign-in.

### permission_denied
403 — the caller lacks the permission, or may see the resource but not change it.

### not_found
404 — the resource does not exist **or is not visible to the caller**.

### conflict
409 — for example a duplicate upload (`duplicate_of` is returned only when the caller can
see the existing document) or a document that is not ready yet.

### gone
410 — an expired export or an already-used download link.

### payload_too_large
413 — the body exceeds the configured limit (checked on the header and while streaming).

### unsupported_media_type
415 — the file type is not allowed or does not match its extension.

### content_rejected
422 — the file failed security checks (macros, active content, malware, archive bombs).

### url_import_refused
422 — the URL is not allowed by the import/egress policy.

### url_import_failed
422 — the remote document could not be fetched safely.

### rate_limited
429 — too many requests; see `Retry-After`.

### quota_exceeded
429 — the organisation's monthly AI token budget is used up.

### llm_policy_denied
403 — no approved model may process data of this classification.

### llm_unavailable
503 — the model provider is unavailable (or its circuit breaker is open).

### llm_output_invalid
503 — the model returned output that failed schema validation.

### llm_refused
503 — the model declined the request.

### llm_error
503 — another model-provider error.

### feature_disabled
404 — the feature is switched off in this deployment.

### service_unavailable
503 — a dependency is temporarily unavailable.

### internal_error
500 — an unexpected error; quote the `request_id` when reporting it.

### method_not_allowed
405 — the HTTP method is not supported on this path.

### invalid_request
400 — a malformed request (for example an invalid `Content-Length`).

### http_error
Other HTTP-level errors.
