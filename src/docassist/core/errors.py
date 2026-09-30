"""Application error hierarchy.

Services raise these; the HTTP layer maps them to RFC 9457 ``application/problem+json``.
The ``public_message`` is the only text a client ever sees - internal detail goes to
``internal_detail`` which is logged (redacted) but never serialised into a response.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    title: str = "Internal error"
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        public_message: str | None = None,
        *,
        internal_detail: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.public_message = public_message or self.default_message
        self.internal_detail = internal_detail
        self.extra = extra or {}
        super().__init__(internal_detail or self.public_message)


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"
    title = "Validation failed"
    default_message = "The request is invalid."


class AuthenticationFailed(AppError):
    status_code = 401
    code = "authentication_failed"
    title = "Authentication required"
    default_message = "Authentication failed."


class MfaRequired(AppError):
    status_code = 401
    code = "mfa_required"
    title = "Multi-factor authentication required"
    default_message = "A one-time code is required to complete sign-in."


class PermissionDenied(AppError):
    status_code = 403
    code = "permission_denied"
    title = "Forbidden"
    default_message = "You don't have permission to perform this action."


class NotFound(AppError):
    """Also used for resources that exist but are invisible to the caller (no existence leak)."""

    status_code = 404
    code = "not_found"
    title = "Not found"
    default_message = "The requested resource was not found."


class Conflict(AppError):
    status_code = 409
    code = "conflict"
    title = "Conflict"
    default_message = "The request conflicts with the current state of the resource."


class PayloadTooLarge(AppError):
    status_code = 413
    code = "payload_too_large"
    title = "Payload too large"
    default_message = "The uploaded content exceeds the allowed size."


class UnsupportedMediaType(AppError):
    status_code = 415
    code = "unsupported_media_type"
    title = "Unsupported media type"
    default_message = "This file type is not supported."


class RejectedContent(AppError):
    status_code = 422
    code = "content_rejected"
    title = "Content rejected"
    default_message = "The file was rejected by security checks."


class RateLimited(AppError):
    status_code = 429
    code = "rate_limited"
    title = "Too many requests"
    default_message = "Too many requests. Please retry later."

    def __init__(self, retry_after: float, public_message: str | None = None) -> None:
        super().__init__(public_message, extra={"retry_after": max(1, round(retry_after))})
        self.retry_after = max(1, round(retry_after))


class QuotaExceeded(AppError):
    status_code = 429
    code = "quota_exceeded"
    title = "Quota exceeded"
    default_message = "Your organization's AI usage quota has been reached."


class ServiceUnavailable(AppError):
    status_code = 503
    code = "service_unavailable"
    title = "Service unavailable"
    default_message = "A dependency is temporarily unavailable. Please retry later."


class FeatureDisabled(AppError):
    status_code = 404
    code = "feature_disabled"
    title = "Not found"
    default_message = "This feature is not enabled."
