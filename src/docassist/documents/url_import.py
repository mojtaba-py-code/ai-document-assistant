"""Server-side document import from a URL (opt-in, allowlisted, SSRF-guarded).

The feature is off unless ``upload.url_import_enabled`` is set. A URL is fetched only when
its host is permitted by **both** ``upload.url_import_allowed_domains`` and the deployment's
outbound egress policy (:class:`ImportEgressPolicy`). Everything else is delegated to
:mod:`docassist.security.ssrf`: scheme/credential/IP-literal checks, resolution to public
addresses only, DNS pinning, per-hop re-validation of redirects, a response-size cap equal to
the upload limit, and a content-type allowlist of the supported document formats. The
fetched bytes then go through exactly the same validation and scanning as a browser upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from urllib.parse import unquote, urlsplit

import httpx

from docassist.core.config import UploadSettings
from docassist.core.errors import FeatureDisabled, PayloadTooLarge, ValidationFailed
from docassist.documents.validation import (
    FALLBACK_FILENAME,
    MIME_TYPES,
    TEXT_EXTENSIONS,
    extension_of,
    sanitize_filename,
)
from docassist.security.ssrf import (
    EgressDenied,
    EgressPolicy,
    GuardedTransport,
    fetch,
    validate_url,
)

IMPORT_CONTENT_TYPES: Final[dict[str, str]] = {
    MIME_TYPES["pdf"]: "pdf",
    MIME_TYPES["docx"]: "docx",
    MIME_TYPES["xlsx"]: "xlsx",
    "text/csv": "csv",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
}
"""Response content types accepted from remote servers, and the format each implies."""

_TOO_LARGE_REASON = "response too large"  # message of the EgressDenied raised by ssrf.fetch


class UrlImportRefused(ValidationFailed):
    code = "url_import_refused"
    title = "URL not importable"
    default_message = "This URL cannot be imported."


class UrlImportFailed(ValidationFailed):
    code = "url_import_failed"
    title = "Download failed"
    default_message = "The document could not be downloaded from the URL."


@dataclass(frozen=True)
class ImportEgressPolicy(EgressPolicy):
    """An egress policy that allows a host only if the deployment policy (``base``) does too."""

    base: EgressPolicy | None = None

    def host_allowed(self, host: str) -> bool:
        if self.base is not None and not self.base.host_allowed(host):
            return False
        return super().host_allowed(host)


def build_import_policy(upload: UploadSettings, egress: EgressPolicy) -> ImportEgressPolicy:
    """Intersect the import allowlist with the egress policy.

    Plain ``http`` and private addresses stay forbidden unless the host is also on the
    egress policy's private-service allowlist *and* on the import allowlist.
    """
    domains = frozenset(
        h.strip().lower().rstrip(".") for h in upload.url_import_allowed_domains if h.strip()
    )
    matcher = EgressPolicy(allowed_hosts=domains)
    private = frozenset(h for h in egress.private_allowlist if matcher.host_allowed(h))
    return ImportEgressPolicy(
        allowed_hosts=domains | private,
        private_allowlist=private,
        max_response_bytes=min(egress.max_response_bytes, upload.max_upload_bytes),
        timeout_seconds=egress.timeout_seconds,
        max_redirects=egress.max_redirects,
        base=egress,
    )


def filename_for(url: str, content_type: str) -> str:
    """Display filename for a fetched document: the URL's last path segment, with the
    extension forced to match the response content type (``text/plain`` also covers
    ``.csv``/``.md`` names)."""
    expected = IMPORT_CONTENT_TYPES[content_type]
    segment = unquote(urlsplit(url).path).rstrip("/").rsplit("/", 1)[-1]
    candidate = sanitize_filename(segment) if segment else FALLBACK_FILENAME
    ext = extension_of(candidate)
    if ext == expected or (expected == "txt" and ext in TEXT_EXTENSIONS):
        return candidate
    stem = candidate[: -(len(ext) + 1)] if ext else candidate
    return sanitize_filename(f"{stem}.{expected}")


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    filename: str
    content_type: str
    content: bytes
    host: str


class UrlImporter:
    def __init__(
        self,
        upload: UploadSettings,
        egress: EgressPolicy,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._upload = upload
        self._policy = build_import_policy(upload, egress)
        self._transport = transport

    @property
    def enabled(self) -> bool:
        return self._upload.url_import_enabled

    @property
    def policy(self) -> ImportEgressPolicy:
        return self._policy

    def check(self, url: str) -> str:
        """Validate ``url`` without any network access; return its (IDNA) host."""
        if not self.enabled:
            raise FeatureDisabled("Importing documents from URLs is not enabled.")
        try:
            return validate_url(url, self._policy).host
        except EgressDenied as exc:
            raise UrlImportRefused(internal_detail=str(exc)) from exc

    async def fetch(self, url: str) -> FetchedDocument:
        self.check(url)
        policy = self._policy
        client = httpx.AsyncClient(
            transport=GuardedTransport(policy, inner=self._transport),
            follow_redirects=False,
            timeout=httpx.Timeout(
                policy.timeout_seconds, connect=min(10.0, policy.timeout_seconds)
            ),
            trust_env=False,
        )
        try:
            async with client:
                result = await fetch(
                    url, policy, allowed_content_types=IMPORT_CONTENT_TYPES, client=client
                )
        except EgressDenied as exc:
            if str(exc) == _TOO_LARGE_REASON:
                raise PayloadTooLarge(
                    "The remote document exceeds the maximum upload size.",
                    internal_detail="url import response too large",
                ) from exc
            raise UrlImportRefused(internal_detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise UrlImportFailed(internal_detail=type(exc).__name__) from exc
        host = urlsplit(result.url).hostname or ""
        return FetchedDocument(
            filename=filename_for(result.url, result.content_type),
            content_type=result.content_type,
            content=result.content,
            host=host,
        )
