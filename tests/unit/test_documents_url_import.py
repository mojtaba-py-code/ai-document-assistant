"""URL import: feature flag, allowlist intersection, SSRF refusals, limits, filenames."""

from __future__ import annotations

import httpx
import pytest

from docassist.core.config import UploadSettings
from docassist.core.errors import FeatureDisabled, PayloadTooLarge
from docassist.documents.url_import import (
    UrlImporter,
    UrlImportFailed,
    UrlImportRefused,
    build_import_policy,
    filename_for,
)
from docassist.security import ssrf
from docassist.security.ssrf import EgressPolicy, Target
from tests.fixtures import sample_files as sf

EGRESS = EgressPolicy.build(
    ["api.anthropic.com", "docs.example.com", "*.files.example.org", "cdn.example.net"],
    ["intranet-docs"],
    max_response_bytes=10_000_000,
)


def _upload(**overrides: object) -> UploadSettings:
    values: dict[str, object] = {
        "url_import_enabled": True,
        "url_import_allowed_domains": ["docs.example.com", "*.files.example.org", "intranet-docs"],
        "max_upload_bytes": 50_000,
    }
    values.update(overrides)
    return UploadSettings(**values)  # type: ignore[arg-type]


def _importer(handler: object, **overrides: object) -> UrlImporter:
    return UrlImporter(
        _upload(**overrides),
        EGRESS,
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(target: Target) -> str:
        return "10.0.0.8" if target.private_allowed else "93.184.216.34"

    monkeypatch.setattr(ssrf, "resolve_checked", resolve)


def test_policy_is_the_intersection_of_import_and_egress_allowlists() -> None:
    policy = build_import_policy(_upload(), EGRESS)
    assert policy.host_allowed("docs.example.com")
    assert policy.host_allowed("a.files.example.org")
    assert not policy.host_allowed("files.example.org")  # wildcard never matches the apex
    assert not policy.host_allowed("api.anthropic.com")  # egress-only host
    assert not policy.host_allowed("evil.example")
    assert policy.private_allowlist == frozenset({"intranet-docs"})
    assert policy.max_response_bytes == 50_000  # capped at the upload limit


def test_import_domain_missing_from_egress_policy_is_refused() -> None:
    importer = UrlImporter(_upload(url_import_allowed_domains=["only-import.example"]), EGRESS)
    with pytest.raises(UrlImportRefused):
        importer.check("https://only-import.example/a.pdf")


def test_disabled_feature() -> None:
    importer = UrlImporter(_upload(url_import_enabled=False), EGRESS)
    with pytest.raises(FeatureDisabled):
        importer.check("https://docs.example.com/a.pdf")


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.example.com/a.pdf",  # plain http to a public host
        "https://user:pw@docs.example.com/a.pdf",
        "file:///etc/passwd",
        "ftp://docs.example.com/a.pdf",
        "https://127.0.0.1/a.pdf",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::1]/a.pdf",
        "https://localhost/a.pdf",
        "https://evil.example/a.pdf",
        "https://docs.example.com.evil.example/a.pdf",
        "https://2130706433/a.pdf",
    ],
)
def test_ssrf_refusals_without_network(url: str) -> None:
    importer = UrlImporter(_upload(), EGRESS)
    with pytest.raises(UrlImportRefused):
        importer.check(url)


async def test_successful_fetch(public_dns: None) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=sf.build_pdf()
        )

    fetched = await _importer(handler).fetch("https://docs.example.com/reports/Q3%20report.pdf")
    assert fetched.filename == "Q3 report.pdf"
    assert fetched.content_type == "application/pdf"
    assert fetched.host == "docs.example.com"
    assert fetched.content.startswith(b"%PDF-")
    assert seen[0].url.host == "93.184.216.34"  # pinned to the vetted address
    assert seen[0].headers["host"] == "docs.example.com"


async def test_resolution_to_private_address_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_getaddrinfo(*_a: object, **_k: object) -> list[tuple[object, ...]]:
        return [(2, 1, 6, "", ("10.1.2.3", 443))]

    import asyncio

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    importer = _importer(lambda request: httpx.Response(200))
    with pytest.raises(UrlImportRefused):
        await importer.fetch("https://docs.example.com/a.pdf")


async def test_redirect_to_disallowed_host_is_refused(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://169.254.169.254/latest"})

    with pytest.raises(UrlImportRefused):
        await _importer(handler).fetch("https://docs.example.com/a.pdf")


async def test_redirect_within_allowlist_is_followed(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://x.files.example.org/b.csv"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"a,b\n1,2\n")

    fetched = await _importer(handler).fetch("https://docs.example.com/start")
    assert fetched.host == "x.files.example.org"
    assert fetched.filename == "b.csv"  # text/plain keeps a .csv name


async def test_disallowed_content_type_is_refused(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<script>")

    with pytest.raises(UrlImportRefused):
        await _importer(handler).fetch("https://docs.example.com/page")


async def test_oversize_response_maps_to_payload_too_large(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 60_000)

    with pytest.raises(PayloadTooLarge):
        await _importer(handler).fetch("https://docs.example.com/big.txt")


async def test_remote_error_status_is_refused(public_dns: None) -> None:
    with pytest.raises(UrlImportRefused):
        await _importer(lambda request: httpx.Response(404)).fetch("https://docs.example.com/x.pdf")


async def test_network_failure_is_reported(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(UrlImportFailed):
        await _importer(handler).fetch("https://docs.example.com/x.pdf")


@pytest.mark.parametrize(
    ("url", "content_type", "expected"),
    [
        ("https://h/a/report.pdf", "application/pdf", "report.pdf"),
        ("https://h/download?id=5", "application/pdf", "download.pdf"),
        ("https://h/", "text/plain", "document.txt"),
        ("https://h/data.csv", "text/plain", "data.csv"),
        ("https://h/notes.md", "text/markdown", "notes.md"),
        ("https://h/fake.pdf", "text/plain", "fake.txt"),
        ("https://h/..%2F..%2Fetc%2Fpasswd", "text/plain", "passwd.txt"),
        (
            "https://h/sheet",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "sheet.xlsx",
        ),
    ],
)
def test_filename_for(url: str, content_type: str, expected: str) -> None:
    assert filename_for(url, content_type) == expected


async def test_private_service_on_both_allowlists_may_use_http(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"hello")

    fetched = await _importer(handler).fetch("http://intranet-docs/wiki/page.txt")
    assert (fetched.host, fetched.filename) == ("intranet-docs", "page.txt")
