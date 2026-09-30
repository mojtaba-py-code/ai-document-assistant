"""SSRF guard: URL validation, address classification, DNS pinning, redirects, size caps."""

from __future__ import annotations

import ipaddress

import httpx
import pytest

from docassist.security import ssrf
from docassist.security.ssrf import EgressDenied, EgressPolicy, Target, is_public_ip, validate_url

POLICY = EgressPolicy.build(
    ["api.example.com", "*.docs.example.org"], ["ollama"], max_response_bytes=1_000
)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1",
        "https://intranet.docs.example.org/file.pdf",
        "http://ollama:11434/v1",
    ],
)
def test_allowed(url: str) -> None:
    validate_url(url, POLICY)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("ftp://api.example.com/x", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("gopher://api.example.com", "scheme"),
        ("https://user:pw@api.example.com/", "credentials"),
        ("https://api.example.com@evil.com/", "credentials"),
        ("https://evil.com/", "allowlist"),
        ("https://docs.example.org/", "allowlist"),  # wildcard does not match the apex
        ("http://api.example.com/", "plain http"),
        ("https://169.254.169.254/latest/meta-data", "allowlist"),
        ("https://localhost/", "allowlist"),
    ],
)
def test_denied(url: str, reason: str) -> None:
    with pytest.raises(EgressDenied, match=reason):
        validate_url(url, POLICY)


def test_obfuscated_ip_forms_rejected() -> None:
    policy = EgressPolicy.build(["2130706433", "0x7f000001", "127.1"])
    for host in ("2130706433", "0x7f000001", "127.1"):
        with pytest.raises(EgressDenied):
            validate_url(f"https://{host}/", policy)


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "fd00:ec2::254",
        "::ffff:127.0.0.1",
        "2002:7f00:0001::",
        "198.18.0.1",
    ],
)
def test_non_public_addresses(ip: str) -> None:
    assert not is_public_ip(ipaddress.ip_address(ip))


@pytest.mark.parametrize("ip", ["93.184.216.34", "1.1.1.1", "2606:4700:4700::1111"])
def test_public_addresses(ip: str) -> None:
    assert is_public_ip(ipaddress.ip_address(ip))


async def test_resolution_to_private_address_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_getaddrinfo(*_a: object, **_k: object) -> list[tuple[object, ...]]:
        return [(2, 1, 6, "", ("10.1.2.3", 443))]

    import asyncio

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(EgressDenied, match="non-public"):
        await ssrf.resolve_checked(Target("https", "api.example.com", 443, private_allowed=False))


async def test_private_allowlisted_service_may_resolve_privately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_getaddrinfo(*_a: object, **_k: object) -> list[tuple[object, ...]]:
        return [(2, 1, 6, "", ("172.20.0.7", 11434))]

    import asyncio

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    assert (
        await ssrf.resolve_checked(Target("http", "ollama", 11434, private_allowed=True))
        == "172.20.0.7"
    )


async def test_transport_pins_vetted_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    async def fake_resolve(target: Target) -> str:
        return "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"ok")

    monkeypatch.setattr(ssrf, "resolve_checked", fake_resolve)
    transport = ssrf.GuardedTransport(POLICY, inner=httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://api.example.com/v1/x")
    assert response.status_code == 200
    assert seen[0].url.host == "93.184.216.34"
    assert seen[0].headers["host"] == "api.example.com"
    assert seen[0].extensions["sni_hostname"] == "api.example.com"


async def test_fetch_revalidates_redirect_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(target: Target) -> str:
        return "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://169.254.169.254/latest/meta-data"})

    monkeypatch.setattr(ssrf, "resolve_checked", fake_resolve)
    client = httpx.AsyncClient(
        transport=ssrf.GuardedTransport(POLICY, inner=httpx.MockTransport(handler)),
        follow_redirects=False,
    )
    with pytest.raises(EgressDenied):
        await ssrf.fetch(
            "https://api.example.com/doc",
            POLICY,
            allowed_content_types=["text/plain"],
            client=client,
        )
    await client.aclose()


async def test_fetch_enforces_size_and_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(target: Target) -> str:
        return "93.184.216.34"

    bodies = {
        "/big": httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 5_000),
        "/html": httpx.Response(200, headers={"content-type": "text/html"}, content=b"<h1>"),
        "/ok": httpx.Response(
            200, headers={"content-type": "text/plain; charset=utf-8"}, content=b"fine"
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return bodies[request.url.path]

    monkeypatch.setattr(ssrf, "resolve_checked", fake_resolve)
    client = httpx.AsyncClient(
        transport=ssrf.GuardedTransport(POLICY, inner=httpx.MockTransport(handler))
    )
    with pytest.raises(EgressDenied, match="too large"):
        await ssrf.fetch(
            "https://api.example.com/big",
            POLICY,
            allowed_content_types=["text/plain"],
            client=client,
        )
    with pytest.raises(EgressDenied, match="content-type"):
        await ssrf.fetch(
            "https://api.example.com/html",
            POLICY,
            allowed_content_types=["text/plain"],
            client=client,
        )
    result = await ssrf.fetch(
        "https://api.example.com/ok", POLICY, allowed_content_types=["text/plain"], client=client
    )
    assert result.content == b"fine"
    await client.aclose()
