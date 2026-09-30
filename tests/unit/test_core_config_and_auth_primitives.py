"""Settings validation, password policy/hashing, TOTP and the GCRA rate limiter."""

from __future__ import annotations

import base64
import secrets
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fakeredis
import pytest
from pydantic import ValidationError

from docassist.cache.ratelimit import RateLimiter
from docassist.core.config import RateRule, Settings, _file_overrides, parse_encryption_keys
from docassist.core.errors import RateLimited
from docassist.security import totp
from docassist.security.passwords import PasswordPolicy, PasswordPolicyError, PasswordService
from tests.conftest import make_settings


def _prod(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "production",
        "public_base_url": "https://docs.example.com",
        "security": {
            "cookie_secure": True,
            "allowed_hosts": ["docs.example.com"],
            "expose_api_docs": False,
        },
        "database": {"url": "postgresql+asyncpg://a@db/x", "ssl": "verify-full"},
        "redis": {"url": "redis://:pw@redis:6379/0"},
        "upload": {"malware_scanner": "clamav"},
        "llm": {"provider": "anthropic"},
        "embedding": {"provider": "openai_compatible", "base_url": "https://emb.example.com/v1"},
    }
    base.update(over)
    return make_settings(**base)


def test_valid_production_settings() -> None:
    assert _prod().is_production


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        (
            {"security": {"allowed_hosts": ["*"], "cookie_secure": True, "expose_api_docs": False}},
            "allowed_hosts",
        ),
        (
            {"security": {"expose_api_docs": True, "allowed_hosts": ["x"], "cookie_secure": True}},
            "expose_api_docs",
        ),
        ({"redis": {"url": None}}, "redis.url"),
        ({"upload": {"malware_scanner": "none"}}, "malware_scanner"),
        ({"llm": {"provider": "local_extractive"}}, "offline AI"),
        ({"database": {"url": "postgresql+asyncpg://a@db/x", "ssl": "prefer"}}, "database.ssl"),
        ({"public_base_url": "http://docs.example.com"}, "https"),
    ],
)
def test_unsafe_production_settings_refused(override: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        _prod(**override)


def test_weak_or_placeholder_secrets_refused() -> None:
    for bad in ("short", "change-me-" + "a" * 40, "a" * 64):
        with pytest.raises(ValidationError):
            make_settings(security={"jwt_signing_key": bad})


def test_secret_values_never_in_error_messages() -> None:
    secret = "change-me-" + "Zx9!" * 10
    with pytest.raises(ValidationError) as info:
        make_settings(security={"jwt_signing_key": secret})
    assert secret not in str(info.value)


def test_encryption_key_parsing() -> None:
    k = base64.b64encode(secrets.token_bytes(32)).decode()
    assert set(parse_encryption_keys(f"a:{k}, b:{k}")) == {"a", "b"}
    for bad in ("", "nokid", f"a:{k},a:{k}", "a:!!!", "a:" + base64.b64encode(b"short").decode()):
        with pytest.raises(ValueError):
            parse_encryption_keys(bad)


def test_file_secrets(tmp_path: Path) -> None:
    secret_file = tmp_path / "jwt"
    secret_file.write_text("from-file-value\n", encoding="utf-8")
    overrides = _file_overrides(
        {"DOCASSIST_SECURITY__JWT_SIGNING_KEY_FILE": str(secret_file), "OTHER_FILE": "x"}
    )
    assert overrides == {"security": {"jwt_signing_key": "from-file-value"}}


def test_password_policy() -> None:
    policy = PasswordPolicy(min_length=12)
    policy.validate("correct-battery-staple-9", email="a@b.c")
    for bad, email in (
        ("short1!", None),
        ("password1234", None),
        ("Password123!", None),
        ("aaaaaaaaaaaaaaaa", None),
        ("jsmith-is-my-password", "jsmith@corp.example"),
    ):
        with pytest.raises(PasswordPolicyError):
            policy.validate(bad, email=email)


def test_password_hashing() -> None:
    svc = PasswordService(time_cost=1, memory_kib=8192, parallelism=1)
    hashed = svc.hash("a very long passphrase")
    assert hashed.startswith("$argon2id$")
    assert svc.verify(hashed, "a very long passphrase")
    assert not svc.verify(hashed, "a very long passphrasE")
    assert not svc.verify(None, "anything")  # unknown account path
    assert not svc.verify("not-a-hash", "x")


def test_totp_rfc6238_vector_and_replay() -> None:
    secret = base64.b32encode(b"12345678901234567890").decode()
    assert totp.code_at(secret, 59 // 30) == "287082"  # RFC 6238 test vector (SHA1, 6 digits)
    now = 1_700_000_000.0
    step = totp.current_step(now)
    code = totp.code_at(secret, step)
    assert totp.verify(secret, code, last_used_step=None, now=now) == step
    assert totp.verify(secret, code, last_used_step=step, now=now) is None  # replay
    assert totp.verify(secret, "12345", last_used_step=None, now=now) is None
    assert totp.verify(secret, "abcdef", last_used_step=None, now=now) is None
    assert totp.provisioning_uri(secret, "a@b.c", "Doc Assist").startswith("otpauth://totp/")


async def test_rate_limiter_redis_burst_then_block() -> None:
    limiter = RateLimiter(fakeredis.FakeAsyncRedis(), "t")
    rule = RateRule(requests=3, per_seconds=60)
    for _ in range(3):
        await limiter.enforce("login", "1.2.3.4", rule)
    with pytest.raises(RateLimited) as info:
        await limiter.enforce("login", "1.2.3.4", rule)
    assert info.value.retry_after >= 1
    await limiter.enforce("login", "5.6.7.8", rule)  # independent identity


async def test_rate_limiter_falls_back_when_redis_down() -> None:
    class Broken(fakeredis.FakeAsyncRedis):  # type: ignore[misc]
        pass

    redis = Broken()
    limiter = RateLimiter(redis, "t")
    await redis.aclose()
    await redis.connection_pool.disconnect()

    async def boom(*_a: object, **_k: object) -> None:
        from redis.exceptions import ConnectionError as RedisConnectionError

        raise RedisConnectionError("down")

    limiter._script = boom  # type: ignore[assignment]
    rule = RateRule(requests=2, per_seconds=60)
    await limiter.enforce("x", "id", rule)
    await limiter.enforce("x", "id", rule)
    with pytest.raises(RateLimited):
        await limiter.enforce("x", "id", rule)  # still limited: no fail-open


async def test_local_rate_limiter_admits_a_new_key_at_any_clock_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # At this reading a float clock made `now + emission - tolerance` round above `now`, so a
    # new key's first request was rejected (about 0.2% of readings for a one-request rule).
    seconds = 1024.074115
    clock = SimpleNamespace(monotonic=lambda: seconds, monotonic_ns=lambda: int(seconds * 1e9))
    monkeypatch.setattr("docassist.cache.ratelimit.time", clock)
    limiter = RateLimiter(None, "t")
    rule = RateRule(requests=1, per_seconds=3600)
    await limiter.enforce("llm_org", "org-a", rule)
    await limiter.enforce("llm_org", "org-b", rule)
    with pytest.raises(RateLimited):
        await limiter.enforce("llm_org", "org-a", rule)


async def test_rate_limiter_disabled() -> None:
    limiter = RateLimiter(None, "t", enabled=False)
    for _ in range(10):
        await limiter.enforce("x", "id", RateRule(requests=1, per_seconds=60))
