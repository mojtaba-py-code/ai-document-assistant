"""JWT hardening and opaque-token hashing."""

from __future__ import annotations

import time
import uuid

import jwt
import pytest

from docassist.security.tokens import TokenError, TokenService

KEY = "k" * 16 + "Q9$z" * 8


def service(**kw: object) -> TokenService:
    params: dict[str, object] = {
        "signing_key": KEY,
        "previous_keys": [],
        "issuer": "docassist",
        "audience": "docassist-api",
        "access_ttl_seconds": 600,
        "pepper": "p" * 40,
    }
    params.update(kw)
    return TokenService(**params)  # type: ignore[arg-type]


def issue(svc: TokenService) -> str:
    token, _ = svc.issue_access_token(
        user_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        role="employee",
        session_id=uuid.uuid4(),
        token_version=3,
    )
    return token


def test_round_trip() -> None:
    svc = service()
    uid, org, sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    token, _ = svc.issue_access_token(
        user_id=uid, org_id=org, role="auditor", session_id=sid, token_version=7
    )
    claims = svc.verify_access_token(token)
    assert (
        claims.user_id,
        claims.org_id,
        claims.role,
        claims.session_id,
        claims.token_version,
    ) == (
        uid,
        org,
        "auditor",
        sid,
        7,
    )


def test_platform_token_has_no_org() -> None:
    svc = service()
    token, _ = svc.issue_access_token(
        user_id=uuid.uuid4(),
        org_id=None,
        role="platform_admin",
        session_id=uuid.uuid4(),
        token_version=1,
    )
    assert svc.verify_access_token(token).org_id is None


@pytest.mark.filterwarnings("ignore::jwt.warnings.InsecureKeyLengthWarning")  # deliberate forgery
@pytest.mark.parametrize("alg", ["none", "HS512"])
def test_other_algorithms_rejected(alg: str) -> None:
    svc = service()
    claims = jwt.decode(issue(svc), options={"verify_signature": False})
    forged = jwt.encode(
        claims, key="" if alg == "none" else KEY, algorithm=alg, headers={"kid": "x"}
    )
    with pytest.raises(TokenError):
        svc.verify_access_token(forged)


def test_wrong_audience_and_issuer_rejected() -> None:
    other = service(audience="someone-else")
    with pytest.raises(TokenError):
        service().verify_access_token(issue(other))
    with pytest.raises(TokenError):
        service().verify_access_token(issue(service(issuer="evil")))


def test_expired_token_rejected() -> None:
    svc = service(access_ttl_seconds=60)
    claims = jwt.decode(issue(svc), options={"verify_signature": False})
    claims["exp"] = int(time.time()) - 120
    header = jwt.get_unverified_header(issue(svc))
    expired = jwt.encode(claims, KEY, algorithm="HS256", headers={"kid": header["kid"]})
    with pytest.raises(TokenError):
        svc.verify_access_token(expired)


def test_missing_claim_rejected() -> None:
    svc = service()
    claims = jwt.decode(issue(svc), options={"verify_signature": False})
    del claims["sid"]
    header = jwt.get_unverified_header(issue(svc))
    token = jwt.encode(claims, KEY, algorithm="HS256", headers={"kid": header["kid"]})
    with pytest.raises(TokenError):
        svc.verify_access_token(token)


def test_key_rotation() -> None:
    old = service()
    token = issue(old)
    new_key = "n" * 20 + "R7#w" * 8
    rotated = service(signing_key=new_key, previous_keys=[KEY])
    assert rotated.verify_access_token(token)  # old tokens survive the rotation window
    dropped = service(signing_key=new_key)
    with pytest.raises(TokenError):
        dropped.verify_access_token(token)


def test_oversized_token_rejected() -> None:
    with pytest.raises(TokenError):
        service().verify_access_token("a" * 5000)


def test_opaque_hash_is_keyed() -> None:
    a, b = service(), service(pepper="q" * 40)
    token = TokenService.new_opaque_token()
    assert len(token) >= 43
    assert a.hash_opaque(token) == a.hash_opaque(token)
    assert a.hash_opaque(token) != b.hash_opaque(token)


def test_signatures() -> None:
    svc = service()
    sig = svc.sign("export", "abc|1")
    assert svc.verify_signature("export", "abc|1", sig)
    assert not svc.verify_signature("export", "abc|2", sig)
    assert not svc.verify_signature("other", "abc|1", sig)
