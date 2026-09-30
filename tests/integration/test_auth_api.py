"""Authentication flows through the real HTTP API and database."""

from __future__ import annotations

import pytest

from docassist.security import totp
from tests.conftest import login

pytestmark = [pytest.mark.db]


async def test_login_me_logout(client, factory) -> None:
    org = await factory.org()
    user = await factory.user(org, "employee")
    headers = await login(client, user)
    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == user.email
    assert "document:read" in body["permissions"]
    assert body["mfa_enabled"] is False
    assert body["full_name"]
    assert (await client.post("/api/v1/auth/logout", headers=headers)).status_code == 204
    # the access token dies with its session - immediately, not after its TTL
    assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 401


async def test_wrong_password_and_unknown_user_look_identical(client, factory) -> None:
    org = await factory.org()
    user = await factory.user(org)
    wrong = await client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": "nope-nope-nope"}
    )
    unknown = await client.post(
        "/api/v1/auth/login", json={"email": "ghost@example.test", "password": "nope-nope"}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


async def test_account_lockout(client, factory, container) -> None:
    org = await factory.org()
    user = await factory.user(org)
    threshold = container.settings.security.lockout_threshold
    for _ in range(threshold):
        await client.post(
            "/api/v1/auth/login", json={"email": user.email, "password": "wrong-password-x"}
        )
    locked = await client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": user.password}
    )
    assert locked.status_code == 401  # correct password is refused while locked


async def test_refresh_rotation_and_reuse_detection(client, factory) -> None:
    org = await factory.org()
    user = await factory.user(org)
    first = await client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": user.password, "token_transport": "body"},
    )
    refresh_1 = first.json()["refresh_token"]
    rotated = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": refresh_1, "token_transport": "body"}
    )
    assert rotated.status_code == 200
    refresh_2 = rotated.json()["refresh_token"]
    assert refresh_2 != refresh_1
    # replaying the old token is treated as theft: the whole session dies
    replay = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": refresh_1, "token_transport": "body"}
    )
    assert replay.status_code == 401
    after = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": refresh_2, "token_transport": "body"}
    )
    assert after.status_code == 401
    access = rotated.json()["access_token"]
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})
    ).status_code == 401


async def test_cookie_refresh_requires_csrf_header(client, factory) -> None:
    org = await factory.org()
    user = await factory.user(org)
    await client.post("/api/v1/auth/login", json={"email": user.email, "password": user.password})
    assert "docassist_refresh" in client.cookies
    no_header = await client.post("/api/v1/auth/refresh")
    assert no_header.status_code == 403
    ok = await client.post("/api/v1/auth/refresh", headers={"X-CSRF-Protection": "1"})
    assert ok.status_code == 200
    assert "refresh_token" not in ok.json()


async def test_password_reset_flow(client, factory, container) -> None:
    org = await factory.org()
    user = await factory.user(org)
    old_headers = await login(client, user)
    container.email.sent.clear()
    response = await client.post("/api/v1/auth/password/forgot", json={"email": user.email})
    assert response.status_code == 202
    await container.auth.wait_for_background()  # delivery runs off the request path
    ghost = await client.post("/api/v1/auth/password/forgot", json={"email": "nobody@example.test"})
    assert ghost.status_code == 202 and ghost.json() == response.json()
    await container.auth.wait_for_background()
    assert len(container.email.sent) == 1
    token = container.email.sent[0][2].split("token=")[1].split()[0]
    new_password = "N3w-and-very-different-passphrase"
    done = await client.post(
        "/api/v1/auth/password/reset", json={"token": token, "new_password": new_password}
    )
    assert done.status_code == 204
    again = await client.post(
        "/api/v1/auth/password/reset", json={"token": token, "new_password": new_password + "x"}
    )
    assert again.status_code == 422  # single use
    assert (await client.get("/api/v1/auth/me", headers=old_headers)).status_code == 401
    user.password = new_password
    assert (
        await client.get("/api/v1/auth/me", headers=await login(client, user))
    ).status_code == 200


async def test_weak_password_rejected_without_echo(client, factory) -> None:
    org = await factory.org()
    user = await factory.user(org)
    headers = await login(client, user)
    response = await client.post(
        "/api/v1/auth/password/change",
        headers=headers,
        json={"current_password": user.password, "new_password": "password123"},
    )
    assert response.status_code == 422
    assert "password123" not in response.text


async def test_mfa_enrollment_and_login(client, factory) -> None:
    org = await factory.org()
    user = await factory.user(org)
    headers = await login(client, user)
    enroll = await client.post(
        "/api/v1/auth/mfa/enroll", headers=headers, json={"password": user.password}
    )
    secret = enroll.json()["secret"]
    step = totp.current_step()
    confirm = await client.post(
        "/api/v1/auth/mfa/confirm", headers=headers, json={"code": totp.code_at(secret, step)}
    )
    assert confirm.status_code == 204
    challenge = await client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": user.password}
    )
    body = challenge.json()
    assert body["mfa_required"] is True and "access_token" not in body
    # the code used for enrollment cannot be replayed
    replay = await client.post(
        "/api/v1/auth/mfa/verify",
        json={
            "challenge": body["mfa_challenge"],
            "code": totp.code_at(secret, step),
            "token_transport": "body",
        },
    )
    assert replay.status_code == 401
    fresh = await client.post(
        "/api/v1/auth/mfa/verify",
        json={
            "challenge": body["mfa_challenge"],
            "code": totp.code_at(secret, step + 1),
            "token_transport": "body",
        },
    )
    assert fresh.status_code == 200, fresh.text
    assert fresh.json()["access_token"]


async def test_tampered_and_foreign_tokens_rejected(client, factory) -> None:
    import jwt

    org = await factory.org()
    user = await factory.user(org, "employee")
    headers = await login(client, user)
    token = headers["Authorization"].split()[1]
    claims = jwt.decode(token, options={"verify_signature": False})
    claims["role"] = "organization_admin"
    forged_none = jwt.encode(claims, key="", algorithm="none")
    forged_key = jwt.encode(claims, key="x" * 40, algorithm="HS256")
    for bad in (forged_none, forged_key, token[:-4] + "AAAA", "not-a-token"):
        response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {bad}"})
        assert response.status_code == 401


async def test_disabled_user_is_rejected(client, factory) -> None:
    org = await factory.org()
    user = await factory.user(org, status="disabled")
    response = await client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": user.password}
    )
    assert response.status_code == 401


async def test_security_headers_present(client) -> None:
    response = await client.get("/health/live")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-request-id"]
