"""Authentication endpoints.

Browser clients receive the refresh token in an ``HttpOnly; Secure; SameSite=Strict``
cookie (``__Host-`` prefixed in production) and keep the short-lived access token in memory
only. Cookie-based refresh/logout additionally require the ``X-CSRF-Protection: 1`` header,
which a cross-site form cannot set. API clients may ask for ``token_transport="body"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from docassist.api.container import Container
from docassist.api.deps import client_ip, get_container, get_principal
from docassist.authz.principal import Principal
from docassist.core.errors import AuthenticationFailed, PermissionDenied
from docassist.identity.auth import TokenPair

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_CSRF_HEADER = "x-csrf-protection"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class LoginRequest(_Strict):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)
    token_transport: Literal["cookie", "body"] = "cookie"


class MfaVerifyRequest(_Strict):
    challenge: str = Field(min_length=20, max_length=200)
    code: str = Field(min_length=6, max_length=10)
    token_transport: Literal["cookie", "body"] = "cookie"


class RefreshRequest(_Strict):
    refresh_token: str | None = Field(default=None, min_length=20, max_length=200)
    token_transport: Literal["cookie", "body"] = "cookie"


class LogoutRequest(_Strict):
    everywhere: bool = False


class ChangePasswordRequest(_Strict):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=1, max_length=1024)


class ForgotPasswordRequest(_Strict):
    email: str = Field(min_length=3, max_length=320)


class ResetPasswordRequest(_Strict):
    token: str = Field(min_length=20, max_length=200)
    new_password: str = Field(min_length=1, max_length=1024)


class MfaCodeRequest(_Strict):
    code: str = Field(min_length=6, max_length=10)


class MfaEnrollRequest(_Strict):
    password: str = Field(min_length=1, max_length=1024)


class MfaDisableRequest(_Strict):
    password: str = Field(min_length=1, max_length=1024)
    code: str = Field(min_length=6, max_length=10)


class TokenResponse(BaseModel):
    access_token: str | None = None
    token_type: str = "Bearer"
    expires_at: datetime | None = None
    refresh_token: str | None = None
    refresh_expires_at: datetime | None = None
    mfa_required: bool = False
    mfa_challenge: str | None = None


class MeResponse(BaseModel):
    id: str
    email: str
    full_name: str
    mfa_enabled: bool
    organization_id: str | None
    role: str
    clearance: str
    permissions: list[str]
    department_ids: list[str]
    managed_department_ids: list[str]


class MfaEnrollResponse(BaseModel):
    secret: str
    provisioning_uri: str


def _cookie_name(container: Container) -> str:
    return (
        "__Host-docassist_refresh"
        if container.settings.security.cookie_secure
        else "docassist_refresh"
    )


def _set_refresh_cookie(response: Response, container: Container, pair: TokenPair) -> None:
    max_age = max(1, int((pair.refresh_expires_at - pair.access_expires_at).total_seconds()) + 600)
    response.set_cookie(
        _cookie_name(container),
        pair.refresh_token,
        max_age=max_age,
        path="/",
        secure=container.settings.security.cookie_secure,
        httponly=True,
        samesite="strict",
    )


def _clear_refresh_cookie(response: Response, container: Container) -> None:
    response.delete_cookie(
        _cookie_name(container),
        path="/",
        secure=container.settings.security.cookie_secure,
        httponly=True,
        samesite="strict",
    )


def _token_response(
    pair: TokenPair, transport: str, response: Response, container: Container
) -> TokenResponse:
    body = TokenResponse(access_token=pair.access_token, expires_at=pair.access_expires_at)
    if transport == "body":
        body.refresh_token = pair.refresh_token
        body.refresh_expires_at = pair.refresh_expires_at
    else:
        _set_refresh_cookie(response, container, pair)
    return body


@router.post("/login", response_model=TokenResponse, response_model_exclude_none=True)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    container: Container = Depends(get_container),
) -> TokenResponse:
    result = await container.auth.login(
        payload.email,
        payload.password,
        client_ip=client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    if result.mfa_challenge is not None:
        return TokenResponse(
            mfa_required=True,
            mfa_challenge=result.mfa_challenge,
            # "None" is a token *type* label, not a credential
            token_type="None",  # nosec B106
        )
    if result.tokens is None:  # defensive: the service returns tokens or a challenge
        raise AuthenticationFailed()
    return _token_response(result.tokens, payload.token_transport, response, container)


@router.post("/mfa/verify", response_model=TokenResponse, response_model_exclude_none=True)
async def verify_mfa(
    payload: MfaVerifyRequest,
    request: Request,
    response: Response,
    container: Container = Depends(get_container),
) -> TokenResponse:
    pair = await container.auth.complete_mfa(
        payload.challenge,
        payload.code,
        client_ip=client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    return _token_response(pair, payload.token_transport, response, container)


@router.post("/refresh", response_model=TokenResponse, response_model_exclude_none=True)
async def refresh(
    request: Request,
    response: Response,
    payload: RefreshRequest | None = None,
    container: Container = Depends(get_container),
) -> TokenResponse:
    payload = payload or RefreshRequest()
    token = payload.refresh_token
    transport = payload.token_transport
    if token is None:
        if request.headers.get(_CSRF_HEADER) != "1":
            raise PermissionDenied("Missing CSRF protection header.")
        token = request.cookies.get(_cookie_name(container))
        transport = "cookie"
    if not token:
        raise AuthenticationFailed("Your session has expired. Please sign in again.")
    try:
        pair = await container.auth.refresh(token, client_ip=client_ip(request))
    except AuthenticationFailed:
        _clear_refresh_cookie(response, container)
        raise
    return _token_response(pair, transport, response, container)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    payload: LogoutRequest | None = None,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> Response:
    await container.auth.logout(principal, everywhere=bool(payload and payload.everywhere))
    out = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_refresh_cookie(out, container)
    return out


@router.get("/me", response_model=MeResponse)
async def me(
    principal: Principal = Depends(get_principal), container: Container = Depends(get_container)
) -> MeResponse:
    full_name, mfa_enabled = await container.auth.profile(principal)
    return MeResponse(
        id=str(principal.user_id),
        email=principal.email,
        full_name=full_name,
        mfa_enabled=mfa_enabled,
        organization_id=str(principal.org_id) if principal.org_id else None,
        role=principal.role.value,
        clearance=principal.clearance.value,
        permissions=sorted(p.value for p in principal.permissions),
        department_ids=sorted(str(d) for d in principal.department_ids),
        managed_department_ids=sorted(str(d) for d in principal.managed_department_ids),
    )


@router.post("/password/change", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    payload: ChangePasswordRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> Response:
    await container.auth.change_password(principal, payload.current_password, payload.new_password)
    out = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_refresh_cookie(out, container)
    return out


@router.post("/password/forgot", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(
    payload: ForgotPasswordRequest, request: Request, container: Container = Depends(get_container)
) -> dict[str, str]:
    await container.auth.request_password_reset(payload.email, client_ip=client_ip(request))
    return {"detail": "If the account exists, a reset link has been sent."}


@router.post("/password/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_password(
    payload: ResetPasswordRequest, container: Container = Depends(get_container)
) -> Response:
    await container.auth.reset_password(payload.token, payload.new_password)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/mfa/enroll", response_model=MfaEnrollResponse)
async def mfa_enroll(
    payload: MfaEnrollRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> MfaEnrollResponse:
    enrollment = await container.auth.begin_mfa_enrollment(principal, payload.password)
    return MfaEnrollResponse(secret=enrollment.secret, provisioning_uri=enrollment.provisioning_uri)


@router.post("/mfa/confirm", status_code=status.HTTP_204_NO_CONTENT)
async def mfa_confirm(
    payload: MfaCodeRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> Response:
    await container.auth.confirm_mfa(principal, payload.code)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/mfa/disable", status_code=status.HTTP_204_NO_CONTENT)
async def mfa_disable(
    payload: MfaDisableRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> Response:
    await container.auth.disable_mfa(principal, payload.password, payload.code)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
