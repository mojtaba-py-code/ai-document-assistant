"""Input schemas, tag/role normalisation and the download ``Content-Disposition`` header."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from urllib.parse import unquote

import pytest
from pydantic import ValidationError

from docassist.api.routers.documents import content_disposition
from docassist.core.errors import ValidationFailed
from docassist.documents.schemas import DocumentUpdate, GrantCreate, UploadForm
from docassist.documents.service import normalize_roles, normalize_tags


def test_update_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DocumentUpdate.model_validate({"title": "x", "owner_id": str(uuid.uuid4())})


def test_update_tracks_explicit_nulls() -> None:
    patch = DocumentUpdate.model_validate({"department_id": None, "retention_until": None})
    assert patch.model_fields_set == {"department_id", "retention_until"}
    with pytest.raises(ValidationError):
        DocumentUpdate.model_validate({"classification": None})
    with pytest.raises(ValidationError):
        DocumentUpdate.model_validate({"legal_hold": None})


def test_upload_form_limits() -> None:
    with pytest.raises(ValidationError):
        UploadForm.model_validate({"classification": "SECRET"})
    with pytest.raises(ValidationError):
        UploadForm.model_validate({"classification": "INTERNAL", "tags": ["t"] * 21})
    with pytest.raises(ValidationError):
        UploadForm.model_validate({"classification": "INTERNAL", "extra": "x"})
    form = UploadForm.model_validate(
        {"classification": "INTERNAL", "allow_duplicate": "true", "allowed_roles": ["employee"]}
    )
    assert form.allow_duplicate is True


@pytest.mark.parametrize(
    "payload",
    [
        {"grantee_type": "user"},
        {"grantee_type": "user", "user_id": str(uuid.uuid4()), "role": "employee"},
        {"grantee_type": "department", "user_id": str(uuid.uuid4())},
        {"grantee_type": "role", "role": "platform_admin"},
        {"grantee_type": "role", "role": "employee", "expires_at": "2030-01-01T00:00:00"},
        {"grantee_type": "role", "role": "employee", "permission": "owner"},
    ],
)
def test_grant_shape_is_strict(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GrantCreate.model_validate(payload)


def test_valid_grant() -> None:
    grant = GrantCreate.model_validate(
        {"grantee_type": "role", "role": "auditor", "expires_at": "2030-01-01T00:00:00Z"}
    )
    assert grant.expires_at == datetime(2030, 1, 1, tzinfo=UTC)


def test_normalize_tags() -> None:
    zero_width = chr(0x200B)
    assert normalize_tags(["Finance", " finance ", f"Q{zero_width}3", "", "x" * 100]) == [
        "finance",
        "q3",
        "x" * 64,
    ]
    with pytest.raises(ValidationFailed):
        normalize_tags([f"t{i}" for i in range(21)])


def test_normalize_roles() -> None:
    assert normalize_roles(["employee", "EMPLOYEE", "auditor"]) == ["auditor", "employee"]
    with pytest.raises(ValidationFailed):
        normalize_roles(["platform_admin"])
    with pytest.raises(ValidationFailed):
        normalize_roles(["superuser"])


@pytest.mark.parametrize(
    ("name", "fallback"),
    [
        ("report.pdf", "report.pdf"),
        ("r" + chr(0xE9) + "sum" + chr(0xE9) + ".pdf", "resume.pdf"),
        (chr(0x62A5) + chr(0x544A) + ".docx", "document.docx"),
        ('a"b;c.txt', "a_b_c.txt"),
    ],
)
def test_content_disposition(name: str, fallback: str) -> None:
    header = content_disposition(name)
    assert header.startswith(f"attachment; filename=\"{fallback}\"; filename*=UTF-8''")
    encoded = header.split("filename*=UTF-8''", 1)[1]
    assert unquote(encoded) == name
    assert all(ch.isascii() and ch not in '"; \\' for ch in encoded)
