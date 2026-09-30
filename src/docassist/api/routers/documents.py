"""Documents API (``/api/v1/documents``).

Multipart uploads are parsed inside the handlers, *after* authentication and the permission
check: FastAPI would otherwise read and spool the whole request body before running any
dependency, letting anonymous clients make the server buffer files. The form is limited to
one file and a handful of fields, and the file is handed to the service as a chunked stream
(never read into memory at once).

Downloads are always ``Content-Disposition: attachment`` with an ASCII fallback name and an
RFC 5987 ``filename*``, served as the *detected* MIME type with ``nosniff``, a sandboxing
Content-Security-Policy and ``Cache-Control: no-store``.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from starlette.datastructures import FormData, UploadFile

from docassist.api.container import Container
from docassist.api.deps import get_container, require
from docassist.authz.permissions import Permission
from docassist.authz.principal import Principal
from docassist.core.enums import Classification, DocumentStatus, DocumentType, Role
from docassist.core.errors import ValidationFailed
from docassist.documents.schemas import (
    ContentPage,
    DocumentDetail,
    DocumentPage,
    DocumentUpdate,
    GrantCreate,
    GrantInfo,
    UploadForm,
    UploadResult,
    UrlImportRequest,
    VersionForm,
)
from docassist.documents.service import DownloadPayload
from docassist.documents.validation import extension_of

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

_UPLOAD_CHUNK = 256 * 1024
_MAX_FORM_FIELDS = 16
_LIST_SEPARATORS = re.compile(r"[,\n]")
_UNSAFE_ASCII_FILENAME = re.compile(r"[^A-Za-z0-9._ ()-]")
DOWNLOAD_CSP = "sandbox; default-src 'none'"

_READ = Depends(require(Permission.DOCUMENT_READ))
_UPLOAD = Depends(require(Permission.DOCUMENT_UPLOAD))
_CONTAINER = Depends(get_container)


def _multipart_schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file", *required],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            **properties,
                        },
                    }
                }
            },
        }
    }


_UPLOAD_OPENAPI = _multipart_schema(
    ["classification"],
    {
        "title": {"type": "string", "maxLength": 300},
        "classification": {"type": "string", "enum": [c.value for c in Classification]},
        "department_id": {"type": "string", "format": "uuid"},
        "doc_type": {"type": "string", "enum": [t.value for t in DocumentType]},
        "tags": {"type": "string", "description": "Comma-separated, at most 20"},
        "allowed_roles": {
            "type": "string",
            "description": "Comma-separated roles: "
            + ", ".join(r.value for r in Role if r is not Role.PLATFORM_ADMIN),
        },
        "allow_duplicate": {"type": "boolean", "default": False},
    },
)
_VERSION_OPENAPI = _multipart_schema(
    [],
    {
        "change_note": {"type": "string", "maxLength": 500},
        "allow_duplicate": {"type": "boolean", "default": False},
    },
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _file_chunks(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(_UPLOAD_CHUNK):
        yield chunk


def _form_fields(form: FormData, list_fields: frozenset[str]) -> tuple[UploadFile, dict[str, Any]]:
    """Split a multipart form into its single file and its (non-empty) text fields."""
    upload: UploadFile | None = None
    data: dict[str, Any] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            if key != "file" or upload is not None:
                raise ValidationFailed("Upload exactly one file, in the 'file' field.")
            upload = value
            continue
        if key == "file" or key in data:
            raise ValidationFailed("The form contains duplicate or misplaced fields.")
        text = value.strip()
        if not text:
            continue  # an empty optional form field means "not provided"
        if key in list_fields:
            data[key] = [part.strip() for part in _LIST_SEPARATORS.split(text) if part.strip()]
        else:
            data[key] = text
    if upload is None:
        raise ValidationFailed("Upload exactly one file, in the 'file' field.")
    return upload, data


def _validated[M: BaseModel](model: type[M], data: dict[str, Any]) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def content_disposition(filename: str) -> str:
    """``attachment`` with a safe ASCII ``filename`` and the exact UTF-8 ``filename*``."""
    extension = extension_of(filename)
    stem = filename[: -(len(extension) + 1)] if extension else filename
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    ascii_stem = _UNSAFE_ASCII_FILENAME.sub("_", ascii_stem).strip(" ._") or "document"
    fallback = f"{ascii_stem}.{extension}" if extension else ascii_stem
    # quote(safe="") leaves only RFC 5987 attr-chars (ALPHA / DIGIT / "-._~") unescaped.
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


def _download_response(payload: DownloadPayload) -> StreamingResponse:
    headers = {
        "Content-Disposition": content_disposition(payload.filename),
        "Content-Length": str(payload.size),
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": DOWNLOAD_CSP,
        "Cache-Control": "no-store",
    }
    if payload.quarantined:
        headers["X-Docassist-Quarantined"] = "true"
    return StreamingResponse(payload.chunks, media_type=payload.content_type, headers=headers)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=UploadResult,
    openapi_extra=_UPLOAD_OPENAPI,
)
async def upload_document(
    request: Request,
    response: Response,
    principal: Principal = _UPLOAD,
    container: Container = _CONTAINER,
) -> UploadResult:
    form = await request.form(max_files=1, max_fields=_MAX_FORM_FIELDS)
    try:
        upload, data = _form_fields(form, frozenset({"tags", "allowed_roles"}))
        fields = _validated(UploadForm, data)
        result = await container.documents.upload(
            principal,
            stream=_file_chunks(upload),
            filename=upload.filename or "",
            declared_mime=upload.content_type,
            title=fields.title,
            classification=fields.classification,
            department_id=fields.department_id,
            doc_type=fields.doc_type,
            tags=fields.tags,
            allowed_roles=fields.allowed_roles,
            allow_duplicate=fields.allow_duplicate,
        )
    finally:
        await form.close()
    response.headers["Location"] = f"{router.prefix}/{result.document_id}"
    return result


@router.post("/import-url", status_code=status.HTTP_201_CREATED, response_model=UploadResult)
async def import_document_from_url(
    payload: UrlImportRequest,
    response: Response,
    principal: Principal = _UPLOAD,
    container: Container = _CONTAINER,
) -> UploadResult:
    result = await container.documents.import_url(
        principal,
        payload.url,
        title=payload.title,
        classification=payload.classification,
        department_id=payload.department_id,
        doc_type=payload.doc_type,
        tags=payload.tags,
        allowed_roles=payload.allowed_roles,
        allow_duplicate=payload.allow_duplicate,
    )
    response.headers["Location"] = f"{router.prefix}/{result.document_id}"
    return result


@router.get("", response_model=DocumentPage)
async def list_documents(
    *,
    q: str | None = Query(None, max_length=200),
    doc_type: DocumentType | None = Query(None),
    classification: Classification | None = Query(None),
    department_id: uuid.UUID | None = Query(None),
    status_filter: DocumentStatus | None = Query(None, alias="status"),
    tag: str | None = Query(None, max_length=64),
    owner: Literal["me"] | None = Query(None),
    cursor: str | None = Query(None, max_length=256),
    limit: int = Query(25, ge=1, le=100),
    principal: Principal = _READ,
    container: Container = _CONTAINER,
) -> DocumentPage:
    return await container.documents.list_documents(
        principal,
        q=q,
        doc_type=doc_type,
        classification=classification,
        department_id=department_id,
        status=status_filter,
        tag=tag,
        owner=owner,
        cursor=cursor,
        limit=limit,
    )


# --------------------------------------------------------------------------- #
# Single document
# --------------------------------------------------------------------------- #
@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID, principal: Principal = _READ, container: Container = _CONTAINER
) -> DocumentDetail:
    return await container.documents.get(principal, document_id)


@router.patch("/{document_id}", response_model=DocumentDetail)
async def update_document(
    document_id: uuid.UUID,
    patch: DocumentUpdate,
    principal: Principal = _READ,
    container: Container = _CONTAINER,
) -> DocumentDetail:
    return await container.documents.update(principal, document_id, patch)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, principal: Principal = _READ, container: Container = _CONTAINER
) -> Response:
    await container.documents.delete(principal, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{document_id}/versions",
    status_code=status.HTTP_201_CREATED,
    response_model=UploadResult,
    openapi_extra=_VERSION_OPENAPI,
)
async def add_document_version(
    document_id: uuid.UUID,
    request: Request,
    principal: Principal = _UPLOAD,
    container: Container = _CONTAINER,
) -> UploadResult:
    form = await request.form(max_files=1, max_fields=_MAX_FORM_FIELDS)
    try:
        upload, data = _form_fields(form, frozenset())
        fields = _validated(VersionForm, data)
        return await container.documents.add_version(
            principal,
            document_id,
            stream=_file_chunks(upload),
            filename=upload.filename or "",
            declared_mime=upload.content_type,
            change_note=fields.change_note,
            allow_duplicate=fields.allow_duplicate,
        )
    finally:
        await form.close()


@router.get("/{document_id}/download")
async def download_current_version(
    document_id: uuid.UUID,
    acknowledge_risk: bool = Query(False),
    principal: Principal = _READ,
    container: Container = _CONTAINER,
) -> StreamingResponse:
    payload = await container.documents.download(
        principal, document_id, acknowledge_risk=acknowledge_risk
    )
    return _download_response(payload)


@router.get("/{document_id}/versions/{version_number}/download")
async def download_version(
    document_id: uuid.UUID,
    version_number: int,
    acknowledge_risk: bool = Query(False),
    principal: Principal = _READ,
    container: Container = _CONTAINER,
) -> StreamingResponse:
    if version_number < 1:
        raise ValidationFailed("version_number must be positive.")
    payload = await container.documents.download(
        principal, document_id, version_number=version_number, acknowledge_risk=acknowledge_risk
    )
    return _download_response(payload)


@router.get("/{document_id}/content", response_model=ContentPage)
async def document_content(
    *,
    document_id: uuid.UUID,
    version: int | None = Query(None, ge=1),
    page: int | None = Query(None, ge=1),
    cursor: str | None = Query(None, max_length=256),
    limit: int = Query(20, ge=1, le=100),
    principal: Principal = _READ,
    container: Container = _CONTAINER,
) -> ContentPage:
    return await container.documents.content(
        principal, document_id, version_number=version, page=page, cursor=cursor, limit=limit
    )


# --------------------------------------------------------------------------- #
# Grants
# --------------------------------------------------------------------------- #
@router.get("/{document_id}/grants", response_model=list[GrantInfo])
async def list_document_grants(
    document_id: uuid.UUID, principal: Principal = _READ, container: Container = _CONTAINER
) -> list[GrantInfo]:
    return await container.documents.list_grants(principal, document_id)


@router.post("/{document_id}/grants", status_code=status.HTTP_201_CREATED, response_model=GrantInfo)
async def add_document_grant(
    document_id: uuid.UUID,
    grant: GrantCreate,
    principal: Principal = _READ,
    container: Container = _CONTAINER,
) -> GrantInfo:
    return await container.documents.add_grant(principal, document_id, grant)


@router.delete("/{document_id}/grants/{grant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_document_grant(
    document_id: uuid.UUID,
    grant_id: uuid.UUID,
    principal: Principal = _READ,
    container: Container = _CONTAINER,
) -> Response:
    await container.documents.revoke_grant(principal, document_id, grant_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
