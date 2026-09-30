"""Load the RAG evaluation golden corpus for ``docassist evaluate``.

The golden dataset (``docassist.rag.evaluation.GOLDEN_DATASET``) describes users, documents
(page by page) and questions for two organisations. To evaluate a real deployment the corpus
is loaded into two *dedicated* organisations, ``eval-<org>``, so that no other document
can influence retrieval or the refusal metrics:

* one account per golden persona - with an unusable password (the harness calls the answer
  service directly; nobody signs in as a persona);
* every document uploaded as a PDF with one page per golden page (so page numbers in
  citations match the dataset) through the real upload path, owned by its golden owner;
* the ingestion jobs processed by the real worker.

Loading is idempotent (organisations by slug, accounts by email, documents by title).
Dates in the texts are rendered relative to the day of the *first* load.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Any

from docassist.authz.principal import Principal
from docassist.core.enums import Classification, DocumentType, Role
from docassist.demo.content import DemoDocument, DemoOrg, DemoUser
from docassist.demo.pdf import PAGE_BREAK, build_pdf
from docassist.demo.seed import (
    DemoComponentMissing,
    SeedReport,
    ensure_org,
    ensure_user,
    process_pending_jobs,
    seed_documents,
)
from docassist.identity.schemas import slugify

if TYPE_CHECKING:
    from docassist.api.container import Container

EVAL_ORG_PREFIX = "eval-"


def eval_org_slug(org_key: str) -> str:
    return f"{EVAL_ORG_PREFIX}{slugify(org_key)}"


def persona_email(user_key: str, org_key: str) -> str:
    return f"{slugify(user_key)}@{eval_org_slug(org_key)}.example"


def golden_documents(
    dataset: Any, today: date, render: Callable[[str, date], str]
) -> list[DemoDocument]:
    """Turn golden documents into uploadable PDFs (one PDF page per golden page)."""
    created = datetime.combine(today, time(9, 0), tzinfo=UTC)
    out: list[DemoDocument] = []
    for doc in dataset.documents:
        lines: list[str] = []
        for index, page in enumerate(doc.pages):
            if index:
                lines.append(PAGE_BREAK)
            if index == 0:
                lines += [doc.title, ""]
            lines += [page.section, "", render(page.text, today)]
        out.append(
            DemoDocument(
                key=doc.key,
                org=doc.org_key,
                title=doc.title,
                filename=f"{slugify(doc.key)}.pdf",
                classification=Classification(doc.classification),
                department=slugify(doc.department_key) if doc.department_key else None,
                doc_type=DocumentType(doc.doc_type),
                uploader=doc.owner_key,
                data=build_pdf(lines, title=doc.title, created=created),
            )
        )
    return out


async def load_golden_corpus(
    container: Container,
    dataset: Any,
    *,
    today: date,
    render: Callable[[str, date], str],
    process: bool = True,
    max_jobs: int = 500,
) -> tuple[dict[str, Principal], SeedReport]:
    """Load (or complete) the evaluation organisations; returns ``persona key -> Principal``.

    A missing documents service or worker is recorded in ``report.unavailable``.
    """
    report = SeedReport()
    org_keys = sorted({u.org_key for u in dataset.users} | {d.org_key for d in dataset.documents})
    tenants = {}
    for key in org_keys:
        names = {
            *(d for u in dataset.users if u.org_key == key for d in (*u.departments, *u.managed)),
            *(d.department_key for d in dataset.documents if d.org_key == key and d.department_key),
        }
        spec = DemoOrg(
            slug=eval_org_slug(key),
            name=f"Evaluation - {key}",
            departments=tuple((slugify(n), n.replace("_", " ").title()) for n in sorted(names)),
        )
        tenants[key] = await ensure_org(container, spec, report)
    principals: dict[str, Principal] = {}
    for user in dataset.users:
        persona = DemoUser(
            key=user.key,
            org=eval_org_slug(user.org_key),
            email=persona_email(user.key, user.org_key),
            full_name=f"Evaluation persona {user.key}",
            role=Role(user.role),
            departments=tuple(
                slugify(d) for d in dict.fromkeys((*user.departments, *user.managed))
            ),
            managed=tuple(slugify(d) for d in user.managed),
        )
        principals[user.key] = await ensure_user(
            container,
            persona,
            tenants[user.org_key],
            report,
            reset_password=False,
            clearance=Classification(user.clearance) if user.clearance else None,
            sign_in=False,
        )
    try:
        await seed_documents(
            container, golden_documents(dataset, today, render), principals, tenants, report
        )
        if process:
            report.jobs_processed = await process_pending_jobs(
                container, max_jobs=max_jobs, report=report
            )
    except DemoComponentMissing as exc:
        report.unavailable = exc.component
    return principals, report
