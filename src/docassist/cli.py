"""``docassist`` - operator command line.

Security conventions for every command:

* **Secrets never travel through argv** (visible in ``ps`` and shell history): passwords are
  read with ``getpass`` (asked twice) or from a file.
* **Secrets are never printed**: ``init-env`` prints only the path it wrote, ``seed-demo``
  writes demo passwords to an owner-only (0600) file.
* **No stack traces by default**: errors go to stderr as one line; ``--debug`` shows the
  traceback.

Exit codes::

    0  success                       4  target already exists (file, slug, email)
    1  unexpected error              5  required component or dependency unavailable
    2  invalid command-line usage    6  verification failed (audit chain, evaluation)
    3  invalid configuration         7  invalid input (validation, password policy)
                                   130  interrupted
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import getpass
import importlib
import json
import os
import secrets
import sys
import tempfile
import traceback
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from docassist import __version__

if TYPE_CHECKING:
    from docassist.api.container import Container
    from docassist.core.config import Settings
    from docassist.demo.seed import Credential

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_CONFLICT = 4
EXIT_UNAVAILABLE = 5
EXIT_VERIFY_FAILED = 6
EXIT_INPUT = 7
EXIT_INTERRUPTED = 130

DEFAULT_DEMO_CREDENTIALS = Path("var") / "demo-credentials.txt"

ROTATE_JWT_PROCEDURE = """\
Rotating the JWT signing key (no downtime, no forced logout):

  1. Generate a new key:   python -c "import secrets; print(secrets.token_urlsafe(48))"
  2. Append the CURRENT key to DOCASSIST_SECURITY__JWT_PREVIOUS_SIGNING_KEYS (a JSON list,
     e.g. '["<current key>"]'); previous keys only verify, they never sign.
  3. Set DOCASSIST_SECURITY__JWT_SIGNING_KEY to the new key (via your secret store or a
     *_FILE secret) and restart every API replica (rolling restart).
  4. Wait at least security.access_token_ttl_seconds (default 10 minutes): every access
     token signed with the old key has then expired. Refresh tokens are opaque, not JWTs,
     so they keep working; waiting a full refresh-token lifetime (docs/operations.md) is a
     conservative margin.
  5. Remove the old key from DOCASSIST_SECURITY__JWT_PREVIOUS_SIGNING_KEYS and restart again.

Suspected key compromise: skip step 2 (every access token dies at once) and consider
revoking sessions - per user with POST /api/v1/users/{id}/revoke-sessions, per tenant by
suspending and reactivating the organisation (PATCH /api/v1/platform/organizations/{id}).
"""


class CliError(Exception):
    """A failure with a user-facing one-line message and an exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# --------------------------------------------------------------------------- #
# Hooks (tests replace these to inject settings/overrides)
# --------------------------------------------------------------------------- #
def _load_settings() -> Settings:
    from pydantic import ValidationError

    from docassist.core.config import load_settings

    try:
        return load_settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors()[:10]
        )
        raise CliError(f"invalid configuration: {problems}", EXIT_CONFIG) from None
    except OSError as exc:
        raise CliError(f"cannot read a *_FILE secret: {exc.strerror}", EXIT_CONFIG) from None


def _build_container(settings: Settings, role: str) -> Container:
    from docassist.api.container import build_container

    return build_container(settings, role=role)


# --------------------------------------------------------------------------- #
# Private files
# --------------------------------------------------------------------------- #
def write_private_file(path: Path, content: str, *, overwrite: bool) -> None:
    """Write ``content`` with owner-only permissions (0600 on POSIX).

    Without ``overwrite`` the file is created exclusively (``O_CREAT | O_EXCL``, which also
    refuses to follow a planted symlink) and ``FileExistsError`` is raised if it exists.
    With ``overwrite`` the data goes to a fresh ``mkstemp`` file (created 0600) in the same
    directory that then atomically replaces the target, so the secret is never readable by
    others - not even when it replaces a looser pre-existing file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    if not overwrite:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def read_secret_file(path: Path) -> str:
    """First line of a secret file, without the trailing newline."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc.strerror}", EXIT_INPUT) from None
    if os.name == "posix" and path.stat().st_mode & 0o077:
        print(f"warning: {path} is readable by other users", file=sys.stderr)
    line = text.splitlines()[0] if text else ""
    if not line:
        raise CliError(f"{path} is empty", EXIT_INPUT)
    return line


# --------------------------------------------------------------------------- #
# init-env
# --------------------------------------------------------------------------- #
def render_env_file(now: datetime | None = None) -> str:
    """A development ``.env`` with fresh random secrets (all distinct, all >= 32 bytes)."""
    now = now or datetime.now(UTC)
    kid = f"k{now:%Y%m%d}"
    key = base64.b64encode(secrets.token_bytes(32)).decode()

    def secret() -> str:
        return secrets.token_urlsafe(48)

    def db_password() -> str:
        return secrets.token_urlsafe(24)

    lines = [
        f"# Development configuration generated by `docassist init-env` on {now:%Y-%m-%d}.",
        "# Every secret below is random and unique to this file. Never commit it.",
        "# The database roles must exist with these passwords (set them when provisioning).",
        "DOCASSIST_ENVIRONMENT=development",
        "DOCASSIST_PUBLIC_BASE_URL=http://localhost:8000",
        "",
        (
            "DOCASSIST_DATABASE__URL=postgresql+asyncpg://docassist_app:"
            f"{db_password()}@localhost:5432/docassist"
        ),
        (
            "DOCASSIST_DATABASE__WORKER_URL=postgresql+asyncpg://docassist_worker:"
            f"{db_password()}@localhost:5432/docassist"
        ),
        (
            "DOCASSIST_DATABASE__MIGRATION_URL=postgresql+asyncpg://docassist_owner:"
            f"{db_password()}@localhost:5432/docassist"
        ),
        "DOCASSIST_DATABASE__SSL=disable",
        "DOCASSIST_REDIS__URL=redis://localhost:6379/0",
        "",
        f"DOCASSIST_SECURITY__JWT_SIGNING_KEY={secret()}",
        f"DOCASSIST_SECURITY__TOKEN_PEPPER={secret()}",
        f"DOCASSIST_SECURITY__AUDIT_HMAC_KEY={secret()}",
        f"DOCASSIST_SECURITY__ENCRYPTION_KEYS={kid}:{key}",
        f"DOCASSIST_SECURITY__ACTIVE_ENCRYPTION_KEY_ID={kid}",
        "# Plain-HTTP local development only; production requires secure cookies.",
        "DOCASSIST_SECURITY__COOKIE_SECURE=false",
        "",
        "DOCASSIST_OBSERVABILITY__LOG_FORMAT=console",
    ]
    return "\n".join(lines) + "\n"


def cmd_init_env(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        write_private_file(path, render_env_file(), overwrite=args.force)
    except FileExistsError:
        raise CliError(f"{path} already exists (use --force to overwrite)", EXIT_CONFLICT) from None
    print(path)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# migrate / serve / worker
# --------------------------------------------------------------------------- #
def _find_alembic_ini(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    candidates += [
        Path.cwd() / "alembic.ini",
        Path(__file__).resolve().parents[2] / "alembic.ini",  # source checkout
        Path(__file__).resolve().parent / "_alembic" / "alembic.ini",  # installed wheel
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise CliError(
        "alembic.ini not found: run from the project directory or pass --config", EXIT_CONFIG
    )


def cmd_migrate(args: argparse.Namespace) -> int:
    settings = _load_settings()
    url = settings.database.migration_url
    if url is None:
        raise CliError(
            "DOCASSIST_DATABASE__MIGRATION_URL (schema owner DSN) is not set", EXIT_CONFIG
        )
    from alembic import command
    from alembic.config import Config

    config = Config(str(_find_alembic_ini(args.config)))
    # The DSN stays in memory (never argv); env.py reads it from the "-x dburl" options.
    config.cmd_opts = argparse.Namespace(x=[f"dburl={url.get_secret_value()}"])
    # The migration reads these from the environment; keep them consistent with settings.
    os.environ["DOCASSIST_DATABASE__APP_ROLE"] = settings.database.app_role
    os.environ["DOCASSIST_DATABASE__WORKER_ROLE"] = settings.database.worker_role
    os.environ["DOCASSIST_EMBEDDING__DIMENSIONS"] = str(settings.embedding.dimensions)
    command.upgrade(config, args.revision)
    print(f"database upgraded to {args.revision}")
    return EXIT_OK


def uvicorn_options(settings: Settings, host: str, port: int, workers: int) -> dict[str, Any]:
    """Proxy headers are honoured only from configured trusted proxies; no Server header."""
    trusted = settings.security.trusted_proxies
    return {
        "host": host,
        "port": port,
        "workers": workers,
        "factory": True,
        "proxy_headers": bool(trusted),
        "forwarded_allow_ips": ",".join(trusted) if trusted else None,
        "server_header": False,
        "date_header": True,
        "access_log": False,  # the application writes its own structured access log
        "log_config": None,
    }


def cmd_serve(args: argparse.Namespace) -> int:
    settings = _load_settings()
    import uvicorn

    uvicorn.run(
        "docassist.api.app:create_app",
        **uvicorn_options(settings, args.host, args.port, args.workers),
    )
    return EXIT_OK


def _import_optional(module: str, what: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and module.startswith(exc.name):
            raise CliError(f"the {what} is not available in this build", EXIT_UNAVAILABLE) from None
        raise


def cmd_worker(args: argparse.Namespace) -> int:
    settings = _load_settings()
    worker_module = _import_optional("docassist.jobs.worker", "background worker")

    async def run() -> None:
        container = _build_container(settings, "worker")
        try:
            worker = worker_module.Worker(
                container,
                concurrency=args.concurrency or settings.worker.concurrency,
                poll_interval=settings.worker.poll_interval_seconds,
            )
            await worker.run()
        finally:
            await container.close()

    asyncio.run(run())
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Accounts and organisations
# --------------------------------------------------------------------------- #
def _with_container[T](role: str, body: Callable[[Container], Coroutine[Any, Any, T]]) -> T:
    settings = _load_settings()

    async def run() -> T:
        container = _build_container(settings, role)
        try:
            return await body(container)
        finally:
            await container.close()

    return asyncio.run(run())


def _input_error(exc: Exception) -> NoReturn:
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise CliError(f"invalid input: {problems}", EXIT_INPUT) from None
    raise CliError(f"invalid input: {exc}", EXIT_INPUT) from None


def _ask_password() -> str:
    first = getpass.getpass("New password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        raise CliError("the passwords do not match", EXIT_INPUT)
    return first


def cmd_create_platform_admin(args: argparse.Namespace) -> int:
    from docassist.identity.schemas import clean_name, normalize_email

    try:
        email = normalize_email(args.email)
        name = clean_name(args.name)
    except ValueError as exc:
        _input_error(exc)
    password = read_secret_file(Path(args.password_file)) if args.password_file else _ask_password()

    async def body(container: Container) -> Any:
        return await container.admin.create_platform_admin(
            email=email, full_name=name, password=password
        )

    user_id = _with_container("api", body)
    print(f"platform admin created: {user_id}")
    return EXIT_OK


def cmd_create_org(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from docassist.audit.service import Actor
    from docassist.identity.schemas import OrganizationCreate

    try:
        data = OrganizationCreate(
            slug=args.slug, name=args.name, admin_email=args.admin_email, admin_name=args.admin_name
        )
    except ValidationError as exc:
        _input_error(exc)

    async def body(container: Container) -> Any:
        return await container.admin.provision_organization(data, actor=Actor.system(None))

    created = _with_container("api", body)
    print(f"organization created: {created.organization.id} ({created.organization.slug})")
    print(f"first administrator: {created.admin_user_id} <{data.admin_email}>")
    if created.invitation_sent:
        print("an invitation email with a password link was sent")
    else:
        print(
            "warning: the invitation email could not be sent; use 'Forgot password'",
            file=sys.stderr,
        )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #
def _merge_credentials(path: Path, credentials: list[Credential]) -> str:
    """Rewrite the credentials file: new passwords replace older lines for the same email."""
    kept: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "\t" in line:
                kept[line.split("\t", 1)[0]] = line
    for cred in credentials:
        kept[cred.email] = "\t".join((cred.email, cred.role, cred.organization, cred.password))
    header = [
        "# AI Document Assistant demo accounts. Keep this file private; delete it when done.",
        f"# Updated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Columns: email, role, org, password",
    ]
    return "\n".join([*header, *sorted(kept.values())]) + "\n"


def cmd_seed_demo(args: argparse.Namespace) -> int:
    from docassist.demo.seed import SeedReport, seed_demo

    path = Path(args.password_file) if args.password_file else DEFAULT_DEMO_CREDENTIALS
    report = SeedReport()

    async def body(container: Container) -> None:
        await seed_demo(
            container,
            report=report,
            reset_passwords=args.reset_passwords,
            documents=not args.no_documents,
            process=not args.no_process,
            max_jobs=args.max_jobs,
        )

    try:
        _with_container("worker", body)
    finally:
        # Accounts created before a later failure still get their passwords recorded.
        if report.credentials:
            write_private_file(path, _merge_credentials(path, report.credentials), overwrite=True)
            print(f"{len(report.credentials)} demo password(s) written to {path} (mode 0600)")
    if not report.credentials:
        print("existing demo accounts kept their passwords (use --reset-passwords for new ones)")
    print(
        f"organizations created: {len(report.organizations_created)}, "
        f"users created: {report.users_created}, "
        f"documents uploaded: {len(report.documents_uploaded)}, "
        f"already present: {len(report.documents_existing)}"
    )
    if report.jobs_processed is not None:
        print(f"background jobs processed: {report.jobs_processed} ({report.jobs_dead} failed)")
    for key, code in sorted(report.documents_rejected.items()):
        print(f"document rejected: {key} ({code})", file=sys.stderr)
    if report.unavailable:
        raise CliError(
            f"the {report.unavailable} is unavailable: documents were not (fully) processed; "
            "start `docassist worker` or re-run seed-demo later",
            EXIT_UNAVAILABLE,
        )
    return EXIT_ERROR if report.documents_rejected else EXIT_OK


# --------------------------------------------------------------------------- #
# Audit verification & evaluation
# --------------------------------------------------------------------------- #
def cmd_audit_verify(args: argparse.Namespace) -> int:
    async def body(container: Container) -> list[tuple[str, Any]]:
        admin = container.admin
        if args.org:
            org_id = await admin.find_organization_id(args.org)
            if org_id is None:
                raise CliError(f"unknown organization: {args.org}", EXIT_INPUT)
            return [(args.org, await admin.verify_chain_as_operator(org_id))]
        results = [("platform", await admin.verify_chain_as_operator(None))]
        for org_id, slug in await admin.list_organization_ids():
            results.append((slug, await admin.verify_chain_as_operator(org_id)))
        return results

    results = _with_container("api", body)
    failed = False
    for name, report in results:
        if report.valid:
            print(
                f"{name}: valid ({report.checked} sealed events, head {report.head_seq}, "
                f"{report.unsealed} not yet sealed)"
            )
        else:
            failed = True
            print(
                f"{name}: INVALID at seq {report.first_bad_seq}: {report.reason}", file=sys.stderr
            )
    return EXIT_VERIFY_FAILED if failed else EXIT_OK


def _jsonable(report: Any) -> Any:
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    dump = getattr(report, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if dataclasses.is_dataclass(report) and not isinstance(report, type):
        return dataclasses.asdict(report)
    return {"report": str(report)}


def cmd_evaluate(args: argparse.Namespace) -> int:
    evaluation = _import_optional("docassist.rag.evaluation", "evaluation harness")
    from docassist.demo.golden import load_golden_corpus

    async def body(container: Container) -> Any:
        principals, loaded = await load_golden_corpus(
            container,
            evaluation.GOLDEN_DATASET,
            today=datetime.now(UTC).date(),
            render=evaluation.render_text,
            process=not args.no_process,
        )
        for key, code in sorted(loaded.documents_rejected.items()):
            print(f"warning: golden document rejected: {key} ({code})", file=sys.stderr)
        if loaded.unavailable:
            raise CliError(
                f"the {loaded.unavailable} is unavailable; the evaluation corpus is not ready",
                EXIT_UNAVAILABLE,
            )
        return await evaluation.run_evaluation(container, principals)

    report = _with_container("worker", body)
    data = _jsonable(report)
    if args.output:
        Path(args.output).write_text(
            json.dumps(data, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"full report written to {args.output}")
    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
    for name, value in sorted(metrics.items()):
        print(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
    failures = data.get("failures", []) if isinstance(data, dict) else []
    for failure in failures:
        print(f"below threshold: {failure}", file=sys.stderr)
    if getattr(report, "passed", True) is False:
        return EXIT_VERIFY_FAILED
    print("evaluation passed")
    return EXIT_OK


def cmd_rotate_jwt(_args: argparse.Namespace) -> int:
    print(ROTATE_JWT_PROCEDURE, end="")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Parser & entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: "--password x" must never be read as "--password-file x".
    parser = argparse.ArgumentParser(
        prog="docassist", description="AI Document Assistant operator commands.", allow_abbrev=False
    )
    parser.add_argument("--version", action="version", version=f"docassist {__version__}")
    parser.add_argument("--debug", action="store_true", help="show tracebacks on errors")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser(
        "init-env", allow_abbrev=False, help="write a development .env with fresh random secrets"
    )
    p.add_argument("--path", default=".env", help="file to write (default: .env)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(handler=cmd_init_env)

    p = sub.add_parser(
        "migrate", allow_abbrev=False, help="apply database migrations (schema owner DSN)"
    )
    p.add_argument("--revision", default="head", help="target revision (default: head)")
    p.add_argument("--config", help="path to alembic.ini")
    p.set_defaults(handler=cmd_migrate)

    p = sub.add_parser("serve", allow_abbrev=False, help="run the API server (uvicorn)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(handler=cmd_serve)

    p = sub.add_parser("worker", allow_abbrev=False, help="run the background job worker")
    p.add_argument("--concurrency", type=int, help="override worker.concurrency")
    p.set_defaults(handler=cmd_worker)

    p = sub.add_parser(
        "create-platform-admin", allow_abbrev=False, help="create a platform operator account"
    )
    p.add_argument("--email", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--password-file", help="read the password from this file (else prompt)")
    p.set_defaults(handler=cmd_create_platform_admin)

    p = sub.add_parser(
        "create-org", allow_abbrev=False, help="create an organisation and invite its first admin"
    )
    p.add_argument("--slug", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--admin-email", required=True)
    p.add_argument("--admin-name", required=True)
    p.set_defaults(handler=cmd_create_org)

    p = sub.add_parser(
        "seed-demo", allow_abbrev=False, help="create the demo tenants, accounts and documents"
    )
    p.add_argument(
        "--password-file",
        help=f"where demo passwords are written, mode 0600 (default: {DEFAULT_DEMO_CREDENTIALS})",
    )
    p.add_argument("--reset-passwords", action="store_true", help="issue new demo passwords")
    p.add_argument("--no-documents", action="store_true", help="accounts only, no documents")
    p.add_argument("--no-process", action="store_true", help="upload without running the worker")
    p.add_argument("--max-jobs", type=int, default=500, help="worker drain limit (default: 500)")
    p.set_defaults(handler=cmd_seed_demo)

    p = sub.add_parser("audit-verify", allow_abbrev=False, help="verify the audit hash chains")
    p.add_argument("--org", help="organisation slug (default: platform chain + every org)")
    p.set_defaults(handler=cmd_audit_verify)

    p = sub.add_parser(
        "evaluate",
        allow_abbrev=False,
        help="load the golden corpus into eval-* organisations and run the RAG evaluation",
    )
    p.add_argument("--output", help="write the full JSON report to this file")
    p.add_argument("--no-process", action="store_true", help="do not run the worker first")
    p.set_defaults(handler=cmd_evaluate)

    p = sub.add_parser(
        "rotate-jwt", allow_abbrev=False, help="print the JWT signing-key rotation procedure"
    )
    p.set_defaults(handler=cmd_rotate_jwt)
    return parser


def _configure_logging() -> None:
    from docassist.core.logging import configure_logging

    level = os.environ.get("DOCASSIST_OBSERVABILITY__LOG_LEVEL", "WARNING").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        level = "WARNING"
    configure_logging(level, "console")


def main(argv: Sequence[str] | None = None, *, configure_logs: bool = True) -> int:
    """Entry point. Returns the process exit code (see the module docstring)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help/--version (0) or usage error (2)
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    if configure_logs:
        _configure_logging()
    handler: Callable[[argparse.Namespace], int] = args.handler
    from docassist.core.errors import AppError, Conflict, ValidationFailed

    try:
        return handler(args)
    except CliError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return exc.code
    except (Conflict, ValidationFailed) as exc:
        print(f"error: {exc.public_message}", file=sys.stderr)
        return EXIT_CONFLICT if isinstance(exc, Conflict) else EXIT_INPUT
    except AppError as exc:
        if args.debug:
            traceback.print_exc()
        print(f"error: {exc.public_message}", file=sys.stderr)
        return EXIT_UNAVAILABLE if exc.status_code >= 500 else EXIT_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 - CLI boundary: one line on stderr, not a traceback
        if args.debug:
            traceback.print_exc()
        else:
            print(
                f"error: unexpected {type(exc).__name__} (re-run with --debug for details)",
                file=sys.stderr,
            )
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
