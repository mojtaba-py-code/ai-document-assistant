"""The ``docassist`` CLI, called in-process through ``main(argv)``.

Tests without the ``db`` marker need no database; the others use the session test database
through the real CLI code paths (settings and container construction are the only hooks).
"""

from __future__ import annotations

import asyncio
import os
import stat
import uuid
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from docassist import cli
from docassist.core.config import Settings
from docassist.security.passwords import PasswordService
from tests.conftest import make_settings

POSIX = os.name == "posix"


def run(argv: list[str]) -> int:
    return cli.main(argv, configure_logs=False)


def _query(url: str, sql: str, *args: Any) -> list[asyncpg.Record]:
    async def go() -> list[asyncpg.Record]:
        conn = await asyncpg.connect(url)
        try:
            return list(await conn.fetch(sql, *args))
        finally:
            await conn.close()

    return asyncio.run(go())


def _execute_as_superuser(url: str, *statements: tuple[str, tuple[Any, ...]]) -> None:
    async def go() -> None:
        conn = await asyncpg.connect(url)
        try:
            async with conn.transaction():
                await conn.execute("SET LOCAL session_replication_role = replica")
                for sql, args in statements:
                    await conn.execute(sql, *args)
        finally:
            await conn.close()

    asyncio.run(go())


@pytest.fixture
def cli_env(settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Point the CLI at the test database; capture emails in memory."""
    import fakeredis

    from docassist.api.container import build_container
    from docassist.identity.notifications import MemoryEmailSender

    email = MemoryEmailSender()

    def build(s: Settings, role: str) -> Any:
        return build_container(
            s, role=role, overrides={"redis": fakeredis.FakeAsyncRedis(), "email": email}
        )

    monkeypatch.setattr(cli, "_load_settings", lambda: settings)
    monkeypatch.setattr(cli, "_build_container", build)
    monkeypatch.chdir(tmp_path)
    return email


# --------------------------------------------------------------------------- #
# init-env
# --------------------------------------------------------------------------- #
def test_init_env_writes_a_private_file_that_loads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "conf" / ".env"
    assert run(["init-env", "--path", str(target)]) == 0
    out = capsys.readouterr()
    assert out.out == f"{target}\n" and out.err == ""  # only the path is printed
    if POSIX:
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    loaded = Settings(_env_file=target)  # type: ignore[call-arg]
    sec = loaded.security
    values = [
        sec.jwt_signing_key.get_secret_value(),
        sec.token_pepper.get_secret_value(),
        sec.audit_hmac_key.get_secret_value(),
    ]
    assert len(set(values)) == 3 and all(len(v) >= 32 for v in values)
    assert loaded.environment.value == "development"
    assert loaded.database.migration_url is not None and loaded.database.worker_url is not None
    assert sec.active_encryption_key_id in sec.encryption_keys.get_secret_value()
    for value in values:
        assert value not in out.out


def test_init_env_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / ".env"
    target.write_text("KEEP=1\n", encoding="utf-8")
    if POSIX:
        target.chmod(0o644)
    assert run(["init-env", "--path", str(target)]) == cli.EXIT_CONFLICT
    assert "already exists" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "KEEP=1\n"
    assert run(["init-env", "--path", str(target), "--force"]) == 0
    first = target.read_text(encoding="utf-8")
    assert "KEEP=1" not in first and "DOCASSIST_SECURITY__JWT_SIGNING_KEY=" in first
    if POSIX:
        assert stat.S_IMODE(target.stat().st_mode) == 0o600  # tightened, not inherited
    assert run(["init-env", "--path", str(target), "--force"]) == 0
    assert target.read_text(encoding="utf-8") != first  # fresh secrets every time
    assert [p.name for p in tmp_path.iterdir()] == [".env"]  # no temp files left behind


@pytest.mark.skipif(not POSIX, reason="symlinks need privileges on Windows")
def test_private_file_does_not_follow_a_planted_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("original", encoding="utf-8")
    link = tmp_path / ".env"
    link.symlink_to(victim)
    with pytest.raises(FileExistsError):
        cli.write_private_file(link, "SECRET=1\n", overwrite=False)
    assert victim.read_text(encoding="utf-8") == "original"


# --------------------------------------------------------------------------- #
# Parser, errors, help-only commands
# --------------------------------------------------------------------------- #
def test_usage_errors_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert run([]) == cli.EXIT_USAGE
    assert run(["no-such-command"]) == cli.EXIT_USAGE
    assert run(["create-org", "--slug", "x"]) == cli.EXIT_USAGE  # missing required options
    assert run(["--version"]) == 0
    assert "docassist" in capsys.readouterr().out


def test_passwords_cannot_be_passed_in_argv() -> None:
    assert (
        run(["create-platform-admin", "--email", "a@example.com", "--name", "A", "--password", "x"])
        == cli.EXIT_USAGE
    )


def test_rotate_jwt_prints_the_procedure(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["rotate-jwt"]) == 0
    text = capsys.readouterr().out
    assert "JWT_PREVIOUS_SIGNING_KEYS" in text and "access_token_ttl_seconds" in text


def test_invalid_configuration_is_reported_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    for key in list(os.environ):
        if key.upper().startswith("DOCASSIST_") and key.upper() != "DOCASSIST_TEST_DATABASE_URL":
            monkeypatch.delenv(key)
    weak = "short-jwt-secret-value"
    monkeypatch.setenv("DOCASSIST_SECURITY__JWT_SIGNING_KEY", weak)
    assert run(["audit-verify"]) == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "invalid configuration" in err and "database" in err
    assert weak not in err


def test_unexpected_errors_are_one_line_unless_debug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom() -> Settings:
        raise RuntimeError("internal detail /srv/secret/path")

    monkeypatch.setattr(cli, "_load_settings", boom)
    assert run(["audit-verify"]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert err.strip() == "error: unexpected RuntimeError (re-run with --debug for details)"
    assert run(["--debug", "audit-verify"]) == cli.EXIT_ERROR
    assert "Traceback" in capsys.readouterr().err


def test_serve_uses_hardened_uvicorn_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))
    monkeypatch.setattr(cli, "_load_settings", make_settings)
    assert run(["serve", "--port", "8123"]) == 0
    options = calls[-1]
    assert options["app"] == "docassist.api.app:create_app" and options["factory"] is True
    assert options["host"] == "127.0.0.1" and options["port"] == 8123
    assert options["proxy_headers"] is False and options["forwarded_allow_ips"] is None
    assert options["server_header"] is False

    proxied = make_settings(security={"trusted_proxies": ["10.0.0.0/8", "192.168.1.1/32"]})
    monkeypatch.setattr(cli, "_load_settings", lambda: proxied)
    assert run(["serve"]) == 0
    assert calls[-1]["proxy_headers"] is True
    assert calls[-1]["forwarded_allow_ips"] == "10.0.0.0/8,192.168.1.1/32"


def test_migrate_requires_the_schema_owner_dsn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_load_settings", make_settings)
    assert run(["migrate"]) == cli.EXIT_CONFIG
    assert "MIGRATION_URL" in capsys.readouterr().err


def test_optional_components_fail_with_a_clear_message() -> None:
    with pytest.raises(cli.CliError) as info:
        cli._import_optional("docassist.no_such_component", "widget")
    assert info.value.code == cli.EXIT_UNAVAILABLE
    assert "widget is not available" in info.value.message


def test_create_platform_admin_validates_input_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run(["create-platform-admin", "--email", "not-an-email", "--name", "A"]) == cli.EXIT_INPUT
    )
    empty = tmp_path / "pw"
    empty.write_text("", encoding="utf-8")
    assert (
        run(
            [
                "create-platform-admin",
                "--email",
                "a@example.com",
                "--name",
                "A",
                "--password-file",
                str(empty),
            ]
        )
        == cli.EXIT_INPUT
    )
    assert "empty" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Database-backed commands
# --------------------------------------------------------------------------- #
@pytest.mark.db
def test_create_org_flow(
    cli_env: Any, test_database: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    slug = f"cli-{uuid.uuid4().hex[:8]}"
    admin_email = f"founder-{slug}@example.test"
    argv = [
        "create-org",
        "--slug",
        slug,
        "--name",
        "CLI Co",
        "--admin-email",
        admin_email,
        "--admin-name",
        "Founder",
    ]
    assert run(argv) == 0
    out = capsys.readouterr().out
    assert slug in out and "invitation email" in out
    assert [m[0] for m in cli_env.sent] == [admin_email]
    assert "token=" in cli_env.sent[0][2]

    orgs = _query(
        test_database.admin_url, "SELECT id, name, status FROM organizations WHERE slug = $1", slug
    )
    assert len(orgs) == 1 and orgs[0]["status"] == "active"
    users = _query(
        test_database.admin_url,
        "SELECT role, clearance, organization_id FROM users WHERE email = $1",
        admin_email,
    )
    assert [tuple(u) for u in users] == [("organization_admin", "RESTRICTED", orgs[0]["id"])]
    tenant_events = _query(
        test_database.admin_url,
        "SELECT action, actor_role FROM audit_events WHERE organization_id = $1",
        orgs[0]["id"],
    )
    assert ("admin.user_created", "system") in [tuple(e) for e in tenant_events]

    assert run(argv) == cli.EXIT_CONFLICT  # same slug again
    assert "already exists" in capsys.readouterr().err
    bad = [
        "create-org",
        "--slug",
        "Bad Slug",
        "--name",
        "X",
        "--admin-email",
        "x@example.test",
        "--admin-name",
        "X",
    ]
    assert run(bad) == cli.EXIT_INPUT


@pytest.mark.db
def test_create_platform_admin(
    cli_env: Any, test_database: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    email = f"operator-{uuid.uuid4().hex[:8]}@example.test"
    secret_file = tmp_path / "admin-password"
    password = f"Platform-Ops-{uuid.uuid4().hex}"
    secret_file.write_text(password + "\n", encoding="utf-8")
    argv = [
        "create-platform-admin",
        "--email",
        email,
        "--name",
        "Op Erator",
        "--password-file",
        str(secret_file),
    ]
    assert run(argv) == 0
    rows = _query(
        test_database.admin_url,
        "SELECT role, organization_id, password_hash FROM users WHERE email = $1",
        email,
    )
    assert rows[0]["role"] == "platform_admin" and rows[0]["organization_id"] is None
    assert PasswordService(time_cost=1, memory_kib=8192, parallelism=1).verify(
        rows[0]["password_hash"], password
    )
    assert run(argv) == cli.EXIT_CONFLICT

    weak = tmp_path / "weak"
    weak.write_text("password123\n", encoding="utf-8")
    other = [
        "create-platform-admin",
        "--email",
        f"w-{email}",
        "--name",
        "W",
        "--password-file",
        str(weak),
    ]
    assert run(other) == cli.EXIT_INPUT

    answers = iter(["First-Passphrase-123", "Different-Passphrase-456"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: next(answers))
    assert run(["create-platform-admin", "--email", f"g-{email}", "--name", "G"]) == cli.EXIT_INPUT


@pytest.mark.db
def test_audit_verify_detects_tampering(
    cli_env: Any, test_database: Any, settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from docassist.api.container import build_container
    from docassist.audit.service import AuditSealer

    slug = f"aud-{uuid.uuid4().hex[:8]}"
    assert (
        run(
            [
                "create-org",
                "--slug",
                slug,
                "--name",
                "Audit Co",
                "--admin-email",
                f"a-{slug}@example.test",
                "--admin-name",
                "A",
            ]
        )
        == 0
    )

    async def seal() -> None:
        container = build_container(settings, role="worker")
        try:
            sealer = AuditSealer(
                container.worker_db, settings.security.audit_hmac_key.get_secret_value().encode()
            )
            while await sealer.seal_pending():
                pass
        finally:
            await container.close()

    asyncio.run(seal())
    capsys.readouterr()
    assert run(["audit-verify", "--org", slug]) == 0
    assert f"{slug}: valid" in capsys.readouterr().out

    org_id = _query(test_database.admin_url, "SELECT id FROM organizations WHERE slug = $1", slug)[
        0
    ]["id"]
    _execute_as_superuser(
        test_database.admin_url,
        (
            "UPDATE audit_events SET actor_role = 'organization_admin' WHERE organization_id = $1 AND seal_seq = 1",
            (org_id,),
        ),
    )
    assert run(["audit-verify", "--org", slug]) == cli.EXIT_VERIFY_FAILED
    assert f"{slug}: INVALID at seq 1: hash mismatch" in capsys.readouterr().err
    assert run(["audit-verify", "--org", "no-such-org"]) == cli.EXIT_INPUT

    # every chain: the platform chain is intact (org chains of other tests may be tampered)
    assert run(["audit-verify"]) in {0, cli.EXIT_VERIFY_FAILED}
    assert "platform: valid" in capsys.readouterr().out


@pytest.mark.db
def test_seed_demo_accounts_are_idempotent(
    cli_env: Any, test_database: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from docassist.demo.content import USERS

    creds = tmp_path / "secrets" / "demo.txt"
    base = ["seed-demo", "--no-documents", "--password-file", str(creds)]
    assert run([*base, "--reset-passwords"]) == 0
    out = capsys.readouterr().out
    if POSIX:
        assert stat.S_IMODE(creds.stat().st_mode) == 0o600
    lines = [
        line.split("\t")
        for line in creds.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    assert sorted(line[0] for line in lines) == sorted(u.email for u in USERS)
    passwords = {line[0]: line[3] for line in lines}
    assert not any(pw in out for pw in passwords.values())  # never printed

    hasher = PasswordService(time_cost=1, memory_kib=8192, parallelism=1)
    admin_email = "alice.admin@acme.example"
    stored = _query(
        test_database.admin_url, "SELECT password_hash FROM users WHERE email = $1", admin_email
    )
    assert hasher.verify(stored[0]["password_hash"], passwords[admin_email])

    before = creds.read_text(encoding="utf-8")
    assert run(base) == 0  # second run: nothing new, file untouched
    assert "kept their passwords" in capsys.readouterr().out
    assert creds.read_text(encoding="utf-8") == before
    counts = _query(
        test_database.admin_url,
        "SELECT (SELECT count(*) FROM organizations WHERE slug IN ('acme', 'globex')) AS orgs,"
        " (SELECT count(*) FROM departments d JOIN organizations o ON o.id = d.organization_id"
        "   WHERE o.slug IN ('acme', 'globex')) AS depts,"
        " (SELECT count(*) FROM users WHERE email = ANY($1::text[])) AS users",
        [u.email for u in USERS],
    )[0]
    assert (counts["orgs"], counts["depts"], counts["users"]) == (2, 6, len(USERS))
    manager = _query(
        test_database.admin_url,
        "SELECT d.slug, ud.is_manager FROM user_departments ud JOIN users u ON u.id = ud.user_id"
        " JOIN departments d ON d.id = ud.department_id WHERE u.email = 'frank.finance@acme.example'",
    )
    assert [tuple(m) for m in manager] == [("finance", True)]


@pytest.mark.db
@pytest.mark.slow
def test_seed_demo_uploads_the_corpus_through_the_documents_service(
    cli_env: Any,
    test_database: Any,
    tmp_path: Path,
    settings: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("docassist.documents.service")
    from docassist.api.container import build_container

    async def has_documents_service() -> bool:
        container = build_container(settings, role="api")
        try:
            return getattr(container, "documents", None) is not None
        finally:
            await container.close()

    if not asyncio.run(has_documents_service()):
        pytest.skip("the documents service is not wired in this build")
    creds = tmp_path / "demo.txt"
    assert run(["seed-demo", "--no-process", "--password-file", str(creds)]) == 0, (
        capsys.readouterr().err
    )
    rows = _query(
        test_database.admin_url,
        "SELECT o.slug, d.title, d.classification, d.status FROM documents d"
        " JOIN organizations o ON o.id = d.organization_id WHERE o.slug IN ('acme', 'globex')",
    )
    assert len(rows) == 15
    assert all(r["status"] in {"processing", "ready"} for r in rows), rows  # nothing quarantined
    titles = {r["title"]: r for r in rows}
    assert titles["Salary Sheet 2026"]["classification"] == "RESTRICTED"
    jobs = _query(
        test_database.admin_url,
        "SELECT count(*) AS n FROM jobs j JOIN organizations o ON o.id = j.organization_id"
        " WHERE o.slug IN ('acme', 'globex') AND j.kind = 'ingest_version'",
    )
    assert jobs[0]["n"] == 15
    capsys.readouterr()
    assert run(["seed-demo", "--no-process", "--password-file", str(creds)]) == 0
    assert "documents uploaded: 0, already present: 15" in capsys.readouterr().out


@pytest.mark.db
def test_seed_demo_reports_a_missing_worker_but_keeps_new_passwords(
    cli_env: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib

    from docassist.demo import seed

    real_import = importlib.import_module

    def no_worker(name: str, *args: Any) -> Any:
        if name == "docassist.jobs.worker":
            raise ImportError(name)
        return real_import(name, *args)

    monkeypatch.setattr(seed.importlib, "import_module", no_worker)
    creds = tmp_path / "demo.txt"
    code = run(["seed-demo", "--reset-passwords", "--password-file", str(creds)])
    captured = capsys.readouterr()
    assert code == cli.EXIT_UNAVAILABLE
    assert "background worker is unavailable" in captured.err
    assert "demo password(s) written" in captured.out
    assert (
        len(
            [
                line
                for line in creds.read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
            ]
        )
        == 12
    )


@pytest.mark.db
@pytest.mark.slow
def test_seed_demo_end_to_end_with_the_worker(
    cli_env: Any, test_database: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("docassist.jobs.worker")
    pytest.importorskip("docassist.documents.service")
    assert run(["seed-demo", "--password-file", str(tmp_path / "demo.txt")]) == 0, (
        capsys.readouterr().err
    )
    out = capsys.readouterr().out
    assert "background jobs processed:" in out
    rows = _query(
        test_database.admin_url,
        "SELECT d.title, d.status, d.current_version_id IS NOT NULL AS indexed,"
        " (SELECT count(*) FROM document_chunks c WHERE c.document_id = d.id) AS chunks"
        " FROM documents d JOIN organizations o ON o.id = d.organization_id"
        " WHERE o.slug IN ('acme', 'globex')",
    )
    assert len(rows) == 15
    not_ready = [(r["title"], r["status"]) for r in rows if r["status"] != "ready"]
    assert not not_ready
    assert all(r["indexed"] and r["chunks"] > 0 for r in rows)
    flagged = _query(
        test_database.admin_url,
        "SELECT max(c.injection_score) AS score FROM document_chunks c"
        " JOIN documents d ON d.id = c.document_id WHERE d.title = 'Vendor Onboarding Checklist'",
    )
    assert flagged[0]["score"] > 0  # the planted prompt injection is detected at ingestion


@pytest.mark.db
@pytest.mark.slow
def test_evaluate_loads_the_golden_corpus_and_reports_metrics(
    cli_env: Any, test_database: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation = pytest.importorskip("docassist.rag.evaluation")
    pytest.importorskip("docassist.jobs.worker")
    import json

    report_path = tmp_path / "eval.json"
    code = run(["evaluate", "--output", str(report_path)])
    captured = capsys.readouterr()
    assert code in {0, cli.EXIT_VERIFY_FAILED}, captured.err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["metrics"]) >= set(evaluation.DEFAULT_THRESHOLDS)
    assert len(report["results"]) == len(evaluation.GOLDEN_DATASET.questions)
    assert ("evaluation passed" in captured.out) == (code == 0)
    orgs = _query(
        test_database.admin_url, "SELECT slug FROM organizations WHERE slug LIKE 'eval-%'"
    )
    assert {r["slug"] for r in orgs} == {"eval-acme", "eval-globex"}
    docs = _query(
        test_database.admin_url,
        "SELECT count(*) AS n FROM documents d JOIN organizations o ON o.id = d.organization_id"
        " WHERE o.slug LIKE 'eval-%' AND d.status = 'ready'",
    )
    assert docs[0]["n"] == len(evaluation.GOLDEN_DATASET.documents)
    personas = _query(
        test_database.admin_url,
        "SELECT u.email, u.role FROM users u JOIN organizations o ON o.id = u.organization_id"
        " WHERE o.slug LIKE 'eval-%'",
    )
    assert {r["email"].split("@")[0] for r in personas} == {
        u.key.replace("_", "-") for u in evaluation.GOLDEN_DATASET.users
    }
    assert "password" not in captured.out.lower()  # personas get no usable credentials


def test_python_dash_m_entry_point() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "docassist", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("docassist ")


def test_worker_command_builds_a_worker_container_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    events: list[tuple[str, Any]] = []

    class FakeContainer:
        async def close(self) -> None:
            events.append(("closed", None))

    class FakeWorker:
        def __init__(self, container: Any, *, concurrency: int, poll_interval: float) -> None:
            events.append(("worker", (concurrency, poll_interval)))

        async def run(self) -> None:
            events.append(("run", None))

    settings = make_settings(worker={"concurrency": 3, "poll_interval_seconds": 0.5})
    monkeypatch.setattr(cli, "_load_settings", lambda: settings)
    monkeypatch.setattr(
        cli, "_build_container", lambda s, role: events.append(("role", role)) or FakeContainer()
    )
    monkeypatch.setattr(
        cli, "_import_optional", lambda name, what: types.SimpleNamespace(Worker=FakeWorker)
    )
    assert run(["worker"]) == 0
    assert events == [("role", "worker"), ("worker", (3, 0.5)), ("run", None), ("closed", None)]
    events.clear()
    assert run(["worker", "--concurrency", "7"]) == 0
    assert events[1] == ("worker", (7, 0.5))


@pytest.mark.db
def test_migrate_upgrades_a_fresh_database(
    test_database: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.conftest import TEST_DB_URL, _with_db

    assert TEST_DB_URL
    name = f"docassist_cli_{uuid.uuid4().hex[:8]}"
    server = TEST_DB_URL

    async def admin(sql: str) -> None:
        conn = await asyncpg.connect(server)
        try:
            await conn.execute(sql)
        finally:
            await conn.close()

    asyncio.run(admin(f'CREATE DATABASE "{name}"'))
    try:
        owner = _with_db(server, name).replace("postgresql://", "postgresql+asyncpg://", 1)
        # the CLI exports these for the migration; monkeypatch restores them afterwards
        monkeypatch.setenv("DOCASSIST_DATABASE__APP_ROLE", "docassist_app")
        monkeypatch.setenv("DOCASSIST_DATABASE__WORKER_ROLE", "docassist_worker")
        monkeypatch.setenv("DOCASSIST_EMBEDDING__DIMENSIONS", "1024")
        settings = make_settings(
            database={"url": "postgresql+asyncpg://unused@localhost/unused", "migration_url": owner}
        )
        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        assert run(["migrate"]) == 0
        assert capsys.readouterr().out.strip() == "database upgraded to head"
        tables = _query(
            _with_db(server, name),
            "SELECT count(*) AS n FROM pg_tables WHERE schemaname = 'public' AND tablename IN ('users', 'audit_events', 'alembic_version')",
        )
        assert tables[0]["n"] == 3
        assert run(["migrate"]) == 0  # idempotent
    finally:
        asyncio.run(
            admin(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{name}'"
            )
        )
        asyncio.run(admin(f'DROP DATABASE IF EXISTS "{name}"'))
