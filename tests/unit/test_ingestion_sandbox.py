"""Parser sandbox: isolation, limits, protocol validation, failure mapping."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from docassist.ingestion.model import Limits
from docassist.ingestion.sandbox import (
    ParserSandbox,
    SandboxConfig,
    SandboxError,
    _error_type,
    child_environment,
)
from docassist.jobs.queue import PermanentJobError
from tests.conftest import make_settings
from tests.helpers_ingestion import CONTRACT_TEXT

ALLOWED_ENV = {"LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR"}
# Real child processes are slow to start on loaded CI machines: generous timeouts.
SLOW = 300.0


def config(tmp_path: Path, **overrides: Any) -> SandboxConfig:
    base: dict[str, Any] = {"timeout_seconds": SLOW, "temp_root": tmp_path}
    base.update(overrides)
    return SandboxConfig(**base)


def python(code: str) -> Any:
    """A command factory that runs ``code`` instead of the real child."""

    def factory(fmt: str, ocr_images: bool) -> list[str]:
        return [sys.executable, "-I", "-c", code]

    return factory


def test_config_from_settings() -> None:
    settings = make_settings(
        parser={
            "timeout_seconds": 12,
            "memory_limit_mb": 256,
            "cpu_seconds": 7,
            "max_output_bytes": 8_388_608,
            "max_concurrency": 3,
        },
        upload={"max_pdf_pages": 50, "max_upload_bytes": 2_000_000},
    )
    cfg = SandboxConfig.from_settings(settings)
    assert (cfg.timeout_seconds, cfg.memory_limit_mb, cfg.cpu_seconds, cfg.max_concurrency) == (
        12,
        256,
        7,
        3,
    )
    assert cfg.max_input_bytes == 2_000_000 and cfg.limits.max_pages == 50
    assert cfg.limits.max_total_chars == 8_388_608 // 8


def test_child_environment_is_an_allowlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCASSIST_SECURITY__JWT_SIGNING_KEY", "super-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "also-secret")
    monkeypatch.setenv("PYTHONPATH", "/tmp/evil")
    env = child_environment(str(tmp_path))
    assert set(env) <= ALLOWED_ENV
    assert env["TMP"] == env["TEMP"] == env["TMPDIR"] == str(tmp_path)
    assert "super-secret" not in json.dumps(env) and "also-secret" not in json.dumps(env)


def test_default_command_uses_isolated_interpreter(tmp_path: Path) -> None:
    sandbox = ParserSandbox(config(tmp_path, limits=Limits(max_pages=7)))
    argv = sandbox.default_command("pdf", True)
    assert argv[:5] == [sys.executable, "-I", "-B", "-m", "docassist.ingestion.sandbox_child"]
    assert "--ocr-images" in argv and "max_pages=7" in argv[argv.index("--limits") + 1]
    assert "--ocr-images" not in sandbox.default_command("md", False)


def test_error_type_keeps_only_class_names() -> None:
    stderr = b"Traceback (most recent call last):\n  File x\nValueError: secret document text\n"
    assert _error_type(stderr) == "ValueError"
    assert _error_type(b"garbage output") is None


def test_sandbox_error_is_permanent_and_sanitised() -> None:
    error = SandboxError("parse_timeout")
    assert isinstance(error, PermanentJobError) and error.code == "parse_timeout"
    assert SandboxError("something odd").code == "parse_error"


# --------------------------------------------------------------------------- #
# Real child process
# --------------------------------------------------------------------------- #
async def test_real_child_parses_markdown(tmp_path: Path) -> None:
    sandbox = ParserSandbox(config(tmp_path))
    document = await sandbox.parse("md", CONTRACT_TEXT.encode())
    kinds = [block.kind for page in document.pages for block in page.blocks]
    assert kinds[0] == "heading" and "paragraph" in kinds
    assert document.format == "md" and isinstance(document.network_isolated, bool)
    assert [p.name for p in tmp_path.iterdir()] == []  # private temp dir removed


async def test_real_child_environment_and_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DOCASSIST_DATABASE__URL", "postgresql://secret@db/prod")
    probe = await ParserSandbox(config(tmp_path)).probe()
    assert set(probe["env_keys"]) <= ALLOWED_ENV
    assert not [k for k in probe["env_keys"] if k.upper().startswith("DOCASSIST")]
    forbidden = {
        "sqlalchemy",
        "asyncpg",
        "httpx",
        "anthropic",
        "redis",
        "fastapi",
        "starlette",
        "pydantic",
        "numpy",  # its OpenBLAS threads kill the child under RLIMIT_NPROC = 0
    }
    assert forbidden.isdisjoint(probe["modules"])
    assert {"pypdf", "docx", "openpyxl"} <= set(probe["modules"])
    allowed = (
        "docassist.ingestion.model",
        "docassist.ingestion.parsers",
        "docassist.ingestion.sandbox_child",
    )
    for name in probe["docassist_modules"]:  # never config, db, security, llm, api...
        assert name in {"docassist", "docassist.ingestion"} or name.startswith(allowed), name


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
async def test_real_child_applies_posix_limits(tmp_path: Path) -> None:
    probe = await ParserSandbox(config(tmp_path, memory_limit_mb=512, cpu_seconds=9)).probe()
    assert probe["rlimits_applied"] is True
    limits = probe["rlimits"]
    assert limits["RLIMIT_FSIZE"] == [0, 0] and limits["RLIMIT_CORE"] == [0, 0]
    assert limits["RLIMIT_NOFILE"][1] <= 64 and limits["RLIMIT_CPU"][0] <= 9
    assert limits["RLIMIT_AS"][0] <= 512 * 1024 * 1024


async def test_real_child_reports_parse_errors(tmp_path: Path) -> None:
    with pytest.raises(SandboxError) as caught:
        await ParserSandbox(config(tmp_path)).parse("docx", b"PK\x03\x04 not really a zip")
    assert caught.value.code == "parse_error"


async def test_real_child_enforces_its_input_cap(tmp_path: Path) -> None:
    sandbox = ParserSandbox(config(tmp_path))
    original = sandbox.default_command

    def small_cap(fmt: str, ocr: bool) -> list[str]:
        argv = original(fmt, ocr)
        argv[argv.index("--max-input-bytes") + 1] = "10"
        return argv

    with pytest.raises(SandboxError) as caught:
        await ParserSandbox(sandbox.config, command=small_cap).parse("txt", b"x" * 100)
    assert caught.value.code == "too_large"


# --------------------------------------------------------------------------- #
# Misbehaving children (substituted commands)
# --------------------------------------------------------------------------- #
async def test_wall_clock_timeout_kills_the_child(tmp_path: Path) -> None:
    sandbox = ParserSandbox(
        config(tmp_path, timeout_seconds=1.0), command=python("import time; time.sleep(120)")
    )
    started = asyncio.get_running_loop().time()
    with pytest.raises(SandboxError) as caught:
        await sandbox.parse("txt", b"hello")
    assert caught.value.code == "parse_timeout"
    assert asyncio.get_running_loop().time() - started < 60


async def test_output_cap_kills_the_child(tmp_path: Path) -> None:
    code = "import sys; sys.stdout.write('x' * 5_000_000); sys.stdout.flush()"
    sandbox = ParserSandbox(config(tmp_path, max_output_bytes=10_000), command=python(code))
    with pytest.raises(SandboxError) as caught:
        await sandbox.parse("txt", b"hello")
    assert caught.value.code == "too_large"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("import sys; sys.exit(3)", "parse_error"),
        ("print('not json')", "parse_error"),
        ("print('[1, 2, 3]')", "parse_error"),
        ("import json; print(json.dumps({'ok': False, 'error': 'unsupported'}))", "unsupported"),
        ("import json; print(json.dumps({'ok': False, 'error': 'rm -rf'}))", "parse_error"),
        (
            "import json; print(json.dumps({'ok': True, 'document': {'format': 'txt'}}))",
            "parse_error",
        ),
        ("import sys; sys.stderr.write('E' * 200000); sys.exit(1)", "parse_error"),
    ],
    ids=[
        "crash",
        "garbage",
        "not-object",
        "error-code",
        "bogus-code",
        "invalid-document",
        "noisy-stderr",
    ],
)
async def test_bad_child_output_is_a_clean_error(tmp_path: Path, code: str, expected: str) -> None:
    sandbox = ParserSandbox(config(tmp_path), command=python(code))
    with pytest.raises(SandboxError) as caught:
        await sandbox.parse("txt", b"hello")
    assert caught.value.code == expected


async def test_valid_output_from_a_substituted_child_is_accepted(tmp_path: Path) -> None:
    document = {
        "format": "txt",
        "pages": [{"number": 1, "blocks": [{"kind": "paragraph", "text": "hi", "level": None, "hidden": ""}], "needs_ocr": False}],
        "metadata": {"title": None, "author": None, "subject": None, "created": None, "modified": None,
                     "page_count": 1, "sheet_names": [], "page_basis": "single"},
        "warnings": [], "needs_ocr": False, "network_isolated": True, "rlimits_applied": True, "ocr_images": [],
    }  # fmt: skip
    code = (
        "import json, sys; sys.stdin.read(); print(json.dumps("
        + repr({"ok": True, "document": document})
        + "))"
    )
    parsed = await ParserSandbox(config(tmp_path), command=python(code)).parse("txt", b"hello")
    assert parsed.pages[0].blocks[0].text == "hi" and parsed.network_isolated


async def test_unsupported_and_oversized_input_never_spawn(tmp_path: Path) -> None:
    def explode(fmt: str, ocr: bool) -> list[str]:
        raise AssertionError("must not spawn")

    sandbox = ParserSandbox(config(tmp_path, max_input_bytes=10), command=explode)
    with pytest.raises(SandboxError) as caught:
        await sandbox.parse("exe", b"MZ")
    assert caught.value.code == "unsupported"
    with pytest.raises(SandboxError) as caught:
        await sandbox.parse("txt", b"x" * 11)
    assert caught.value.code == "too_large"


async def test_concurrency_is_bounded(tmp_path: Path) -> None:
    active = peak = 0

    class Counting(ParserSandbox):
        async def _run_in(
            self, argv: list[str], data: bytes, temp_dir: str
        ) -> tuple[bytes, int | None]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return json.dumps({"ok": False, "error": "empty_document"}).encode(), 2

    sandbox = Counting(config(tmp_path, max_concurrency=2))
    results = await asyncio.gather(
        *(sandbox.parse("txt", b"x") for _ in range(6)), return_exceptions=True
    )
    assert all(isinstance(r, SandboxError) and r.code == "empty_document" for r in results)
    assert peak == 2
