"""Parser sandbox - parent side.

Untrusted files are parsed only in a short-lived child interpreter
(``python -I -B -m docassist.ingestion.sandbox_child``) started with
``asyncio.create_subprocess_exec`` (argument list, never a shell) and:

* a **minimal environment**: ``PATH``, ``SYSTEMROOT`` (Windows needs it), ``LANG``/``LC_ALL``
  and ``TMP``/``TEMP``/``TMPDIR`` pointing at a private, per-run temp directory that is also
  the working directory and is deleted afterwards. No secret and no ``DOCASSIST_*`` variable
  is inherited; ``-I`` additionally makes the interpreter ignore every ``PYTHON*`` variable
  and the user site directory;
* the plaintext on **stdin** only (never written to disk);
* stdout **capped** at ``parser.max_output_bytes`` (the child is killed when it exceeds it);
  stderr capped and discarded - only a fixed error code and, for a crashed child, the
  exception *class name* (never its message) survive;
* a **wall-clock timeout** (``parser.timeout_seconds``): the child's whole process tree is
  killed (:mod:`docassist.ingestion.processes`) and :class:`SandboxError` ``parse_timeout``
  is raised;
* a **concurrency bound** (``asyncio.Semaphore(parser.max_concurrency)``);
* resource limits and network isolation applied by the child itself (see
  :mod:`docassist.ingestion.sandbox_child`).

The child's JSON is validated strictly (:func:`model.parsed_document_from_json`) before use.
:class:`SandboxError` is a :class:`~docassist.jobs.queue.PermanentJobError`: retrying the same
bytes through the same parser cannot succeed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docassist.core.logging import get_logger
from docassist.ingestion.model import (
    ERROR_CODES,
    FORMATS,
    Limits,
    ModelValidationError,
    ParsedDocument,
    parsed_document_from_json,
)
from docassist.ingestion.processes import terminate
from docassist.jobs.queue import PermanentJobError

if TYPE_CHECKING:
    from docassist.core.config import Settings

log = get_logger(__name__)

SANDBOX_ERROR_CODES: tuple[str, ...] = (*ERROR_CODES, "parse_timeout")
_READ_CHUNK = 65_536
_ENV_PASSTHROUGH = ("PATH", "SYSTEMROOT")
CHILD_MODULE = "docassist.ingestion.sandbox_child"


class SandboxError(PermanentJobError):
    """Parsing failed in (or because of) the sandbox. ``code`` is in :data:`SANDBOX_ERROR_CODES`."""

    def __init__(self, code: str) -> None:
        if code not in SANDBOX_ERROR_CODES:
            code = "parse_error"
        super().__init__(f"document could not be parsed ({code})", code=code)


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    timeout_seconds: float = 60.0
    memory_limit_mb: int = 1_024
    cpu_seconds: int = 60
    max_output_bytes: int = 67_108_864
    max_concurrency: int = 2
    max_input_bytes: int = 52_428_800
    max_stderr_bytes: int = 65_536
    temp_root: Path | None = None
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def from_settings(cls, settings: Settings) -> SandboxConfig:
        parser = settings.parser
        out = parser.max_output_bytes
        # Output budget: JSON escaping costs at most ~6 bytes per character and block
        # envelopes ~100 bytes, so these caps keep an honest child below max_output_bytes.
        limits = Limits(
            max_pages=settings.upload.max_pdf_pages,
            max_total_chars=max(1_000, out // 8),
            max_blocks=max(100, out // 512),
            max_block_chars=min(100_000, max(1_000, out // 16)),
            max_ocr_bytes=max(1_024, out // 4),
        )
        return cls(
            timeout_seconds=parser.timeout_seconds,
            memory_limit_mb=parser.memory_limit_mb,
            cpu_seconds=parser.cpu_seconds,
            max_output_bytes=out,
            max_concurrency=parser.max_concurrency,
            max_input_bytes=settings.upload.max_upload_bytes,
            temp_root=settings.storage.temp_dir,
            limits=limits,
        )


CommandFactory = Callable[[str, bool], list[str]]
"""``(format, ocr_images) -> argv``; tests substitute misbehaving children through it."""


@dataclass(slots=True)
class _Captured:
    data: bytes = b""
    overflow: bool = False


_ERROR_TYPE = re.compile(
    rb"^([A-Za-z_][A-Za-z0-9_.]{0,80}(?:Error|Exception|Exit|Interrupt))\b", re.MULTILINE
)


def _error_type(stderr: bytes) -> str | None:
    """The last exception *class name* in the child's stderr - never the message."""
    names = _ERROR_TYPE.findall(stderr[-8_192:])
    return names[-1].decode("ascii") if names else None


def child_environment(temp_dir: str) -> dict[str, str]:
    """The complete environment handed to the child: an allowlist, never a copy."""
    env = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
    env.update(
        {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TMP": temp_dir,
            "TEMP": temp_dir,
            "TMPDIR": temp_dir,
        }
    )
    return env


class ParserSandbox:
    def __init__(self, config: SandboxConfig, *, command: CommandFactory | None = None) -> None:
        self.config = config
        self._command = command or self.default_command
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    def default_command(self, fmt: str, ocr_images: bool) -> list[str]:
        argv = [
            sys.executable, "-I", "-B", "-m", CHILD_MODULE,
            "--format", fmt,
            "--limits", self.config.limits.to_arg(),
            "--max-input-bytes", str(self.config.max_input_bytes),
            "--memory-mb", str(self.config.memory_limit_mb),
            "--cpu-seconds", str(self.config.cpu_seconds),
        ]  # fmt: skip
        if ocr_images:
            argv.append("--ocr-images")
        return argv

    # ------------------------------------------------------------------ public API
    async def parse(self, fmt: str, data: bytes, *, ocr_images: bool = False) -> ParsedDocument:
        """Parse ``data`` as ``fmt`` in a child process. Raises :class:`SandboxError`."""
        if fmt not in FORMATS:
            raise SandboxError("unsupported")
        if len(data) > self.config.max_input_bytes:
            raise SandboxError("too_large")
        stdout, returncode = await self._run(self._command(fmt, ocr_images and fmt == "pdf"), data)
        envelope = await self._decode(stdout, returncode)
        if envelope.get("ok") is not True:
            error = envelope.get("error")
            raise SandboxError(
                error if isinstance(error, str) and error in ERROR_CODES else "parse_error"
            )
        try:
            return await asyncio.to_thread(
                parsed_document_from_json,
                envelope.get("document"),
                expected_format=fmt,
                limits=self.config.limits,
            )
        except ModelValidationError as exc:
            log.warning("sandbox_output_rejected", format=fmt, reason=str(exc)[:120])
            raise SandboxError("parse_error") from exc

    async def probe(self) -> dict[str, Any]:
        """Run the child's diagnostics mode: env *names*, loaded modules, limits, isolation."""
        stdout, returncode = await self._run(self._command("probe", False), b"")
        envelope = await self._decode(stdout, returncode)
        probe = envelope.get("probe")
        if envelope.get("ok") is not True or not isinstance(probe, dict):
            raise SandboxError("parse_error")
        return probe

    # ------------------------------------------------------------------ internals
    async def _decode(self, stdout: bytes, returncode: int | None) -> dict[str, Any]:
        if not stdout:
            code = "parse_error"
            if sys.platform != "win32" and returncode is not None and -returncode == signal.SIGXCPU:
                code = "parse_timeout"  # RLIMIT_CPU exhausted
            raise SandboxError(code)
        try:
            envelope = await asyncio.to_thread(json.loads, stdout)
        except (ValueError, RecursionError) as exc:
            raise SandboxError("parse_error") from exc
        if not isinstance(envelope, dict):
            raise SandboxError("parse_error")
        return envelope

    async def _run(self, argv: list[str], data: bytes) -> tuple[bytes, int | None]:
        """Run one child; returns (stdout, returncode). Raises :class:`SandboxError`."""
        async with self._semaphore:
            temp_dir = tempfile.mkdtemp(prefix="docassist-sbx-", dir=self.config.temp_root)
            try:
                return await self._run_in(argv, data, temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def _run_in(
        self, argv: list[str], data: bytes, temp_dir: str
    ) -> tuple[bytes, int | None]:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=temp_dir,
            env=child_environment(temp_dir),
            start_new_session=os.name == "posix",
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:  # pragma: no cover
            raise RuntimeError("sandbox pipes were not created")
        feeder = asyncio.create_task(self._feed(proc.stdin, data))
        stderr = asyncio.create_task(
            self._read(proc.stderr, self.config.max_stderr_bytes, keep_draining=True)
        )
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                captured = await self._read(proc.stdout, self.config.max_output_bytes)
                if captured.overflow:
                    raise SandboxError("too_large")
                await proc.wait()
                diagnostics = await stderr
        except TimeoutError as exc:
            raise SandboxError("parse_timeout") from exc
        finally:
            await terminate(proc)  # the whole process tree
            for task in (feeder, stderr):
                task.cancel()
            await asyncio.gather(feeder, stderr, return_exceptions=True)
            if not proc.stdin.is_closing():
                proc.stdin.close()
            # Close the subprocess transport explicitly (asyncio only does it implicitly, and on
            # the Windows proactor loop a transport collected before that raises a
            # ResourceWarning / "I/O operation on closed pipe" during garbage collection).
            transport = getattr(proc, "_transport", None)
            if transport is not None and not transport.is_closing():
                transport.close()
            await asyncio.sleep(0)  # let the pipe transports finish closing
        if proc.returncode not in (0, 2):
            log.warning(
                "sandbox_child_failed",
                returncode=proc.returncode,
                error_type=_error_type(diagnostics.data),
            )
        return captured.data, proc.returncode

    @staticmethod
    async def _feed(stdin: asyncio.StreamWriter, data: bytes) -> None:
        try:
            for start in range(0, len(data), _READ_CHUNK):
                stdin.write(data[start : start + _READ_CHUNK])
                await stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the child stopped reading (it failed or rejected the input early)
        finally:
            stdin.close()  # EOF for the child; also lets the subprocess transport finish
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await stdin.wait_closed()

    @staticmethod
    async def _read(
        stream: asyncio.StreamReader, cap: int, *, keep_draining: bool = False
    ) -> _Captured:
        """Read up to ``cap`` bytes; beyond it stop (stdout) or discard the rest (stderr)."""
        captured = _Captured()
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = await stream.read(_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > cap:
                captured.overflow = True
                if keep_draining:
                    continue
                break
            chunks.append(chunk)
        captured.data = b"".join(chunks)
        return captured
