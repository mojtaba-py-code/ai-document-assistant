"""Optional OCR for scanned PDF pages (``parser.ocr_enabled``), via the ``tesseract`` binary.

* Images are extracted from the PDF **inside the parser sandbox** (JPEG/JPEG 2000 passed
  through, raw samples converted to PNM); this module never decodes a document itself.
* ``tesseract stdin stdout -l <langs> --psm 3`` runs through ``create_subprocess_exec``
  (argument list, no shell) with a minimal environment, the image on stdin and a timeout
  (the process tree is killed on expiry); the recognised text is truncated at
  :data:`MAX_OCR_OUTPUT_BYTES`. ``OMP_THREAD_LIMIT=1`` keeps one OCR job from
  monopolising the CPU.
* ``parser.ocr_languages`` must look like ``eng`` or ``eng+deu`` - anything else is refused
  at construction, so configuration cannot smuggle extra arguments.
* When OCR is disabled or the binary is missing, :meth:`OcrEngine.available` is false and
  the pipeline records the warning ``ocr_unavailable`` and ``needs_ocr = true`` instead.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docassist.core.logging import get_logger
from docassist.ingestion.processes import terminate

if TYPE_CHECKING:
    from docassist.core.config import ParserSettings

log = get_logger(__name__)

_LANGUAGES = re.compile(r"^[a-z_]{3,16}(?:\+[a-z_]{3,16}){0,7}$")
MAX_OCR_OUTPUT_BYTES = 2_000_000
_ENV_PASSTHROUGH = ("PATH", "SYSTEMROOT", "TESSDATA_PREFIX")


class OcrError(Exception):
    """OCR failed for one image (timeout, crash); the page keeps ``needs_ocr``."""


@dataclass(frozen=True, slots=True)
class OcrConfig:
    enabled: bool
    command: tuple[str, ...]
    languages: str = "eng"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not _LANGUAGES.fullmatch(self.languages):
            raise ValueError("parser.ocr_languages must look like 'eng' or 'eng+deu'")
        if self.enabled and not self.command:
            raise ValueError("an OCR command is required when OCR is enabled")

    @classmethod
    def from_settings(cls, parser: ParserSettings) -> OcrConfig:
        return cls(
            enabled=parser.ocr_enabled,
            command=(parser.tesseract_path,),
            languages=parser.ocr_languages,
            timeout_seconds=parser.timeout_seconds,
        )


class OcrEngine:
    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self._resolved: list[str] | None = None
        if config.enabled and config.command:
            executable = shutil.which(config.command[0])
            if executable is not None:
                self._resolved = [executable, *config.command[1:]]

    @property
    def available(self) -> bool:
        return self.config.enabled and self._resolved is not None

    def _argv(self) -> list[str]:
        if self._resolved is None:
            raise OcrError("OCR is not available")
        return [*self._resolved, "stdin", "stdout", "-l", self.config.languages, "--psm", "3"]

    @staticmethod
    def _environment() -> dict[str, str]:
        env = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
        env["OMP_THREAD_LIMIT"] = "1"
        return env

    async def recognize(self, image: bytes) -> str:
        """Text recognised in one image. Raises :class:`OcrError` on failure or timeout."""
        proc = await asyncio.create_subprocess_exec(
            *self._argv(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._environment(),
            start_new_session=os.name == "posix",
        )
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                stdout, _ = await proc.communicate(image)
        except TimeoutError as exc:
            raise OcrError("OCR timed out") from exc
        finally:
            await terminate(proc)
        if proc.returncode != 0:
            raise OcrError(f"OCR exited with status {proc.returncode}")
        return stdout[:MAX_OCR_OUTPUT_BYTES].decode("utf-8", errors="replace")

    async def recognize_all(self, images: Sequence[bytes]) -> tuple[list[str], int]:
        """Recognise several images sequentially; returns (texts, failures)."""
        texts: list[str] = []
        failures = 0
        for image in images:
            try:
                texts.append(await self.recognize(image))
            except (OcrError, OSError) as exc:
                failures += 1
                log.warning("ocr_failed", error_type=type(exc).__name__)
        return texts, failures
