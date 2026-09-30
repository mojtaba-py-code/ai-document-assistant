"""OCR engine: configuration hardening and a fake tesseract binary (no real tesseract needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from docassist.core.config import ParserSettings
from docassist.ingestion.ocr import OcrConfig, OcrEngine, OcrError


def fake(tmp_path: Path, body: str) -> tuple[str, ...]:
    script = tmp_path / "fake_tesseract.py"
    script.write_text("import sys, os, json, time\n" + body, encoding="utf-8")
    return (sys.executable, str(script))


@pytest.mark.parametrize(
    "languages", ["eng --psm 0", "../../etc", "eng;rm", "e", "eng+" * 9 + "eng"]
)
def test_language_argument_is_validated(languages: str) -> None:
    with pytest.raises(ValueError, match="ocr_languages"):
        OcrConfig(enabled=False, command=("tesseract",), languages=languages)


def test_disabled_or_missing_binary_is_unavailable() -> None:
    assert not OcrEngine(OcrConfig.from_settings(ParserSettings())).available  # disabled by default
    missing = OcrEngine(
        OcrConfig(enabled=True, command=("definitely-not-a-real-tesseract-binary",))
    )
    assert not missing.available
    with pytest.raises(ValueError):
        OcrConfig(enabled=True, command=())


async def test_recognize_passes_image_on_stdin(tmp_path: Path) -> None:
    command = fake(
        tmp_path,
        "data = sys.stdin.buffer.read()\n"
        "print(json.dumps({'argv': sys.argv[1:], 'size': len(data), 'env': sorted(os.environ)}))\n",
    )
    engine = OcrEngine(
        OcrConfig(enabled=True, command=command, languages="eng+deu", timeout_seconds=120)
    )
    assert engine.available
    output = json.loads(await engine.recognize(b"\x89PNG-image"))
    assert output["argv"] == ["stdin", "stdout", "-l", "eng+deu", "--psm", "3"]
    assert output["size"] == len(b"\x89PNG-image")
    assert set(output["env"]) <= {"PATH", "SYSTEMROOT", "TESSDATA_PREFIX", "OMP_THREAD_LIMIT"}


async def test_failures_raise_ocr_error(tmp_path: Path) -> None:
    crash = OcrEngine(
        OcrConfig(enabled=True, command=fake(tmp_path, "sys.exit(2)\n"), timeout_seconds=120)
    )
    with pytest.raises(OcrError):
        await crash.recognize(b"x")
    slow = OcrEngine(
        OcrConfig(enabled=True, command=fake(tmp_path, "time.sleep(60)\n"), timeout_seconds=1)
    )
    with pytest.raises(OcrError, match="timed out"):
        await slow.recognize(b"x")


async def test_recognize_all_counts_failures(tmp_path: Path) -> None:
    command = fake(
        tmp_path, "if sys.stdin.buffer.read() == b'bad':\n    sys.exit(1)\nprint('text')\n"
    )
    engine = OcrEngine(OcrConfig(enabled=True, command=command, timeout_seconds=120))
    texts, failures = await engine.recognize_all([b"good", b"bad", b"good"])
    assert failures == 1 and [t.strip() for t in texts] == ["text", "text"]


async def test_unavailable_engine_refuses_to_run() -> None:
    engine = OcrEngine(OcrConfig(enabled=False, command=("tesseract",)))
    with pytest.raises(OcrError):
        await engine.recognize(b"x")
