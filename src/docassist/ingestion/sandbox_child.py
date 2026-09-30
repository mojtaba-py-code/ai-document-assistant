"""Parser sandbox - child side. Run only by :mod:`docassist.ingestion.sandbox`.

Protocol: the plaintext document arrives on **stdin** (it never touches the disk); exactly one
JSON envelope is written to **stdout**:

* ``{"ok": true, "document": {...}}`` (exit 0) - a serialised ``model.ParsedDocument``;
* ``{"ok": false, "error": "<code>"}`` (exit 2) - ``code`` is one of ``model.ERROR_CODES``;
* ``--format probe`` answers ``{"ok": true, "probe": {...}}`` with *names* of environment
  variables and loaded top-level modules plus the applied limits. Operators use it to check
  a deployment's isolation (``ParserSandbox.probe``); it never echoes values.

Hardening applied before any document byte is read:

* Linux: best-effort ``unshare(CLONE_NEWUSER | CLONE_NEWNET)`` - a new network namespace
  with no interfaces, so parser code cannot reach the network (``network_isolated``);
* POSIX: ``RLIMIT_AS`` (memory), ``RLIMIT_CPU``, ``RLIMIT_FSIZE = 0`` (no file writes; the
  interpreter runs with ``-B`` so no bytecode is written and ``SIGXFSZ`` is ignored so a
  write fails instead of killing the process), ``RLIMIT_NOFILE = 64``, ``RLIMIT_CORE = 0``
  (no core dump containing plaintext) and ``RLIMIT_NPROC = 0`` where available (no fork);
* ``defusedxml.defuse_stdlib()``, a modest recursion limit.

This module may import only the standard library, ``defusedxml`` and
``docassist.ingestion.{model,parsers}`` (enforced by an import-linter contract).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import sys
from typing import Any

from docassist.ingestion.model import FORMATS, Limits

RECURSION_LIMIT = 2_000
MAX_OPEN_FILES = 64
EXIT_OK = 0
EXIT_PARSE_ERROR = 2


def _emit(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Lone surrogates (possible in broken text layers) cannot be UTF-8 encoded: replace them.
    sys.stdout.buffer.write(body.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def _isolate_network() -> bool:
    isolated = False
    if sys.platform == "linux":
        try:
            os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
            isolated = True
        except (OSError, AttributeError):
            isolated = False  # user namespaces disabled: parsing proceeds without isolation
    return isolated


def _apply_rlimits(memory_mb: int, cpu_seconds: int) -> tuple[bool, dict[str, list[int]]]:
    """POSIX resource limits; returns (every limit applied, effective ``[soft, hard]`` values)."""
    report: dict[str, list[int]] = {}
    if os.name != "posix":
        return False, report
    resource: Any = importlib.import_module("resource")
    sigxfsz = getattr(signal, "SIGXFSZ", None)
    if sigxfsz is not None:
        signal.signal(sigxfsz, signal.SIG_IGN)  # a write beyond RLIMIT_FSIZE fails, not kills
    wanted: list[tuple[str, int, int]] = [
        ("RLIMIT_AS", memory_mb * 1024 * 1024, memory_mb * 1024 * 1024),
        ("RLIMIT_CPU", cpu_seconds, cpu_seconds + 1),
        ("RLIMIT_FSIZE", 0, 0),
        ("RLIMIT_NOFILE", MAX_OPEN_FILES, MAX_OPEN_FILES),
        ("RLIMIT_CORE", 0, 0),
        ("RLIMIT_NPROC", 0, 0),
    ]
    applied = True
    for name, wanted_soft, wanted_hard in wanted:
        resource_id = getattr(resource, name, None)
        if resource_id is None:
            continue
        try:
            soft, hard = wanted_soft, wanted_hard
            _, current_hard = resource.getrlimit(resource_id)
            if current_hard != resource.RLIM_INFINITY:  # never try to raise a hard limit
                hard = min(hard, current_hard)
                soft = min(soft, hard)
            resource.setrlimit(resource_id, (soft, hard))
            report[name] = list(resource.getrlimit(resource_id))
        except (OSError, ValueError):
            applied = False
    return applied, report


def _read_input(max_bytes: int) -> bytes | None:
    data = sys.stdin.buffer.read(max_bytes + 1)
    return None if len(data) > max_bytes else data


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="docassist-sandbox", add_help=False)
    parser.add_argument("--format", required=True, choices=[*FORMATS, "probe"])
    parser.add_argument("--limits", required=True)
    parser.add_argument("--max-input-bytes", type=int, required=True)
    parser.add_argument("--memory-mb", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--ocr-images", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    args = _parse_args(argv)
    limits = Limits.from_arg(args.limits)
    network_isolated = _isolate_network()
    rlimits_applied, rlimits = _apply_rlimits(args.memory_mb, args.cpu_seconds)

    import defusedxml

    defusedxml.defuse_stdlib()
    sys.setrecursionlimit(RECURSION_LIMIT)

    from docassist.ingestion.parsers import ParseError, get_parser, parse_document

    if args.format == "probe":
        for fmt in FORMATS:  # report the widest import set any document could trigger
            get_parser(fmt)
        _emit(
            {
                "ok": True,
                "probe": {
                    "env_keys": sorted(os.environ),
                    "modules": sorted({name.partition(".")[0] for name in sys.modules}),
                    "docassist_modules": sorted(
                        n for n in sys.modules if n.startswith("docassist")
                    ),
                    "network_isolated": network_isolated,
                    "rlimits_applied": rlimits_applied,
                    "rlimits": rlimits,
                    "platform": sys.platform,
                },
            }
        )
        return EXIT_OK

    data = _read_input(args.max_input_bytes)
    if data is None:
        _emit({"ok": False, "error": "too_large"})
        return EXIT_PARSE_ERROR
    try:
        document = parse_document(args.format, data, limits, ocr_images=args.ocr_images)
    except ParseError as exc:
        _emit({"ok": False, "error": exc.code})
        return EXIT_PARSE_ERROR
    del data
    document = document.with_sandbox_facts(
        network_isolated=network_isolated, rlimits_applied=rlimits_applied
    )
    _emit({"ok": True, "document": document.to_json()})
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
