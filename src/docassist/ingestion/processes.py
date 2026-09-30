"""Terminating helper processes (parser sandbox children, OCR) completely and promptly.

Killing only the direct child is not enough:

* on POSIX the child runs in its own session (``start_new_session=True``), so the whole
  process group is sent ``SIGKILL``;
* on Windows a virtual-environment ``python.exe`` is a *launcher* that starts the real
  interpreter as a grandchild. ``TerminateProcess`` on the launcher leaves the grandchild -
  and the pipes it holds - alive, and ``asyncio``'s ``Process.wait()`` only returns once
  every pipe is closed. The whole tree is therefore killed with ``taskkill /T /F`` (from
  ``%SYSTEMROOT%``, never looked up on ``PATH``).

The final wait is bounded, so a process that cannot be reaped is logged, never awaited
forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys

from docassist.core.logging import get_logger

log = get_logger(__name__)

REAP_TIMEOUT_SECONDS = 10.0


def _taskkill() -> str:
    root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return os.path.join(root, "System32", "taskkill.exe")


async def _kill_tree_windows(pid: int) -> None:
    try:
        killer = await asyncio.create_subprocess_exec(
            _taskkill(), "/T", "/F", "/PID", str(pid),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        async with asyncio.timeout(REAP_TIMEOUT_SECONDS):
            await killer.wait()
    except (OSError, TimeoutError) as exc:
        log.warning("process_tree_kill_failed", error_type=type(exc).__name__)


async def terminate(proc: asyncio.subprocess.Process) -> None:
    """Kill ``proc`` and its descendants if still running, then reap it (bounded)."""
    if proc.returncode is None:
        if sys.platform == "win32":
            await _kill_tree_windows(proc.pid)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # callers start children in a new session
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
    try:
        async with asyncio.timeout(REAP_TIMEOUT_SECONDS):
            await proc.wait()
    except TimeoutError:
        log.warning("process_not_reaped", pid=proc.pid)
