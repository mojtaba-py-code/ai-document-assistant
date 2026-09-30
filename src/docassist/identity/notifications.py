"""Outbound user notifications (password reset links, invitations).

Security rule: a reset token is a credential - it must never appear in logs. The
development sender therefore writes ``.eml`` files (owner-only permissions) into an outbox
directory instead of logging; production deployments plug in an SMTP/SES/Graph sender that
implements :class:`EmailSender`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from docassist.core.ids import uuid7


class EmailSender(Protocol):
    async def send(self, to: str, subject: str, body: str) -> None: ...


@dataclass
class MemoryEmailSender:
    """Test double: keeps messages in memory."""

    sent: list[tuple[str, str, str]] = field(default_factory=list)

    async def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append((to, subject, body))


class OutboxEmailSender:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    async def send(self, to: str, subject: str, body: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        message = EmailMessage()
        message["To"] = to
        message["From"] = "no-reply@docassist.local"
        message["Subject"] = subject
        message.set_content(body)
        path = self._dir / f"{uuid7().hex}.eml"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(bytes(message))
