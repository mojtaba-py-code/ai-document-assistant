"""Service wiring for the identity area."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docassist.identity.admin import AdminService

if TYPE_CHECKING:
    from docassist.api.container import Container


def wire(container: Container) -> None:
    """Attach the administration service (``container.admin``)."""
    container.admin = AdminService(container)
