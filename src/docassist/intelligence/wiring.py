"""Service wiring for the intelligence area."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docassist.intelligence.service import IntelligenceService

if TYPE_CHECKING:
    from docassist.api.container import Container


def wire(container: Container) -> None:
    """Attach ``container.intelligence``.

    The LLM gateway, object storage and search service are looked up on the container at
    call time, so this area works whichever order the other areas are wired in and
    degrades to its deterministic paths when the LLM area is not configured.
    """
    container.intelligence = IntelligenceService(container)
