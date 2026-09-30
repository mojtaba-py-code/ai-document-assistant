"""Service wiring for the ingestion area."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docassist.api.container import Container


def wire(container: Container) -> None:
    """Attach the ingestion pipeline (``container.pipeline``).

    The pipeline resolves ``container.storage`` and ``container.embeddings`` lazily, at use
    time, so it does not depend on the order in which other areas are wired.
    """
    from docassist.ingestion.pipeline import IngestionPipeline

    container.pipeline = IngestionPipeline(container)
