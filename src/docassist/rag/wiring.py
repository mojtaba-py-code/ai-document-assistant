"""Service wiring for the RAG area: ``container.answers`` and ``container.agent``.

The services keep a reference to the container and read ``container.search`` /
``container.llm`` at call time, so the wiring order of the areas does not matter and tests
can swap either collaborator on a live container.
"""

from __future__ import annotations

from typing import Protocol

from docassist.rag.agent import AgentService
from docassist.rag.ports import RagDependencies
from docassist.rag.service import AnswerService


class RagWiringTarget(RagDependencies, Protocol):
    answers: AnswerService
    agent: AgentService


def wire(container: RagWiringTarget) -> None:
    """Attach this area's services to the container."""
    container.answers = AnswerService(container)
    container.agent = AgentService(container)
