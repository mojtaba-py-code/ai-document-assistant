"""Build the configured embedding provider."""

from __future__ import annotations

from docassist.core.config import Settings
from docassist.embeddings.base import EmbeddingProvider
from docassist.embeddings.hashing import HashingEmbedder
from docassist.embeddings.openai_compat import OpenAICompatibleEmbedder
from docassist.security.ssrf import EgressPolicy


def create_embedding_provider(settings: Settings, egress: EgressPolicy) -> EmbeddingProvider:
    cfg = settings.embedding
    if cfg.provider == "hashing":
        return HashingEmbedder(dimensions=cfg.dimensions, model=cfg.model)
    if not cfg.base_url:
        raise ValueError("embedding.base_url is required for the openai_compatible provider")
    return OpenAICompatibleEmbedder(
        base_url=cfg.base_url,
        model=cfg.model,
        dimensions=cfg.dimensions,
        api_key=cfg.api_key.get_secret_value() if cfg.api_key else None,
        egress=egress,
        is_external=cfg.is_external,
        batch_size=cfg.batch_size,
        timeout_seconds=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
        send_dimensions=cfg.send_dimensions,
    )
